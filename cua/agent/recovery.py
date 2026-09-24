"""Budgets and the per-step recovery ladder.

For every step:
  1. run the tool's first engine, then verify
  2. verification/action failed and the step is safe to repeat -> retry once (never repeat typing/clicking/closing)
  3. non-repeatable step whose check failed -> ask an engine to *repair* (state-aware: inspect, then fix what's missing)
  4. try the next engine that handles the tool (alternative deterministic mechanism, then OCR/UIA fallbacks)
  5. give up on the step; the Agent may then replan (bounded) or report failure
Every level is bounded by the task Budget, so no path can loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from cua.actions.base import ActionEngine
from cua.actions.registry import ToolRegistry
from cua.types import Aborted, AgentError, Step, StepResult


class BudgetExceeded(AgentError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason            # "max_steps" | "timeout"


@dataclass
class Budget:
    max_steps: int = 12
    max_replans: int = 1
    max_retries: int = 1
    timeout_s: float = 90.0
    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    replans: int = 0
    retries: int = 0

    def check_time(self):
        if time.monotonic() - self.started > self.timeout_s:
            raise BudgetExceeded("timeout")

    def tick_step(self):
        self.check_time()
        self.steps += 1
        if self.steps > self.max_steps:
            raise BudgetExceeded("max_steps")


class StepRunner:
    def __init__(self, engines: list[ActionEngine], verifier, registry: ToolRegistry, budget_of=None):
        self.engines, self.verifier, self.registry = engines, verifier, registry

    def run(self, step: Step, ctx, budget: Budget) -> StepResult:
        t0 = time.perf_counter()
        budget.tick_step()
        engines = [e for e in self.engines if e.can_handle(step)]
        idem = self.registry.is_idempotent(step.action)
        sr = StepResult(step, False, detail="no engine handles this action")
        for eng in engines:
            action_done = False
            for attempt in range(1 + budget.max_retries):
                ctx.check_kill()
                budget.check_time()
                sr.attempts += 1
                sr.engine = eng.name
                er = self._execute(eng, step, ctx, sr)
                if er is not None and er.ok:
                    action_done = True
                    ok, detail = self._verify(step, ctx)
                    if ok:
                        return self._done(sr, detail, t0)
                    sr.detail = detail
                    if not idem:
                        if self._repair(step, ctx, sr):          # step 3
                            return self._done(sr, sr.detail, t0)
                        break                                    # never repeat a non-idempotent action
                elif er is not None:
                    sr.detail = er.detail
                    if not er.retryable:
                        break                                    # straight to the next engine
                if attempt < budget.max_retries:
                    budget.retries += 1
                    ctx.log.event("retry", action=step.action, engine=eng.name, attempt=attempt + 1, why=sr.detail)
                    ctx.sleep(0.2)
            if action_done and not idem:
                break            # the action ran once; a different engine must not run it again
            ctx.log.event("fallback_engine", action=step.action, from_engine=eng.name, why=sr.detail)
        sr.ms = (time.perf_counter() - t0) * 1000
        return sr

    def _execute(self, eng, step, ctx, sr):
        try:
            with ctx.log.span("engine", engine=eng.name, action=step.action):
                return eng.execute(step, ctx)
        except Aborted:
            raise
        except BudgetExceeded:
            raise
        except Exception as e:      # engine bug / COM error: a failed attempt, never a crash
            sr.detail = f"{type(e).__name__}: {e}"
            return None

    def _verify(self, step, ctx) -> tuple[bool, str]:
        with ctx.log.span("verify"):
            vr = self.verifier.check(step.verify, ctx)
        return vr.ok, vr.detail

    def _repair(self, step, ctx, sr) -> bool:
        for eng in self.engines:
            fix = getattr(eng, "repair", None)
            if fix is None:
                continue
            try:
                with ctx.log.span("repair"):
                    res = fix(step, ctx)
            except Aborted:
                raise
            except Exception:
                continue
            if res is not None and res.ok:
                ok, detail = self._verify(step, ctx)
                if ok:
                    sr.engine, sr.detail = eng.name, f"repaired: {res.detail}"
                    sr.attempts += 1
                    ctx.log.event("repaired", action=step.action, engine=eng.name, note=res.detail)
                    return True
        return False

    @staticmethod
    def _done(sr: StepResult, detail: str, t0: float) -> StepResult:
        sr.ok, sr.detail = True, detail
        sr.recovered = sr.attempts > 1
        sr.ms = (time.perf_counter() - t0) * 1000
        return sr
