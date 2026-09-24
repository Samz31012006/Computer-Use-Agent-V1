"""Orchestrator. plan -> [confirm -> run step with recovery ladder]* -> bounded replan -> result.

The LLM (if any) is called at most `1 + max_replans` times per task, never per action."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from cua.actions.registry import ToolRegistry, default_registry
from cua.actions.skills import SkillEngine
from cua.actions.uia import UIAEngine
from cua.agent.recovery import Budget, BudgetExceeded, StepRunner
from cua.agent.router import RouterPlanner
from cua.agent.state import Ctx, TaskState
from cua.computer.driver import WindowsDriver
from cua.config import Config
from cua.logging.events import Logger, ResourceProbe
from cua.perception.uia import UIAPerception
from cua.safety.kill_switch import KillSwitch
from cua.safety.policy import deny_all, needs_confirmation
from cua.types import Aborted, NeedsClarification, Step, StepResult, TaskPlan, TaskResult
from cua.verification.verifier import PollingVerifier

EventFn = Callable[..., None]


class Agent:
    def __init__(self, planner=None, engines=None, driver=None, perception=None, verifier=None,
                 kill: KillSwitch | None = None, confirm=deny_all, folders: dict | None = None,
                 log_dir: str | Path | None = "logs", registry: ToolRegistry | None = None,
                 config: Config | None = None, on_event: EventFn | None = None):
        self.cfg = config or Config()
        self.kill = kill
        self.registry = registry or default_registry()
        self.driver = driver or WindowsDriver(kill_check=kill.check if kill else (lambda: None))
        self.perception = perception or UIAPerception()
        self.planner = planner or RouterPlanner(self.registry)
        self.engines = engines or [SkillEngine(), UIAEngine()]
        self.verifier = verifier or PollingVerifier()
        self.confirm = confirm
        self.folders = folders or {}
        self.log_dir = Path(log_dir) if log_dir else None
        self.on_event: EventFn = on_event or (lambda *a, **k: None)

    # ------------------------------------------------------------------------------------------
    def run(self, text: str, context: dict | None = None) -> TaskResult:
        """`context`, if given, is used AS the task's scratch dict (not copied) -- the caller keeps the same
        object across calls to carry lightweight state forward (active app, last target, ...) between tasks.
        See AgentService, which is the one caller that does this in production."""
        log = Logger(self.log_dir / "tasks.jsonl" if self.log_dir else None)
        if hasattr(self.perception, "_on_time"):
            self.perception._on_time = lambda ms: log.stage_ms.__setitem__("perceive", log.stage_ms["perceive"] + ms)
        state = TaskState(text=text, context=context if context is not None else {})
        ctx = Ctx(self.driver, self.perception, log, state, self.kill, self.confirm, self.folders)
        budget = Budget(self.cfg.max_steps, self.cfg.max_replans, self.cfg.max_retries, self.cfg.task_timeout_s)
        runner = StepRunner(self.engines, self.verifier, self.registry)
        probe, t0 = ResourceProbe().start(), time.perf_counter()
        if self.kill:
            self.kill.busy.set()
        result = TaskResult(text, False, None)
        self.on_event("task_start", text=text)
        try:
            with log.span("plan"):
                plan = self.planner.plan(text, state.context)
            state.plan = result.plan = plan
            if plan is None:
                result.reason = "no_plan"
            elif not self._approve_ai_plan(plan, ctx):
                result.reason = "declined"
            else:
                self.on_event("plan", plan=plan)
                result.reason = self._execute(plan.steps, ctx, result, runner, budget)
                while result.reason == "step_failed" and budget.replans < budget.max_replans:
                    failed = result.steps[-1]
                    budget.replans += 1
                    with log.span("replan"):
                        new = self.planner.replan(text, state.context, failed.step, failed.detail)
                    if new is None:
                        break
                    log.event("replanned", steps=len(new.steps), why=failed.detail[:160])
                    if not self._approve_ai_plan(new, ctx):
                        result.reason = "declined"
                        break
                    self.on_event("plan", plan=new)
                    result.plan = state.plan = new
                    result.reason = self._execute(new.steps, ctx, result, runner, budget)
                    if result.reason == "ok":
                        failed.recovered = True
            result.ok = result.reason == "ok"
        except Aborted as e:
            result.reason = "aborted: " + str(e)
            log.event("aborted", why=str(e))
        except NeedsClarification as e:
            result.reason = f"clarify: {e}"
            log.event("clarify_needed", options=e.options)
        except BudgetExceeded as e:
            result.reason = e.reason
            log.event("budget_exceeded", which=e.reason)
        finally:
            if self.kill:
                if self.kill.event.is_set():
                    self.kill.acknowledge()
                self.kill.busy.clear()
        return self._finish(result, ctx, log, budget, probe, t0)

    # ------------------------------------------------------------------------------------------
    def _approve_ai_plan(self, plan: TaskPlan, ctx: Ctx) -> bool:
        """Model-made plans are shown to the user first (config.confirm_llm_plans). Router plans never ask."""
        if plan.source != "llm" or not self.cfg.confirm_llm_plans:
            return True
        summary = "; ".join(f"{s.action}({', '.join(str(v) for v in s.args.values())})"[:48] for s in plan.steps[:4])
        ctx.log.event("confirm_requested", action="ai_plan", plan=summary)
        self.on_event("plan", plan=plan)
        return bool(self.confirm and self.confirm("run this AI plan: " + summary))

    def _execute(self, steps: list[Step], ctx: Ctx, result: TaskResult, runner: StepRunner, budget: Budget) -> str:
        for i, step in enumerate(steps):
            ctx.check_kill()
            fg = ctx.perception.foreground()
            why = needs_confirmation(step, self.registry, fg.title if fg else None)
            if why:                                    # applies to every plan, whatever produced it
                ctx.log.event("confirm_requested", action=why)
                self.on_event("confirm", why=why)
                if not (self.confirm and self.confirm(why)):
                    result.steps.append(StepResult(step, False, detail="declined by user"))
                    return "declined"
            self.on_event("step_start", index=i, step=step)
            sr = runner.run(step, ctx, budget)
            result.steps.append(sr)
            self.on_event("step_end", index=i, result=sr)
            if not sr.ok:
                return "step_failed"
        return "ok"

    def _finish(self, result: TaskResult, ctx: Ctx, log: Logger, budget: Budget, probe: ResourceProbe,
                t0: float) -> TaskResult:
        result.total_ms = (time.perf_counter() - t0) * 1000
        result.stage_ms = {k: round(v, 1) for k, v in log.stage_ms.items()}
        c = ctx.state.context
        result.planner, result.router_hit = c.get("planner", "router"), c.get("router_hit")
        result.planner_calls = c.get("planner_calls", [])
        result.replans, result.retries = budget.replans, budget.retries
        result.metrics = probe.stop()
        if c.get("planner_error"):
            result.metrics["planner_error"] = c["planner_error"]
        s = result.stage_ms
        log.flush({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "command": result.text, "router_hit": result.router_hit,
            "planner": result.planner, "planner_calls": result.planner_calls,
            "planner_ms": s.get("plan", 0) + s.get("replan", 0),
            "action_ms": round(s.get("select", 0) + s.get("act", 0) + s.get("settle", 0), 1),
            "verify_ms": s.get("verify", 0), "perceive_ms": s.get("perceive", 0), "total_ms": round(result.total_ms, 1),
            **result.metrics, "ok": result.ok, "reason": result.reason, "retries": result.retries,
            "replans": result.replans, "recovered": result.recovered, "steps": len(result.steps),
            "error": c.get("planner_error"),
        })
        self.on_event("task_end", result=result)
        return result
