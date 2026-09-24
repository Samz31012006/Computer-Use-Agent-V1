"""AgentService: the single authoritative Buddy core shared by every interface.

Before this module existed, `build_agent()` was called independently by the CLI, the laptop UI, and the phone
server -- each got its own Agent, its own model runtime (its own `llama-server` process), its own kill switch.
Running the laptop app and the phone server at the same time meant two agents fighting over the same keyboard
and mouse, with no shared task state.

AgentService fixes that by being the ONE place that owns:
  - the Agent (and therefore the ChainPlanner, the LLM runtime, ToolRegistry, StepRunner)
  - the KillSwitch (one Ctrl+Alt+Q registration, whichever interface is running)
  - task state (TaskRecord per submission, addressable by task_id)
  - a single FIFO worker thread, so only one task ever touches the keyboard/mouse/windows at a time
  - event broadcasting to every subscriber (laptop UI, phone WebSocket, tests), all reading the same stream

Interfaces (BuddyApp, AgentAPI) submit text and observe state through this object; they never construct an
Agent themselves.
"""
from __future__ import annotations

import threading
import time
import uuid
from queue import Empty, Queue
from typing import Any, Callable

from cua.agent.agent import Agent
from cua.agent.factory import build_agent
from cua.config import Config
from cua.safety.kill_switch import KillSwitch
from cua.safety.policy import deny_all
from cua.types import TaskResult

_ACTIVE_STATUSES = {"received", "planning", "executing"}


class TaskRecord:
    __slots__ = ("id", "text", "status", "result", "plan", "created", "source")

    def __init__(self, text: str, source: str = "unknown"):
        self.id = "task_" + uuid.uuid4().hex[:8]
        self.text = text
        self.status = "received"
        self.result: TaskResult | None = None
        self.plan = None                  # raw TaskPlan once planning completes; in-process consumers only
        self.created = time.time()
        self.source = source              # "laptop" | "phone" | ... informational only, never used for auth

    def to_dict(self) -> dict:
        d = {"id": self.id, "text": self.text, "status": self.status, "created": self.created, "source": self.source}
        if self.result:
            d["result"] = {
                "ok": self.result.ok, "reason": self.result.reason, "total_ms": self.result.total_ms,
                "steps": [{"action": s.step.action, "ok": s.ok, "detail": s.detail, "ms": s.ms}
                          for s in self.result.steps],
                "stage_ms": self.result.stage_ms, "planner": self.result.planner, "router_hit": self.result.router_hit,
            }
        return d


class AgentService:
    """Builds exactly one Agent and serializes every task through a single worker thread.

    `agent=` / `kill=` let tests inject fakes (same convention as `Agent(driver=..., perception=...)` elsewhere
    in this codebase) without spinning up a real model, a real kill-switch hotkey registration, or touching
    the real desktop.
    """

    def __init__(self, cfg: Config | None = None, confirm: Callable[[str], bool] = deny_all,
                 folders: dict | None = None, log_dir="logs", runtime=None,
                 agent: Agent | None = None, kill: KillSwitch | None = None):
        self.kill = kill or KillSwitch().start()
        if agent is not None:
            self.agent = agent
            self.agent.kill = self.kill
        else:
            self.agent = build_agent(cfg=cfg, kill=self.kill, confirm=confirm, on_event=self._on_agent_event,
                                     folders=folders, log_dir=log_dir, runtime=runtime)
        self.agent.on_event = self._on_agent_event

        self._tasks: dict[str, TaskRecord] = {}
        self._queue: list[TaskRecord] = []
        self._current: TaskRecord | None = None
        # Carried forward between tasks -- active app, last target, last outcome -- so "open notepad" then,
        # separately, "type hello" resolves to Notepad. The SAME dict is handed to every Agent.run() call, and
        # skills.py already writes last_app/last_hwnd/folder/... into it during execution, so this needs no
        # new plumbing beyond passing it through. Deliberately just a few fixed keys, not a growing history.
        self._session_context: dict = {}
        self._lock = threading.Lock()
        self._subscribers: list[Queue] = []
        self._job_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True, name="agent-service-worker")
        self._worker.start()

    # ---- submission / state ---------------------------------------------------------------
    def submit_task(self, text: str, source: str = "unknown") -> TaskRecord:
        rec = TaskRecord(text, source=source)
        with self._lock:
            self._tasks[rec.id] = rec
            self._queue.append(rec)
        self._broadcast({"type": "received", "id": rec.id, "text": text, "source": source, "time": time.time()})
        self._job_event.set()
        return rec

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def is_busy(self) -> bool:
        with self._lock:
            return self._current is not None or bool(self._queue)

    def cancel_task(self, task_id: str) -> bool:
        """Stop a queued task before it starts, or abort it (via the same path as the physical hotkey) if it is
        the one currently executing. Returns False if the id is unknown or the task already finished."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None or rec.status not in _ACTIVE_STATUSES:
                return False
            is_current = rec is self._current
            if not is_current:
                try:
                    self._queue.remove(rec)
                except ValueError:
                    pass
                rec.status = "cancelled"
        if is_current:
            self.kill.trigger()           # identical to a Ctrl+Alt+Q press; task_end will mark it cancelled
        else:
            self._broadcast({"type": "cancelled", "id": rec.id, "time": time.time()})
        return True

    # ---- pub/sub ----------------------------------------------------------------------------
    def subscribe(self) -> Queue:
        q: Queue = Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    # ---- worker: the ONE thread allowed to call agent.run() ---------------------------------
    def _run_worker(self):
        while True:
            self._job_event.wait()
            self._job_event.clear()
            while True:
                with self._lock:
                    rec = None
                    while self._queue:
                        cand = self._queue.pop(0)
                        if cand.status != "cancelled":     # dequeued by cancel_task while still queued
                            rec = cand
                            break
                    self._current = rec
                if rec is None:
                    break
                try:
                    self.agent.run(rec.text, context=self._session_context)  # status/result set by _on_agent_event
                except Exception as e:                      # Agent.run() already catches its own errors; this
                    rec.status = "failed"                    # is a last-resort net so the worker never dies
                    self._broadcast({"type": "failed", "id": rec.id, "reason": repr(e), "time": time.time()})
                with self._lock:
                    self._current = None

    # ---- Agent.on_event sink: the single source of task-lifecycle + step events -------------
    def _on_agent_event(self, kind: str, **kw):
        rec = self._current
        if rec is None:
            return
        now = time.time()
        if kind == "task_start":
            rec.status = "planning"
            self._broadcast({"type": "planning", "id": rec.id, "time": now})
        elif kind == "plan":
            plan = kw["plan"]
            rec.plan = plan
            rec.status = "executing"
            self._broadcast({"type": "executing", "id": rec.id, "time": now, "source": plan.source,
                             "steps": [{"action": s.action, "args": s.args} for s in plan.steps]})
        elif kind == "step_start":
            step = kw["step"]
            self._broadcast({"type": "step_start", "id": rec.id, "index": kw["index"], "time": now,
                             "action": step.action, "args": step.args})
        elif kind == "step_end":
            sr = kw.get("result")
            self._broadcast({"type": "step_end", "id": rec.id, "index": kw.get("index"), "time": now,
                             "ok": getattr(sr, "ok", None), "detail": getattr(sr, "detail", "")})
        elif kind == "confirm":
            self._broadcast({"type": "confirm", "id": rec.id, "why": kw.get("why"), "time": now})
        elif kind == "task_end":
            result: TaskResult = kw["result"]
            rec.result = result
            rec.status = "cancelled" if result.reason.startswith("aborted") else ("completed" if result.ok else "failed")
            # a few fixed keys only, overwritten every task -- not a growing conversation history
            self._session_context["last_task_text"] = rec.text
            self._session_context["last_task_ok"] = result.ok
            self._broadcast({"type": rec.status, "id": rec.id, "time": now, "ok": result.ok, "reason": result.reason,
                             "total_ms": result.total_ms,
                             "steps": [{"action": s.step.action, "ok": s.ok, "detail": s.detail} for s in result.steps]})
