"""Shared data types. Plain dataclasses, no framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class AgentError(Exception):
    """Base error for anything the agent can recover from or report."""


class DriverError(AgentError):
    """An OS-level action failed."""


class Aborted(AgentError):
    """Kill hotkey pressed or the user declined a confirmation."""


class NeedsClarification(AgentError):
    """The router found more than one plausible correction for something the user typed (e.g. a misspelled
    app/site name) and won't guess which one was meant."""

    def __init__(self, message: str, options: list[str]):
        super().__init__(message)
        self.options = options


@dataclass
class Step:
    action: str                      # launch_app, open_url, search_web, ...
    args: dict[str, Any] = field(default_factory=dict)
    verify: dict[str, Any] | None = None   # predicate spec, see verifier.py
    note: str = ""


@dataclass
class TaskPlan:
    goal: str
    steps: list[Step]
    source: str = "router"           # router | llm | ...
    success_condition: str | None = None
    max_steps: int = 12


@dataclass
class EngineResult:
    ok: bool
    detail: str = ""
    retryable: bool = True           # False for "not found": repeating the same lookup cannot help


@dataclass
class VerifyResult:
    ok: bool
    detail: str = ""
    waited_ms: float = 0.0


@dataclass
class StepResult:
    step: Step
    ok: bool
    engine: str = ""
    attempts: int = 0
    recovered: bool = False          # succeeded only after a retry or fallback engine
    detail: str = ""
    ms: float = 0.0


@dataclass
class TaskResult:
    text: str
    ok: bool
    plan: TaskPlan | None
    steps: list[StepResult] = field(default_factory=list)
    reason: str = ""                 # no_plan | step_failed | aborted | declined | ok
    stage_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    planner: str = ""                # router | llm | none
    router_hit: bool | None = None
    planner_calls: list = field(default_factory=list)
    replans: int = 0
    retries: int = 0
    metrics: dict = field(default_factory=dict)   # cpu_pct, rss_mb, avail_mb

    @property
    def recovered(self) -> bool:
        return self.ok and any(s.recovered for s in self.steps)

    @property
    def had_failure(self) -> bool:
        return any(s.recovered or not s.ok for s in self.steps)


@dataclass
class WinInfo:
    hwnd: int
    title: str
    cls: str
    pid: int
    proc: str


@dataclass
class ElementRef:
    name: str
    role: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom
    native: Any = None               # backend handle (uiautomation Control)
