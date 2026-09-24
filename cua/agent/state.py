"""TaskState (what the agent knows mid-task) and Ctx (everything an engine or verifier may touch)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

from cua.logging.events import Logger
from cua.safety.kill_switch import KillSwitch
from cua.types import Aborted, StepResult, TaskPlan


@dataclass
class TaskState:
    text: str = ""
    plan: TaskPlan | None = None
    results: list[StepResult] = field(default_factory=list)
    # scratch shared between steps: last_app, last_hwnd, found_file, closed_hwnd, ...
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ctx:
    driver: Any
    perception: Any
    log: Logger
    state: TaskState
    kill: KillSwitch | None = None
    confirm: Callable[[str], bool] | None = None
    folders: dict[str, Path] = field(default_factory=dict)   # overrides for known folders

    def check_kill(self):
        if self.kill:
            self.kill.check()

    def sleep(self, seconds: float):
        self.driver.sleep(seconds)


def own_pids() -> set[int]:
    """This process and its ancestors (the terminal / IDE hosting the agent). Never type into or close these."""
    pids, p = set(), psutil.Process(os.getpid())
    try:
        while p:
            pids.add(p.pid)
            p = p.parent()
    except psutil.Error:
        pass
    return pids
