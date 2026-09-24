"""Safety policy: decides whether a step needs the user's explicit OK. The planner cannot bypass it: the Agent
consults this for every step, whatever produced the plan.

Sources of risk, in order:
  1. the tool's declared safety level / data-dependent rule (ToolSpec.confirm_if)   e.g. close_app, overwriting a file
  2. destructive key combos                                                         e.g. Alt+F4, Delete
  3. clicking controls named Delete / Remove / Uninstall / Restart ...
  4. changing anything inside system-configuration windows                          e.g. Settings, Task Manager
There is deliberately no shell/run_command tool; unknown actions fail closed.
"""
from __future__ import annotations

import re

from cua.actions.registry import ToolRegistry, default_registry
from cua.types import Step

_DESTRUCTIVE_HOTKEYS = {"alt+f4", "delete", "shift+delete", "ctrl+shift+delete", "ctrl+shift+esc"}
_DESTRUCTIVE_NAMES = re.compile(
    r"\b(delete|remove|uninstall|format|erase|reset|discard|don'?t save|sign out|log ?out|shut ?down|restart|empty)\b", re.I)
_SYSTEM_APPS = {"settings", "control panel", "task manager", "registry editor", "device manager", "windows security"}
_SYSTEM_TITLES = re.compile(r"^(settings|control panel|task manager|registry editor|windows security|device manager|"
                            r"services)\b", re.I)
_UI_CHANGING = {"click_ui", "type_into_ui", "click", "double_click"}


def needs_confirmation(step: Step, registry: ToolRegistry | None = None, foreground_title: str | None = None) -> str | None:
    """Return a human-readable reason if this step must be confirmed, else None."""
    reason = (registry or default_registry()).confirm_reason(step)
    if reason:
        return reason
    a, g = step.action, step.args
    if a in ("hotkey", "press_key"):
        keys = (g.get("keys") or g.get("key") or "").lower().replace(" ", "")
        if keys in _DESTRUCTIVE_HOTKEYS:
            return f"press {keys}"
    if a in ("click_ui", "type_into_ui") and _DESTRUCTIVE_NAMES.search(g.get("name", "")):
        return f"click '{g['name']}'"
    if a in _UI_CHANGING:
        app = (g.get("app") or "").lower()
        if app in _SYSTEM_APPS or (foreground_title and _SYSTEM_TITLES.match(foreground_title)):
            return "change a system setting" + (f" ('{g['name']}')" if g.get("name") else "")
    return None


def cli_confirm(prompt: str) -> bool:
    try:
        return input(f"  ⚠ Confirm: {prompt}? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def deny_all(prompt: str) -> bool:      # default for non-interactive callers: fail closed
    return False
