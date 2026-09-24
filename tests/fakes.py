"""Shared fakes. Nothing here touches the desktop or needs a model."""
from __future__ import annotations

import time

from cua.models.runtime import GenResult, ModelUnavailable
from cua.types import DriverError, EngineResult, WinInfo


class FakeDriver:
    def __init__(self, fail_launch=0):
        self.calls, self.fail_launch = [], fail_launch

    def launch(self, app, params=None):
        self.calls.append(("launch", app.key))
        if self.fail_launch > 0:
            self.fail_launch -= 1
            raise DriverError("injected")
        self.on_launch(app)

    def on_launch(self, app):
        pass

    def focus(self, hwnd): return True
    def close(self, hwnd): self.calls.append(("close", hwnd))
    def minimize(self, hwnd): self.calls.append(("minimize", hwnd))
    def maximize(self, hwnd): self.calls.append(("maximize", hwnd))
    def sleep(self, s): time.sleep(min(s, 0.005))
    def type_text(self, t): self.calls.append(("type", t))
    def hotkey(self, k): self.calls.append(("hotkey", k))
    def click(self, x, y, button="left", double=False): self.calls.append(("click", x, y, button, double))
    def move(self, x, y): self.calls.append(("move", x, y))
    def scroll(self, a, x=None, y=None): self.calls.append(("scroll", a))
    def open_path(self, p, app=None): self.calls.append(("open_path", str(p)))
    def open_uri(self, u, browser=None): self.calls.append(("open_uri", u))
    def reveal(self, p): self.calls.append(("reveal", str(p)))


class FakePerception:
    def __init__(self):
        self.windows_: list[WinInfo] = []
        self._on_time = None
        self.text = ""

    def match(self, app): return [w for w in self.windows_ if app.key in w.title.lower()]
    def windows(self): return self.windows_
    def foreground(self): return self.windows_[0] if self.windows_ else None
    def find_element(self, hwnd, name, timeout=0.0): return None
    def read_text(self, hwnd): return self.text
    def element_text(self, hwnd, name): return ""
    def inspect(self, hwnd, limit=40): return []


def make_agent(fail_launch=0, **kw):
    """Agent wired to fakes; launching an app makes a window of that name appear."""
    from cua.agent.agent import Agent
    d, p = FakeDriver(fail_launch), FakePerception()
    d.on_launch = lambda app: p.windows_.insert(0, WinInfo(len(p.windows_) + 100, app.key, "c", 1, app.key + ".exe"))
    kw.setdefault("log_dir", None)
    return Agent(driver=d, perception=p, **kw), d, p


class FakeRuntime:
    """Scripted model: returns queued outputs in order (repeating the last one)."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts, self.schemas = [], []

    def generate(self, prompt, schema=None, max_tokens=200, abort=None):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return GenResult(out, 12.0, 300, 40, 5.0, 8.0)

    def unload(self): pass
    def status(self): return {"loaded": True}

    @property
    def calls(self): return len(self.prompts)


class ScriptedEngine:
    """Engine with scripted per-call results, for recovery tests."""

    def __init__(self, name, actions, results, repair=None):
        self.name, self.actions, self.results, self.calls = name, set(actions), list(results), 0
        if repair is not None:
            self.repair = repair

    def can_handle(self, step): return step.action in self.actions

    def execute(self, step, ctx):
        self.calls += 1
        r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(r, Exception):
            raise r
        return r


OK = EngineResult(True, "ok")
BAD = EngineResult(False, "nope")
PLAN_OPEN_CHROME = ('{"goal":"open chrome","steps":[{"action":"open_app","target":"chrome"}]}')
