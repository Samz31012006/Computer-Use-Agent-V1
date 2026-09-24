"""Agent-loop behaviour against fakes: no desktop is touched."""
import threading
import time

from cua.agent.agent import Agent
from cua.safety.kill_switch import KillSwitch
from cua.safety.policy import needs_confirmation
from cua.types import DriverError, EngineResult, Step, TaskPlan, WinInfo


class FakeDriver:
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def launch(self, app, params=None):
        self.calls.append(("launch", app.key))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise DriverError("injected")

    def focus(self, hwnd): return True
    def close(self, hwnd): self.calls.append(("close", hwnd))
    def sleep(self, s): time.sleep(min(s, 0.01))
    def type_text(self, t): self.calls.append(("type", t))
    def hotkey(self, k): self.calls.append(("hotkey", k))


class FakePerception:
    def __init__(self): self.windows_ = []; self._on_time = None
    def match(self, app): return [w for w in self.windows_ if app.key in w.title.lower()]
    def windows(self): return self.windows_
    def foreground(self): return self.windows_[0] if self.windows_ else None


def mk(fail_first=0, **kw):
    d, p = FakeDriver(fail_first), FakePerception()
    real_launch = d.launch

    def launch(app, params=None):
        real_launch(app, params)
        p.windows_.append(WinInfo(len(p.windows_) + 100, app.key, "c", 1, app.key + ".exe"))
    d.launch = launch
    a = Agent(driver=d, perception=p, log_dir=None, **kw)
    return a, d, p


class OnePlan:
    def __init__(self, steps): self.steps = steps
    def plan(self, text, ctx): return TaskPlan(text, self.steps, "test")
    def replan(self, *a): return None


def test_happy_path_and_stage_timings():
    a, d, _ = mk()
    r = a.run("open notepad")
    assert r.ok and d.calls == [("launch", "notepad")]
    assert {"plan", "select", "act", "verify"} <= set(r.stage_ms)


def test_retry_recovers_from_transient_driver_failure():
    a, d, _ = mk(fail_first=1)
    r = a.run("open notepad")
    assert r.ok and r.recovered and r.steps[0].attempts == 2


def test_persistent_failure_reports_failure():
    a, _, _ = mk(fail_first=99)
    r = a.run("open notepad")
    assert not r.ok and r.reason == "step_failed"


def test_no_plan_is_reported_not_guessed():
    a, d, _ = mk()
    r = a.run("write me a poem")
    assert r.reason == "no_plan" and d.calls == []


def test_destructive_requires_confirmation_and_fails_closed():
    assert needs_confirmation(Step("close_app", {"name": "notepad"}))
    assert needs_confirmation(Step("hotkey", {"keys": "Alt+F4"}))
    assert needs_confirmation(Step("click_ui", {"name": "Delete all"}))
    assert not needs_confirmation(Step("open_app", {"name": "notepad"}))
    a, d, p = mk()               # default confirm = deny_all
    p.windows_.append(WinInfo(1, "notepad", "c", 1, "notepad.exe"))
    r = a.run("close notepad")
    assert r.reason == "declined" and d.calls == []


def test_confirmed_close_runs():
    asked = []
    a, d, p = mk(confirm=lambda why: asked.append(why) or True)
    p.windows_.append(WinInfo(1, "notepad", "c", 1, "notepad.exe"))
    a.run("close notepad")
    assert asked and ("close", 1) in d.calls


def test_non_idempotent_step_not_repeated_after_failed_verify():
    a, d, p = mk()
    a.verifier.default_timeout = 0.1
    p.windows_.append(WinInfo(1, "notepad", "c", 1, "notepad.exe"))
    a.planner = OnePlan([Step("hotkey", {"keys": "ctrl+s", "app": "notepad"},
                              {"kind": "window_title", "any_of": ["never"], "timeout": 0.1})])
    r = a.run("x")
    # own_pids guard may refuse (fake pid=1 is not ours, so it types once) - either way exactly <=1 press
    assert len([c for c in d.calls if c[0] == "hotkey"]) <= 1 and not r.ok


def test_kill_hotkey_flag_aborts_between_steps():
    ks = KillSwitch(); ks.hard_exit = False
    a, d, _ = mk(kill=ks)
    a.driver.sleep = lambda s: None
    a.planner = OnePlan([Step("wait", {"seconds": 0}), Step("open_app", {"name": "notepad"})])
    ks.trigger()
    r = a.run("x")
    assert r.reason.startswith("aborted") and d.calls == []
    assert not ks.event.is_set()          # acknowledged and re-armed


def test_kill_watchdog_hard_exits_when_unacknowledged(monkeypatch):
    import cua.safety.kill_switch as s
    hit = threading.Event()
    monkeypatch.setattr(s.os, "_exit", lambda code: hit.set())
    ks = KillSwitch(grace=0.1)
    ks.trigger()
    assert hit.wait(1.0)
