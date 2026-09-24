import json
import time

from cua.actions.registry import default_registry
from cua.agent.planner import ChainPlanner
from cua.config import Config
from cua.safety.policy import needs_confirmation
from cua.types import DriverError, EngineResult, Step, TaskPlan, WinInfo
from tests.fakes import BAD, OK, ScriptedEngine, make_agent

R = default_registry()


class Script:
    """Planner returning fixed plans; counts calls."""

    def __init__(self, first, again=None):
        self.first, self.again, self.replans = first, again, 0

    def plan(self, text, ctx): return TaskPlan(text, self.first, "test")

    def replan(self, text, ctx, failed, err):
        self.replans += 1
        return TaskPlan(text, self.again, "test") if self.again is not None else None


def steps(*specs):
    return [R.make_step(n, **kw) for n, kw in specs]


# ---- budgets ---------------------------------------------------------------------
def test_max_steps_enforced():
    a, d, _ = make_agent(config=Config(max_steps=3))
    a.planner = Script(steps(*[("wait", {"seconds": 0})] * 6))
    r = a.run("x")
    assert r.reason == "max_steps" and len(r.steps) == 3


def test_max_replans_enforced_and_no_infinite_loop():
    a, d, _ = make_agent(fail_launch=999, config=Config(max_replans=1))
    bad = steps(("open_app", {"name": "chrome"}))
    a.planner = Script(bad, bad)
    r = a.run("x")
    assert r.reason == "step_failed" and a.planner.replans == 1 and r.replans == 1
    # each failing step is tried (1 + max_retries) times: 2 plans x 2 attempts
    assert len([c for c in d.calls if c[0] == "launch"]) == 4


def test_replan_recovers_a_failed_plan():
    a, d, _ = make_agent(fail_launch=1, config=Config(max_retries=0))   # first launch fails, no retry -> replan
    a.planner = Script(steps(("open_app", {"name": "chrome"})), steps(("open_app", {"name": "chrome"})))
    r = a.run("x")
    assert r.ok and r.replans == 1 and r.recovered


def test_task_timeout():
    class Slow:
        name = "slow"
        def can_handle(self, s): return True
        def execute(self, s, ctx): time.sleep(0.08); return OK
    a, _, _ = make_agent(config=Config(task_timeout_s=0.2), engines=[Slow()])
    a.planner = Script(steps(*[("wait", {"seconds": 0})] * 10))
    r = a.run("x")
    assert r.reason == "timeout" and 0 < len(r.steps) < 10


# ---- recovery ladder ------------------------------------------------------------------
def test_idempotent_step_retried_once():
    e = ScriptedEngine("a", {"open_app"}, [BAD, OK])
    a, _, p = make_agent(engines=[e])
    p.windows_.append(WinInfo(1, "chrome", "c", 1, "chrome.exe"))
    a.planner = Script(steps(("open_app", {"name": "chrome"})))
    r = a.run("x")
    assert r.ok and r.steps[0].attempts == 2 and r.steps[0].recovered and e.calls == 2


def test_retries_are_bounded():
    e = ScriptedEngine("a", {"open_app"}, [BAD])
    a, _, _ = make_agent(engines=[e], config=Config(max_retries=1))
    a.planner = Script(steps(("open_app", {"name": "chrome"})))
    r = a.run("x")
    assert not r.ok and e.calls == 2


def test_falls_back_to_alternative_engine():
    first = ScriptedEngine("primary", {"open_app"}, [DriverError("boom")])
    second = ScriptedEngine("fallback", {"open_app"}, [OK])
    a, _, p = make_agent(engines=[first, second])
    p.windows_.append(WinInfo(1, "chrome", "c", 1, "chrome.exe"))
    a.planner = Script(steps(("open_app", {"name": "chrome"})))
    r = a.run("x")
    assert r.ok and r.steps[0].engine == "fallback" and r.steps[0].recovered


def test_non_idempotent_step_is_never_repeated_but_can_be_repaired():
    fixed = []
    def repair(step, ctx):
        fixed.append(1)
        return EngineResult(True, "fixed")
    e = ScriptedEngine("a", {"type_text"}, [OK], repair=repair)
    a, _, p = make_agent(engines=[e])
    # verification passes only after repair flips the text in the fake window
    p.windows_.append(WinInfo(1, "x", "c", 1, "x.exe"))
    p.text = ""
    real = e.repair
    e.repair = lambda s, c: (setattr(p, "text", "hello"), real(s, c))[1]
    a.planner = Script([R.make_step("type_text", text="hello")])
    a.verifier.default_timeout = 0.1
    r = a.run("x")
    assert r.ok and e.calls == 1 and fixed == [1]


def test_non_idempotent_failed_check_without_repair_fails_after_one_execution():
    e = ScriptedEngine("a", {"type_text"}, [OK])
    a, _, p = make_agent(engines=[e])
    p.windows_.append(WinInfo(1, "x", "c", 1, "x.exe"))
    a.verifier.default_timeout = 0.1
    a.planner = Script([R.make_step("type_text", text="hello")])
    r = a.run("x")
    assert not r.ok and e.calls == 1


def test_engine_exception_never_crashes_the_loop():
    e = ScriptedEngine("a", {"open_app"}, [RuntimeError("COM exploded")])
    a, _, _ = make_agent(engines=[e])
    a.planner = Script(steps(("open_app", {"name": "chrome"})))
    r = a.run("x")
    assert not r.ok and "COM exploded" in r.steps[0].detail


# ---- logging -------------------------------------------------------------------------
def test_structured_task_record_written(tmp_path):
    a, _, _ = make_agent(log_dir=tmp_path)
    a.run("open notepad")
    rec = [json.loads(l) for l in (tmp_path / "tasks.jsonl").read_text().splitlines() if '"task"' in l][-1]
    for k in ("ts", "command", "router_hit", "planner", "planner_ms", "action_ms", "verify_ms", "total_ms", "cpu_pct",
              "rss_mb", "ok", "reason", "retries", "replans"):
        assert k in rec
    assert rec["command"] == "open notepad" and rec["router_hit"] is None or rec["planner"] == "router"


# ---- safety ---------------------------------------------------------------------------
def test_unknown_action_fails_closed():
    assert needs_confirmation(Step("run_command", {"cmd": "del *"}))


def test_system_settings_changes_need_confirmation():
    assert needs_confirmation(R.make_step("click_ui", name="Dark", app="settings"))
    assert needs_confirmation(R.make_step("click_ui", name="Dark"), foreground_title="Settings")
    assert not needs_confirmation(R.make_step("click_ui", name="Save"), foreground_title="Untitled - Notepad")


def test_overwrite_and_destructive_ui_need_confirmation(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert needs_confirmation(R.make_step("create_file", path=str(f), content="y"))
    assert not needs_confirmation(R.make_step("create_file", path=str(tmp_path / "new.txt")))
    assert needs_confirmation(R.make_step("click_ui", name="Delete account"))
    assert needs_confirmation(R.make_step("hotkey", keys="alt+f4"))
    assert needs_confirmation(R.make_step("move_file", src="a", dst="b"))


def test_create_file_outside_profile_fails_and_writes_nothing():
    a, _, _ = make_agent()
    a.planner = Script([Step("create_file", {"path": "C:/Windows/evil.txt", "content": "x"})])
    r = a.run("x")
    assert not r.ok
    import os
    assert not os.path.exists("C:/Windows/evil.txt")


def test_create_file_and_verify_in_profile(tmp_path):
    a, _, _ = make_agent(folders={"documents": tmp_path})
    a.planner = Script([R.make_step("create_file", path="documents/t.py", content="print('hi')\n")])
    r = a.run("x")
    assert r.ok and (tmp_path / "t.py").read_text() == "print('hi')\n"
