"""AgentService: the shared core both the laptop UI and the phone API submit through.

Nothing here touches the desktop or needs a model -- fakes stand in for the driver/perception/engines, same
convention as test_agent.py. These tests specifically cover the Phase 8 requirements: one Agent, one worker
(no two tasks running at once), shared task ids/state/events across "laptop" and "phone" submitters, and
cancellation via both an explicit request and the same code path the physical Ctrl+Alt+Q hotkey uses.
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from queue import Empty

from cua.agent.service import AgentService
from cua.config import Config
from cua.safety.kill_switch import KillSwitch
from cua.server.api import AgentAPI
from cua.server.auth import AuthStore
from cua.types import EngineResult, Step, TaskPlan, WinInfo
from tests.fakes import make_agent

_ROOT = Path(__file__).resolve().parents[1]


def make_api(svc) -> AgentAPI:
    """AgentAPI with its own isolated AuthStore (not the shared .cache/buddy_auth.json) -- these tests only
    need AgentAPI to exist and delegate to the service; auth itself is covered in test_server.py."""
    return AgentAPI(svc, auth=AuthStore(path=Path(tempfile.mktemp(suffix=".json"))))


def safe_kill() -> KillSwitch:
    """A KillSwitch that never registers a real hotkey or hard-exits the test process (same convention as
    test_agent.py's `ks = KillSwitch(); ks.hard_exit = False`)."""
    ks = KillSwitch()
    ks.hard_exit = False
    return ks


class GateEngine:
    """Handles a fake 'slow_step' action: signals `gate` the instant it starts, then sleeps `hold` seconds
    before returning. Lets a test create a deterministic window during which a task is mid-execution."""

    def __init__(self, gate: threading.Event, hold: float = 0.0):
        self.name, self.gate, self.hold, self.calls = "gate", gate, hold, 0

    def can_handle(self, step):
        return step.action == "wait"

    def execute(self, step, ctx):
        self.calls += 1
        self.gate.set()
        time.sleep(self.hold)
        return EngineResult(True, "done")


class TextPlanner:
    """Deterministic planner: text -> a fixed TaskPlan, bypassing the router/LLM entirely."""

    def __init__(self, plans: dict[str, TaskPlan]):
        self.plans = plans

    def plan(self, text, context):
        return self.plans.get(text)

    def replan(self, *a):
        return None


def make_service(plans: dict, hold: float = 0.0, **agent_kw):
    gate = threading.Event()
    engine = GateEngine(gate, hold=hold)
    agent, driver, perception = make_agent(engines=[engine], **agent_kw)
    agent.planner = TextPlanner(plans)
    svc = AgentService(agent=agent, kill=safe_kill())
    return svc, gate, engine, driver, perception


def wait_for(pred, timeout=2.0, interval=0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def drain(q, task_id, timeout=1.0) -> list[dict]:
    out, deadline = [], time.time() + timeout
    while time.time() < deadline:
        try:
            evt = q.get(timeout=0.05)
        except Empty:
            continue
        if evt.get("id") != task_id:
            continue
        out.append(evt)
        if evt["type"] in ("completed", "failed", "cancelled"):
            break
    return out


ONE_STEP = {"a": TaskPlan("a", [Step("wait", {})])}


# ---- laptop / phone both go through the same service -----------------------------------
def test_laptop_task_uses_shared_service():
    svc, gate, engine, *_ = make_service(ONE_STEP)
    rec = svc.submit_task("a", source="laptop")
    assert wait_for(lambda: svc.get_task(rec.id).status == "completed")
    assert svc.get_task(rec.id) is rec and rec.source == "laptop" and engine.calls == 1


def test_phone_task_uses_shared_service():
    svc, gate, engine, *_ = make_service(ONE_STEP)
    api = make_api(svc)
    rec = api.submit_task("a")                        # AgentAPI always tags source="phone"
    assert wait_for(lambda: svc.get_task(rec.id).status == "completed")
    assert rec.source == "phone" and api.get_task(rec.id) is rec and engine.calls == 1


# ---- only one Agent / one model runtime -------------------------------------------------
def test_only_one_agent_shared_by_both_interfaces():
    svc, *_ = make_service({})
    api = make_api(svc)
    assert api.agent is svc.agent                      # AgentAPI has no Agent of its own

    ui_src = (_ROOT / "cua/ui/app.py").read_text(encoding="utf-8")
    server_src = (_ROOT / "start_server.py").read_text(encoding="utf-8")
    assert "build_agent" not in ui_src and "AgentService" in ui_src
    assert "build_agent" not in server_src and "AgentService" in server_src


def test_agent_service_constructs_agent_exactly_once(monkeypatch):
    import cua.agent.service as svc_mod
    calls = []
    real_build_agent = svc_mod.build_agent

    def spy(*a, **kw):
        calls.append(1)
        return real_build_agent(*a, **kw)
    monkeypatch.setattr(svc_mod, "build_agent", spy)

    svc = AgentService(cfg=Config(planner_enabled=False), kill=safe_kill(), log_dir=None)
    assert len(calls) == 1
    assert svc.agent.runtime is None                   # planner disabled -> no llama-server process was ever built


# ---- one task at a time -------------------------------------------------------------------
def test_second_task_does_not_run_until_first_finishes():
    plans = {"a": TaskPlan("a", [Step("wait", {})]), "b": TaskPlan("b", [Step("wait", {})])}
    svc, gate, engine, *_ = make_service(plans, hold=0.2)
    rec_a = svc.submit_task("a", source="laptop")
    assert wait_for(lambda: gate.is_set(), timeout=1)   # a is now mid-execution
    rec_b = svc.submit_task("b", source="phone")
    time.sleep(0.05)
    assert svc.get_task(rec_b.id).status == "received"  # still queued behind a
    assert engine.calls == 1
    assert wait_for(lambda: svc.get_task(rec_a.id).status == "completed", timeout=2)
    assert wait_for(lambda: svc.get_task(rec_b.id).status == "completed", timeout=2)
    assert engine.calls == 2


# ---- task ids / task state shared ----------------------------------------------------------
def test_task_ids_are_shared_between_service_and_api():
    svc, gate, engine, *_ = make_service(ONE_STEP)
    api = make_api(svc)
    rec = svc.submit_task("a", source="laptop")
    assert rec.id.startswith("task_")
    assert api.get_task(rec.id) is rec
    assert wait_for(lambda: rec.status == "completed")


def test_task_state_identical_from_both_apis():
    svc, gate, engine, *_ = make_service(ONE_STEP)
    api = make_api(svc)
    rec = api.submit_task("a")
    assert wait_for(lambda: svc.get_task(rec.id).status == "completed")
    assert svc.get_task(rec.id).to_dict() == api.get_task(rec.id).to_dict()


# ---- events shared across subscribers (laptop's queue vs. phone's WS queue) ----------------
def test_events_are_shared_across_subscribers():
    svc, gate, engine, *_ = make_service(ONE_STEP)
    q_laptop, q_phone = svc.subscribe(), svc.subscribe()
    rec = svc.submit_task("a", source="laptop")
    types_laptop = [e["type"] for e in drain(q_laptop, rec.id)]
    types_phone = [e["type"] for e in drain(q_phone, rec.id)]
    assert types_laptop == types_phone
    for stage in ("received", "planning", "executing", "completed"):
        assert stage in types_laptop, types_laptop


# ---- cancellation --------------------------------------------------------------------------
def test_cancel_currently_executing_task():
    plans = {"a": TaskPlan("a", [Step("wait", {}), Step("wait", {})])}
    svc, gate, engine, *_ = make_service(plans, hold=0.2)
    rec = svc.submit_task("a", source="phone")
    assert wait_for(lambda: gate.is_set(), timeout=1)
    assert svc.cancel_task(rec.id) is True
    assert wait_for(lambda: svc.get_task(rec.id).status == "cancelled", timeout=2)
    assert svc.get_task(rec.id).result.reason.startswith("aborted")
    assert engine.calls == 1                            # the second slow_step never ran


def test_cancel_queued_task_before_it_starts():
    plans = {"a": TaskPlan("a", [Step("wait", {})]), "b": TaskPlan("b", [Step("wait", {})])}
    svc, gate, engine, *_ = make_service(plans, hold=0.3)
    rec_a = svc.submit_task("a", source="laptop")
    assert wait_for(lambda: gate.is_set(), timeout=1)
    rec_b = svc.submit_task("b", source="phone")
    assert svc.cancel_task(rec_b.id) is True
    assert svc.get_task(rec_b.id).status == "cancelled"
    assert wait_for(lambda: svc.get_task(rec_a.id).status == "completed", timeout=2)
    assert engine.calls == 1                             # b's slow_step never ran

    assert svc.cancel_task("task_doesnotexist") is False
    assert svc.cancel_task(rec_a.id) is False             # already finished


def test_cancel_reachable_from_phone_api():
    plans = {"a": TaskPlan("a", [Step("wait", {}), Step("wait", {})])}
    svc, gate, engine, *_ = make_service(plans, hold=0.2)
    api = make_api(svc)
    rec = api.submit_task("a")
    assert wait_for(lambda: gate.is_set(), timeout=1)
    assert api.cancel_task(rec.id) is True
    assert wait_for(lambda: svc.get_task(rec.id).status == "cancelled", timeout=2)


# ---- global kill switch: identical path for a real hotkey press or a cancel request --------
def test_kill_switch_aborts_current_task_regardless_of_origin():
    plans = {"a": TaskPlan("a", [Step("wait", {}), Step("wait", {})])}
    svc, gate, engine, *_ = make_service(plans, hold=0.1)
    rec = svc.submit_task("a", source="phone")           # e.g. a task the phone started
    assert wait_for(lambda: gate.is_set(), timeout=1)
    svc.kill.trigger()                                    # exactly what the physical Ctrl+Alt+Q hotkey does
    assert wait_for(lambda: svc.get_task(rec.id).status == "cancelled", timeout=2)
    assert engine.calls == 1


def test_only_one_kill_switch_instance_exists():
    svc, *_ = make_service({})
    api = make_api(svc)
    assert svc.kill is svc.agent.kill
    assert api.service.kill is svc.kill                   # the phone side never registers its own hotkey


# ---- existing safety + planner behaviour is unchanged ---------------------------------------
def test_confirmation_gate_still_enforced_through_the_service():
    agent, driver, perception = make_agent(confirm=lambda why: False)  # fail-closed, matches deny_all default
    agent.planner = TextPlanner({"close notepad": TaskPlan("close notepad", [Step("close_app", {"name": "notepad"})])})
    perception.windows_.append(WinInfo(1, "notepad", "c", 1, "notepad.exe"))
    svc = AgentService(agent=agent, kill=safe_kill())

    rec = svc.submit_task("close notepad", source="laptop")
    assert wait_for(lambda: svc.get_task(rec.id).status == "failed")
    assert svc.get_task(rec.id).result.reason == "declined"
    assert driver.calls == []


def test_router_planner_behaviour_unchanged_through_the_service():
    agent, driver, perception = make_agent()             # real RouterPlanner, not swapped out
    svc = AgentService(agent=agent, kill=safe_kill())
    rec = svc.submit_task("open notepad", source="laptop")
    assert wait_for(lambda: svc.get_task(rec.id).status == "completed")
    assert driver.calls == [("launch", "notepad")]


def test_planner_receives_the_exact_voice_transcript_unmodified():
    """Whatever VoiceInput hands off as the final transcript reaches the planner byte-for-byte -- the service
    never "cleans up" a command to make it look more correct than what was actually heard."""
    seen_texts = []

    class RecordingPlanner:
        def plan(self, text, context):
            seen_texts.append(text)
            return TaskPlan(text, [Step("wait", {})])

        def replan(self, *a):
            return None

    agent, driver, perception = make_agent()
    agent.planner = RecordingPlanner()
    svc = AgentService(agent=agent, kill=safe_kill())

    misheard_but_real = "click the first channl on youtube"     # a plausible STT output, typo and all
    rec = svc.submit_task(misheard_but_real, source="laptop")
    assert wait_for(lambda: svc.get_task(rec.id).status == "completed")
    assert seen_texts == [misheard_but_real]                    # exactly what was "heard", no silent rewrite
    assert rec.text == misheard_but_real
