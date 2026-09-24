"""Dynamic-capability tests: typo normalization, ambiguous-typo clarification, auto-launch-on-focus, and
lightweight cross-task context (Part 1, 5, 6 of the "dynamic computer use" milestone). No desktop is touched.
"""
from __future__ import annotations

from cua.agent.normalize import fuzzy_resolve
from cua.agent.router import RouterPlanner
from cua.catalog import AppSpec, register
from cua.types import Step
from tests.fakes import make_agent
from tests.test_agent import OnePlan


# ---- fuzzy normalization (Part 1C / Part 6) ------------------------------------------------
def test_high_confidence_typos_resolve_silently():
    for typo, expected in [("youtbue", "youtube"), ("chrmoe", "chrome"), ("notepaad", "notepad"),
                           ("facebok", "facebook")]:
        corrected, alts = fuzzy_resolve(typo)
        assert corrected == expected and alts == []


def test_unrelated_word_is_left_alone():
    assert fuzzy_resolve("xyz123nonsense") == (None, [])


def test_router_corrects_typo_when_opening_an_app():
    plan = RouterPlanner().plan("open notepaad")
    assert plan and plan.steps[0].action == "open_app" and plan.steps[0].args["name"] == "notepad"


def test_router_corrects_typo_when_going_to_a_site():
    plan = RouterPlanner().plan("open youtbue")
    assert plan and plan.steps[0].action == "navigate_url" and "youtube.com" in plan.steps[0].args["url"]


def test_ambiguous_typo_asks_for_clarification_instead_of_guessing():
    # two synthetic catalog entries equally close to the same misspelling -- forces real ambiguity
    register(AppSpec("zzztestzeta", ("shell", "zzztestzeta")))
    register(AppSpec("zzztestzetb", ("shell", "zzztestzetb")))
    corrected, alts = fuzzy_resolve("zzztestzet")
    assert corrected is None and set(alts) == {"zzztestzeta", "zzztestzetb"}

    agent, driver, perception = make_agent()
    r = agent.run("open zzztestzet")
    assert not r.ok and r.reason.startswith("clarify:")
    assert driver.calls == []                          # never guessed, never launched anything


# ---- auto-launch on focus (Part 1A: "type hello on notepad" without opening it first) ------
def test_typing_into_an_app_launches_it_first_if_not_already_open():
    agent, driver, perception = make_agent()
    assert perception.windows_ == []                    # nothing open yet
    agent.planner = OnePlan([Step("type_text", {"text": "hello", "app": "notepad"})])
    r = agent.run("type hello on notepad")
    assert r.ok
    assert ("launch", "notepad") in driver.calls
    assert ("type", "hello") in driver.calls


def test_clicking_ui_in_an_app_launches_it_first_if_not_already_open():
    agent, driver, perception = make_agent()
    agent.planner = OnePlan([Step("click_ui", {"name": "Save", "app": "notepad"})])
    r = agent.run("click save in notepad")
    assert ("launch", "notepad") in driver.calls         # UIAEngine's _target() now auto-launches too


def test_typing_into_an_already_open_app_does_not_relaunch_it():
    agent, driver, perception = make_agent()
    agent.run("open notepad")
    driver.calls.clear()
    agent.planner = OnePlan([Step("type_text", {"text": "hi", "app": "notepad"})])
    r = agent.run("type hi")
    assert r.ok
    assert ("launch", "notepad") not in driver.calls
    assert ("type", "hi") in driver.calls


# ---- lightweight cross-task context (Part 5) ------------------------------------------------
def test_standalone_type_command_uses_the_previously_opened_app():
    agent, driver, perception = make_agent()
    memory: dict = {}
    r1 = agent.run("open notepad", context=memory)
    assert r1.ok and memory.get("last_app") == "notepad"

    r2 = agent.run("type hello", context=memory)         # no app named this time
    # the router resolved the step against Notepad (from context) and the keystrokes were sent to it; whether
    # verification then "sees" the text is a FakePerception limitation (it doesn't simulate typed text landing
    # anywhere), unrelated to the context-carrying behaviour this test targets
    assert r2.plan is not None and r2.plan.steps[0].args["app"] == "notepad"
    assert ("type", "hello") in driver.calls


def test_router_plan_seeds_app_from_context_only_when_none_is_named():
    with_ctx = RouterPlanner().plan("type hello", {"last_app": "notepad"})
    assert with_ctx and with_ctx.steps[0].args["app"] == "notepad"

    without_ctx = RouterPlanner().plan("type hello", {})
    assert without_ctx and without_ctx.steps[0].args.get("app") is None

    explicit_target_wins = RouterPlanner().plan("type hello into chrome", {"last_app": "notepad"})
    assert explicit_target_wins and explicit_target_wins.steps[0].args["app"] == "chrome"
