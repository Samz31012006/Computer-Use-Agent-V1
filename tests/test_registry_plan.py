import json

import pytest

from cua.actions.base import Safety, ToolError
from cua.actions.paths import resolve_user_path
from cua.actions.registry import default_registry
from cua.agent.plan import PlanError, parse_plan_json
from cua.models.local_llm import select_tools

R = default_registry()


def plan(steps, **extra):
    return json.dumps({"goal": "g", "steps": steps, **extra})


# ---- registry ------------------------------------------------------------------
def test_registry_covers_required_capabilities():
    need = {"open_app", "close_app", "focus_app", "minimize_app", "maximize_app", "type_text", "press_key", "hotkey",
            "click", "double_click", "right_click", "move_mouse", "scroll", "find_window", "switch_window",
            "read_window_title", "inspect_controls", "find_file", "open_file", "create_file", "copy_file", "move_file",
            "open_folder", "list_directory", "navigate_url", "click_ui", "type_into_ui", "read_ui_text", "find_ui"}
    assert need <= set(R.names())
    assert all(R.get(n).description for n in R.names())


def test_no_shell_or_code_tools_exist():
    assert not {"run_command", "shell", "exec", "python", "powershell"} & set(R.names())


def test_declared_safety_levels():
    assert R.get("close_app").safety is Safety.CONFIRM and R.get("move_file").safety is Safety.CONFIRM
    assert R.get("open_app").safety is Safety.SAFE


@pytest.mark.parametrize("name,args", [
    ("open_app", {}), ("open_app", {"name": "x", "bogus": 1}), ("click", {"x": "abc", "y": 1}),
    ("open_settings", {"page": "nope"}), ("navigate_url", {"url": "javascript:alert(1)"}),
    ("navigate_url", {"url": "file:///c:/x"}), ("hotkey", {"keys": "ctrl+banana"}), ("find_file", {}),
    ("no_such_tool", {}),
])
def test_bad_arguments_rejected(name, args):
    with pytest.raises(ToolError):
        R.make_step(name, **args)


def test_argument_coercion_and_normalisation():
    assert R.make_step("click", x="10", y=20.0).args == {"x": 10, "y": 20}
    assert R.make_step("hotkey", keys="Control + S").args["keys"] == "ctrl+s"
    assert R.make_step("navigate_url", url="example.com").args["url"] == "https://example.com"
    assert R.make_step("find_file", extension=".PDF").args["extension"] == "pdf"


def test_tools_attach_their_own_verification():
    assert R.make_step("open_app", name="chrome").verify["kind"] == "app_window"
    assert R.make_step("create_file", path="documents/a.txt").verify["kind"] == "file_exists"
    assert R.make_step("press_key", key="enter").verify is None


def test_schema_makes_hallucinated_tools_and_bad_arguments_unrepresentable():
    item = R.json_schema()["properties"]["steps"]["items"]["oneOf"]
    assert {v["properties"]["action"]["const"] for v in item} == set(R.names())
    settings = next(v for v in item if v["properties"]["action"]["const"] == "open_settings")
    args = settings["properties"]["arguments"]
    assert args["additionalProperties"] is False and "wifi" in args["properties"]["page"]["enum"]
    assert args["required"] == ["page"]
    loose = R.json_schema(per_tool=False)["properties"]["steps"]["items"]["properties"]["action"]["enum"]
    assert set(loose) == set(R.names())


def test_tool_selection_shrinks_prompt_but_keeps_relevant_tools():
    small = select_tools("open chrome", R)
    assert len(small.prompt()) < len(R.prompt()) * 0.6
    assert "close_app" not in small.names()
    assert "close_app" in select_tools("close notepad", R).names()


# ---- plan validation -----------------------------------------------------------
def test_valid_plan_compiles_with_target_shorthand():
    p = parse_plan_json(plan([{"action": "open_app", "target": "chrome"},
                              {"action": "web_search", "target": "cats", "arguments": {"browser": "chrome"}}]), R)
    assert [s.action for s in p.steps] == ["open_app", "web_search"]
    assert p.steps[1].args["query"] == "cats" and p.source == "llm"


def test_tolerates_code_fences_and_prose():
    p = parse_plan_json("Sure!\n```json\n" + plan([{"action": "wait", "target": "1"}]) + "\n```", R)
    assert p.steps[0].action == "wait"


@pytest.mark.parametrize("text,frag", [
    ("", "no JSON"), ("not json at all", "no JSON"), ("{bad json}", "invalid JSON"),
    (json.dumps({"goal": "g", "steps": []}), "steps"),
    (json.dumps({"goal": "g", "steps": [{"action": "wait"}]}), "missing required"),
    (plan([{"action": "run_shell", "target": "del *"}]), "unknown tool"),
    (plan([{"action": "open_app"}]), "missing required"),
    (plan([{"action": "open_app", "target": "x", "arguments": {"evil": 1}}]), "unknown argument"),
    (plan([{"action": "open_app", "target": "x", "extra_field": 1}]), "schema"),
    (plan([{"action": "wait", "target": "1"}] * 5, max_steps=3), "limit"),
])
def test_malformed_or_hostile_plans_raise_planerror(text, frag):
    with pytest.raises(PlanError) as e:
        parse_plan_json(text, R)
    assert frag in str(e.value)


def test_model_verification_hint_cannot_replace_builtin_check():
    p = parse_plan_json(plan([{"action": "open_app", "target": "chrome", "verification": "title:zzz"},
                              {"action": "press_key", "target": "enter", "verification": "element:Save"}]), R)
    assert p.steps[0].verify["kind"] == "app_window"          # built-in wins
    assert p.steps[1].verify == {"kind": "element", "name": "Save", "timeout": 4}   # hint fills a gap


# ---- paths ---------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["C:/Windows/System32/x.dll", "..\\..\\..\\Windows\\x", "documents/../../../Windows/x",
                                 "C:\\Program Files\\a.txt"])
def test_paths_outside_profile_rejected(bad):
    with pytest.raises(ToolError):
        resolve_user_path(bad)


def test_known_folder_and_relative_paths_resolve_inside_profile():
    assert resolve_user_path("downloads/a.pdf").name == "a.pdf"
    assert resolve_user_path("notes.txt").name == "notes.txt"


def test_goal_is_optional_and_not_requested_from_the_model():
    assert parse_plan_json(json.dumps({"steps": [{"action": "wait", "target": "1"}]}), R).steps[0].action == "wait"
    assert "goal" not in R.json_schema()["properties"]
