import json

from cua.actions.registry import default_registry
from cua.agent.planner import ChainPlanner
from cua.agent.router import RouterPlanner
from cua.config import Config
from cua.models.local_llm import LLMPlanner, build_prompt, select_tools
from cua.models.runtime import LlamaServerRuntime, ModelUnavailable
from cua.types import Step
from tests.fakes import PLAN_OPEN_CHROME, FakeRuntime, make_agent

R = default_registry()


def mk(outputs, **cfg):
    rt = FakeRuntime(outputs)
    return LLMPlanner(rt, R, Config(**cfg)), rt


def test_valid_output_becomes_plan_and_records_stats():
    pl, rt = mk([PLAN_OPEN_CHROME])
    ctx = {}
    p = pl.plan("please launch the browser thing", ctx)
    assert p and p.source == "llm" and p.steps[0].action == "open_app"
    assert ctx["planner_calls"][0]["completion_tokens"] == 40
    assert rt.schemas[0]["properties"]["steps"]["items"]["oneOf"]      # per-tool constrained schema was sent


def test_malformed_output_retries_once_then_succeeds():
    pl, rt = mk(["garbage", PLAN_OPEN_CHROME], llm_retries=1)
    assert pl.plan("x", {}) is not None
    assert rt.calls == 2 and "invalid" in rt.prompts[1]


def test_persistent_garbage_returns_none_and_is_bounded():
    pl, rt = mk(["garbage"], llm_retries=1)
    ctx = {}
    assert pl.plan("x", ctx) is None
    assert rt.calls == 2 and ctx["planner_error"]


def test_hallucinated_tool_is_rejected():
    bad = json.dumps({"goal": "g", "steps": [{"action": "format_disk", "target": "c:"}]})
    pl, _ = mk([bad], llm_retries=0)
    ctx = {}
    assert pl.plan("x", ctx) is None and "unknown tool" in ctx["planner_error"]


def test_model_unavailable_is_not_an_exception():
    pl, _ = mk([ModelUnavailable("not enough free RAM")])
    ctx = {}
    assert pl.plan("x", ctx) is None and "RAM" in ctx["planner_error"]


def test_replan_prompt_carries_the_failure():
    pl, rt = mk([PLAN_OPEN_CHROME])
    pl.replan("open chrome", {}, Step("open_app", {"name": "chrome"}), "no window for chrome")
    assert "no window for chrome" in rt.prompts[0] and "NEW plan" in rt.prompts[0]


def test_prompt_is_short():
    tools = select_tools("open chrome and search cats", R)
    prompt = build_prompt("open chrome and search cats", tools, "foreground: 'x'", "chatml", 8)
    assert len(prompt) < 2600            # ~650 tokens including two examples; prefix is cached between calls


# ---- chain ---------------------------------------------------------------------
def test_router_hit_never_calls_the_model():
    pl, rt = mk([PLAN_OPEN_CHROME])
    chain, ctx = ChainPlanner(RouterPlanner(R), pl), {}
    assert chain.plan("open notepad", ctx).source == "router"
    assert rt.calls == 0 and ctx["router_hit"] is True and ctx["planner"] == "router"


def test_router_miss_falls_through_to_model():
    pl, rt = mk([PLAN_OPEN_CHROME])
    chain, ctx = ChainPlanner(RouterPlanner(R), pl), {}
    assert chain.plan("I need something to browse the internet with", ctx).source == "llm"
    assert rt.calls == 1 and ctx["router_hit"] is False


def test_planner_disabled_mode_uses_router_only():
    chain, ctx = ChainPlanner(RouterPlanner(R), None), {}
    assert chain.plan("write me a poem", ctx) is None and ctx["planner"] == "none"


def test_end_to_end_llm_plan_executes_on_fakes():
    pl, rt = mk([json.dumps({"goal": "g", "steps": [{"action": "open_app", "target": "notepad"}]})])
    a, d, _ = make_agent(planner=ChainPlanner(RouterPlanner(R), pl), config=Config(confirm_llm_plans=False))
    r = a.run("I need something to write notes with")
    assert r.ok and r.planner == "llm" and d.calls == [("launch", "notepad")]
    assert rt.calls == 1 and r.planner_calls[0]["completion_tokens"] == 40


def test_llm_plan_cannot_bypass_confirmation():
    plan = json.dumps({"goal": "g", "steps": [{"action": "close_app", "target": "notepad"}]})
    pl, _ = mk([plan])
    a, d, p = make_agent(planner=ChainPlanner(RouterPlanner(R), pl), config=Config(confirm_llm_plans=False))
    from cua.types import WinInfo
    p.windows_.append(WinInfo(1, "notepad", "c", 1, "notepad.exe"))
    r = a.run("get rid of the text editor")          # router miss -> model -> close_app -> policy still asks
    assert r.reason == "declined" and d.calls == []


# ---- runtime guards ------------------------------------------------------------
def test_memory_guard_refuses_to_load(tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"0" * 1024)
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    cfg = Config(models_dir=str(tmp_path), min_free_ram_mb=10**9)
    rt = LlamaServerRuntime(cfg)
    assert "not enough free RAM" in rt.unavailable_reason()
    try:
        rt.generate("hi")
        assert False, "should raise"
    except ModelUnavailable as e:
        assert "RAM" in str(e)


def test_no_model_installed_is_a_clean_unavailable(tmp_path):
    rt = LlamaServerRuntime(Config(models_dir=str(tmp_path)))
    assert rt.unavailable_reason() in ("llama-server not installed", "no .gguf model found")


def test_ai_plans_can_be_set_to_need_approval_and_router_plans_never_do():
    pl, rt = mk([json.dumps({"steps": [{"action": "open_app", "arguments": {"name": "notepad"}}]})])
    asked = []
    a, d, _ = make_agent(planner=ChainPlanner(RouterPlanner(R), pl), confirm=lambda why: asked.append(why) or False,
                         config=Config(confirm_llm_plans=True))
    assert a.run("I need something to write notes with").reason == "declined" and d.calls == [] and "AI plan" in asked[0]
    asked.clear()
    assert a.run("open notepad").ok and asked == []            # router hit: never asks, never calls the model


def test_ai_plans_run_automatically_by_default_but_risky_steps_still_ask():
    safe = json.dumps({"steps": [{"action": "open_app", "arguments": {"name": "notepad"}}]})
    asked = []
    a, d, _ = make_agent(planner=ChainPlanner(RouterPlanner(R), mk([safe])[0]),
                         confirm=lambda why: asked.append(why) or False)
    assert a.run("I need something to write notes with").ok and asked == [] and d.calls == [("launch", "notepad")]
    risky = json.dumps({"steps": [{"action": "close_app", "arguments": {"name": "notepad"}}]})
    a2, d2, p2 = make_agent(planner=ChainPlanner(RouterPlanner(R), mk([risky])[0]),
                            confirm=lambda why: asked.append(why) or False)
    from cua.types import WinInfo
    p2.windows_.append(WinInfo(1, "notepad", "c", 1, "notepad.exe"))
    assert a2.run("dispose of the notes thing").reason == "declined" and d2.calls == [] and asked


# ---- RAM-adaptive model selection ------------------------------------------------------------
# The target machine has ~7.6 GB total and often only ~1.2 GB free, so a fixed choice of the largest model
# meant the planner simply never loaded and every command the router missed failed outright.
def _models(tmp_path, **sizes):
    for name, mb in sizes.items():
        (tmp_path / f"{name}.gguf").write_bytes(b"0" * int(mb * 2**20))
    return Config(models_dir=str(tmp_path), min_free_ram_mb=100)


def test_picks_the_largest_model_that_fits_in_free_ram(tmp_path):
    cfg = _models(tmp_path, big=1000, small=400)
    assert cfg.resolve_model(free_mb=3000).name == "big.gguf"        # plenty of room: use the better model
    assert cfg.resolve_model(free_mb=900).name == "small.gguf"       # tight: fall back rather than fail


def test_configured_model_is_preferred_when_it_fits(tmp_path):
    cfg = _models(tmp_path, big=1000, small=400)
    cfg.model_path = str(tmp_path / "small.gguf")
    assert cfg.resolve_model(free_mb=3000).name == "small.gguf"


def test_configured_model_is_replaced_by_a_smaller_one_when_it_does_not_fit(tmp_path):
    cfg = _models(tmp_path, big=1000, small=400)
    cfg.model_path = str(tmp_path / "big.gguf")
    assert cfg.resolve_model(free_mb=900).name == "small.gguf"       # working beats refusing to plan


def test_reports_the_cheapest_option_when_nothing_fits(tmp_path):
    cfg = _models(tmp_path, big=1000, small=400)
    assert cfg.resolve_model(free_mb=200).name == "small.gguf"       # so the RAM error names a real candidate
