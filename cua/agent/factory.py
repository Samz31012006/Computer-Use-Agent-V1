"""build_agent(): the one place that wires planner chain, model runtime, engines and safety together.
CLI, UI and benchmark all use it, so they behave identically."""
from __future__ import annotations

from cua.actions.registry import default_registry
from cua.agent.agent import Agent
from cua.actions.ocr import OcrActionEngine
from cua.actions.skills import SkillEngine
from cua.actions.uia import UIAEngine
from cua.actions.visual import VisualGroundingEngine
from cua.agent.planner import ChainPlanner
from cua.agent.router import RouterPlanner
from cua.config import Config, load_config
from cua.models.local_llm import LLMPlanner
from cua.models.runtime import LlamaServerRuntime
from cua.perception.grounder import GroundingCascade
from cua.perception.ocr import WindowsOcr
from cua.perception.uia import UIAPerception
from cua.perception.vlm import VLMGrounder
from cua.safety.policy import deny_all


def build_agent(cfg: Config | None = None, kill=None, confirm=deny_all, on_event=None, folders=None,
                log_dir="logs", runtime=None, **kw) -> Agent:
    cfg = cfg or load_config()
    registry = default_registry()
    perception = UIAPerception()
    router = RouterPlanner(registry)
    llm, rt = None, None
    if cfg.planner_enabled:
        rt = runtime or LlamaServerRuntime(cfg)          # lazy: nothing is loaded until the router misses
        llm = LLMPlanner(rt, registry, cfg, perception, abort=(lambda: kill.event.is_set()) if kill else None)
    ocr = WindowsOcr()
    vlm = VLMGrounder(cfg) if cfg.vlm_enabled else None    # lazy: no VLM process starts until the cascade needs it
    cascade = GroundingCascade(ocr, vlm)
    engines = kw.pop("engines", None) or [SkillEngine(), UIAEngine(), OcrActionEngine(ocr),
                                          VisualGroundingEngine(cascade)]
    agent = Agent(planner=ChainPlanner(router, llm), perception=perception, engines=engines, kill=kill, confirm=confirm, folders=folders,
                  log_dir=log_dir, registry=registry, config=cfg, on_event=on_event, **kw)
    agent.runtime = rt
    agent.vlm = vlm
    return agent
