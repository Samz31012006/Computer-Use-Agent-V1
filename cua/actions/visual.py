"""VisualGroundingEngine: the grounding cascade as an ActionEngine, tried LAST for click_ui/find_ui -- after
UIAEngine (exact UIA name match) and OcrActionEngine (exact OCR text match) have both already failed, via the
same engine-fallback-ladder StepRunner already runs for every tool (cua/agent/recovery.py; nothing new there).

Interprets the step's `name` argument as a natural-language description ("first channel", "button in the
top-right", "the Like button") when literal matching hasn't worked, and only ever acts on a grounding result
that clears a confidence threshold -- a model or heuristic's guess is data, never a command: a low-confidence
result is refused (the step fails cleanly, same as "no engine handles this"), never clicked.
"""
from __future__ import annotations

from cua.actions.uia import UIAEngine
from cua.agent.state import Ctx
from cua.perception.grounder import GroundingCascade, MEDIUM_CONFIDENCE
from cua.types import EngineResult, Step

_TOOLS = {"click_ui", "find_ui"}


class VisualGroundingEngine:
    name = "visual_grounding"

    def __init__(self, cascade: GroundingCascade):
        self.cascade = cascade
        self._uia = UIAEngine()          # reused only for its target-window resolution

    def can_handle(self, step: Step) -> bool:
        return step.action in _TOOLS and bool(step.args.get("name"))

    def execute(self, step: Step, ctx: Ctx) -> EngineResult:
        a = step.args
        hwnd = self._uia._target(a, ctx)
        if isinstance(hwnd, EngineResult):
            return hwnd
        fg = ctx.perception.foreground()
        context = fg.title if fg else ""

        with ctx.log.span("perceive_visual"):
            result = self.cascade.ground(a["name"], context, hwnd)
        ctx.state.context["last_ground_result"] = result.to_dict()

        if not result.found:
            return EngineResult(False, f"visual grounding failed: {result.reason}", retryable=False)
        if result.confidence < MEDIUM_CONFIDENCE:
            return EngineResult(False, f"grounding confidence too low ({result.confidence:.2f} < "
                                f"{MEDIUM_CONFIDENCE}) to safely act on '{a['name']}': {result.reason}",
                                retryable=False)

        if step.action == "click_ui":
            with ctx.log.span("act"):
                ctx.driver.click(result.x, result.y)
            return EngineResult(True, f"{result.method}-click '{result.description}' "
                                f"({result.confidence:.2f} confidence) at ({result.x},{result.y})")
        return EngineResult(True, f"{result.method}: {result.description} ({result.confidence:.2f})")
