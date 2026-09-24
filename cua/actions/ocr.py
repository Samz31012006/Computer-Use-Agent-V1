"""OcrActionEngine: the fallback when UI Automation cannot see a control (canvas-drawn apps, games, images).
Tried after UIAEngine, so it costs nothing unless UIA already failed. One capture + one OCR pass per attempt."""
from __future__ import annotations

from cua.actions.uia import UIAEngine
from cua.agent.state import Ctx
from cua.perception.ocr import OcrEngine
from cua.types import EngineResult, Step

_TOOLS = {"click_ui", "find_ui", "read_ui_text"}


class OcrActionEngine:
    name = "ocr"

    def __init__(self, ocr: OcrEngine):
        self.ocr = ocr
        self._uia = UIAEngine()              # reuse its target-window resolution

    def can_handle(self, step: Step) -> bool:
        return step.action in _TOOLS and bool(step.args.get("name")) and getattr(self.ocr, "available", False)

    def execute(self, step: Step, ctx: Ctx) -> EngineResult:
        a = step.args
        hwnd = self._uia._target(a, ctx)
        if isinstance(hwnd, EngineResult):
            return hwnd
        with ctx.log.span("perceive_ocr"):
            hits = self.ocr.find_text(a["name"], hwnd)
        if not hits:
            return EngineResult(False, f"text {a['name']!r} not found by OCR", retryable=False)
        hit = hits[0]
        if step.action == "click_ui":
            with ctx.log.span("act"):
                ctx.driver.click(*hit.center)
            return EngineResult(True, f"ocr-click '{hit.text}' at {hit.center}")
        if step.action == "read_ui_text":
            ctx.state.context["read_text"] = hit.text
        return EngineResult(True, f"ocr found '{hit.text}' at {hit.center}")
