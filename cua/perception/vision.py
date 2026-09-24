"""Visual grounding result type + backend protocol.

GroundResult is the ONE shape every grounding method (OCR, CV heuristics, VLM) must return, and the only
thing the cascade (cua/perception/grounder.py) is allowed to turn into a mouse click. Nothing downstream ever
sees raw model output directly -- a VLM's text response is parsed into a GroundResult and validated (in
range, sane confidence) before it can influence anything, per the milestone's "never let a model directly
issue arbitrary mouse events" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GroundResult:
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0                        # 0..1
    description: str = ""                           # what was found, in words
    bbox: tuple[int, int, int, int] | None = None    # left, top, right, bottom, screen pixels
    reason: str = ""                                 # why found=False, or which heuristic matched
    method: str = ""                                 # "uia" | "ocr_exact" | "ocr_fuzzy" | "cv" | "vlm"

    def to_dict(self) -> dict:
        return {"found": self.found, "x": self.x, "y": self.y, "confidence": round(self.confidence, 3),
               "description": self.description, "bbox": self.bbox, "reason": self.reason, "method": self.method}


class VisionGrounder(Protocol):
    """One grounding backend (OCR+CV, or a VLM). Registered as a step in GroundingCascade; never called
    directly by the planner or by ActionEngines."""
    available: bool

    def ground(self, target: str, context: str, hwnd: int | None = None) -> GroundResult:
        """Locate `target` (a short natural-language description, e.g. "first channel", "the Like button")
        somewhere on screen. `context` is a short hint about what's being searched (e.g. "YouTube search
        results"). May take a screenshot; implementations should only be reached after cheaper methods fail."""


class NullGrounder:
    available = False

    def ground(self, target: str, context: str = "", hwnd: int | None = None) -> GroundResult:
        return GroundResult(found=False, description=target, reason="no grounder configured")
