"""VLM grounding backend: the last resort in the cascade, only reached when UIA, OCR, and the CV heuristics
all fail to confidently locate a target. Model choice and its measured limitations are documented in
cua/config.py; the short version: SmolVLM-500M loads fast and answers general questions about a screenshot
correctly, but its coordinate output is not reliably accurate (measured: out of bounds, inconsistent across
repeated tries). This backend therefore caps its own confidence low (_MAX_CONFIDENCE) -- it exists so the
cascade has SOMETHING to try before giving up, not because its answers should be trusted blindly. The
GroundingCascade / VisualGroundingEngine decide, from that capped confidence, whether it's ever safe to act on.

Uses the same LlamaServerRuntime as the text planner (cua/models/runtime.py), pointed at a separate model +
mmproj, so process lifecycle (lazy load, idle unload, RAM guard) is not duplicated.
"""
from __future__ import annotations

import io
import re

from cua.config import Config
from cua.models.runtime import LlamaServerRuntime, ModelUnavailable
from cua.perception.vision import GroundResult

MAX_CONFIDENCE = 0.35     # deliberately below the cascade's MEDIUM_CONFIDENCE gate -- see module docstring
_MAX_SIDE = 768            # downscale before sending; the model was evaluated at this working size


class VLMGrounder:
    def __init__(self, cfg: Config | None = None, runtime: LlamaServerRuntime | None = None):
        self.cfg = cfg or Config()
        self._rt = runtime

    def _runtime(self) -> LlamaServerRuntime:
        if self._rt is None:
            self._rt = LlamaServerRuntime(self.cfg, model_path=self.cfg.resolve_vlm_model(),
                                          mmproj_path=self.cfg.resolve_vlm_mmproj(),
                                          ctx_size=self.cfg.vlm_ctx_size, idle_unload_s=self.cfg.vlm_idle_unload_s,
                                          log_name="vlm-server.log")
        return self._rt

    @property
    def available(self) -> bool:
        return self.unavailable_reason() is None

    def unavailable_reason(self) -> str | None:
        if not self.cfg.vlm_enabled:
            return "VLM grounding disabled"
        try:
            import PIL  # noqa: F401
        except ImportError:
            return "Pillow not installed"
        return self._runtime().unavailable_reason()

    def unload(self) -> None:
        if self._rt is not None:
            self._rt.unload()

    def ground(self, target: str, context: str = "", hwnd: int | None = None) -> GroundResult:
        reason = self.unavailable_reason()
        if reason:
            return GroundResult(found=False, description=target, reason=reason, method="vlm")

        from PIL import Image
        from cua.perception.capture import capture, window_rect

        rect = window_rect(hwnd)
        w, h, px = capture(rect, hwnd)
        img = Image.frombuffer("RGBA", (w, h), px, "raw", "BGRA", 0, 1).convert("RGB")
        scale = min(1.0, _MAX_SIDE / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        prompt = (f"The image is {img.size[0]}x{img.size[1]} pixels. "
                 f"{'Context: ' + context + '. ' if context else ''}"
                 f'Find "{target}". Reply with ONLY its pixel coordinates in the exact format: x,y')
        try:
            gen = self._runtime().generate_vision(prompt, buf.getvalue(), max_tokens=self.cfg.vlm_max_new_tokens)
        except ModelUnavailable as e:
            return GroundResult(found=False, description=target, reason=str(e), method="vlm")

        point = _parse_point(gen.text, img.size[0], img.size[1])
        if point is None:
            return GroundResult(found=False, description=target,
                                reason=f"unparseable or out-of-range response: {gen.text!r}", method="vlm")
        rx, ry = point
        x, y = rect[0] + int(rx / scale), rect[1] + int(ry / scale)   # map back to real screen coordinates
        return GroundResult(found=True, x=x, y=y, confidence=MAX_CONFIDENCE, description=target,
                            reason="VLM point estimate (low-confidence by design; see module docstring)",
                            method="vlm")


def _parse_point(text: str, w: int, h: int) -> tuple[int, int] | None:
    m = re.search(r"(-?\d+)\D+(-?\d+)", text)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= x <= w and 0 <= y <= h):
        return None
    return x, y
