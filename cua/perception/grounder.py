"""GroundingCascade: the ONE entry point for turning a natural-language visual target ("first channel",
"button in the top-right", "the Like button") into validated screen coordinates.

Order (cheapest/most-reliable first):
  1. OCR exact substring match       (existing WindowsOcr.recognize + match_words)
  2. OCR fuzzy match                 (same OCR pass, difflib-scored against the words actually seen)
  3. lightweight CV heuristics       (ordinal/spatial reasoning over OCR word clusters; see cv_heuristics.py)
  4. VLM                             (last resort; confidence capped low -- see vlm.py)

UIA is NOT part of this cascade: it already runs first, in cua/actions/uia.py, before this is ever reached --
StepRunner's existing engine-fallback-ladder tries UIAEngine, then OcrActionEngine (exact/fuzzy OCR), then
this cascade, for the same click_ui/find_ui step. Nothing new was built for that ordering; it already existed.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from cua.perception import cv_heuristics as cv
from cua.perception.ocr import OcrEngine, match_words
from cua.perception.vision import GroundResult, VisionGrounder

HIGH_CONFIDENCE = 0.75      # execute directly
MEDIUM_CONFIDENCE = 0.5     # execute, but lean on the existing verify/recovery ladder to catch a miss
# below MEDIUM_CONFIDENCE: refuse rather than guess (see cua/actions/visual.py)


class GroundingCascade:
    def __init__(self, ocr: OcrEngine, vlm: VisionGrounder | None = None):
        self.ocr = ocr
        self.vlm = vlm

    def ground(self, target: str, context: str = "", hwnd: int | None = None) -> GroundResult:
        if not self.ocr.available:
            return self._fallback_to_vlm(target, context, hwnd, "OCR not available")

        words = self.ocr.recognize(hwnd) if hasattr(self.ocr, "recognize") else []

        hits = match_words(words, target)
        if hits:
            h = hits[0]
            return GroundResult(True, *h.center, confidence=0.95, description=h.text, bbox=h.rect,
                                reason="exact OCR text match", method="ocr_exact")

        # "first channel" / "button in the top-right" are SELECTION instructions, not text to search for --
        # scoring the whole phrase against a single short OCR word (e.g. "Channel") can spuriously clear the
        # fuzzy threshold, so descriptive/ordinal targets skip straight to CV heuristics instead.
        descriptive = cv.parse_ordinal(target) is not None or bool(cv.parse_spatial(target))
        if not descriptive:
            fuzzy = _fuzzy_word_match(words, target)
            if fuzzy:
                return fuzzy

        cv_result = self._cv_ground(target, words, hwnd)
        if cv_result and cv_result.confidence >= MEDIUM_CONFIDENCE:
            return cv_result

        vlm_result = self._vlm_ground(target, context, hwnd)
        if vlm_result and vlm_result.found:
            return vlm_result
        if cv_result:
            return cv_result               # a low-confidence CV guess is still returned; caller gates on it
        return GroundResult(False, description=target, reason="not found by OCR, CV heuristics, or VLM",
                            method="cascade")

    def _cv_ground(self, target: str, words, hwnd) -> GroundResult | None:
        from cua.perception.capture import window_rect
        bounds = window_rect(hwnd)
        ordinal = cv.parse_ordinal(target)
        spatial = cv.parse_spatial(target)
        regions = cv.cluster_cards(words, bounds=bounds)
        if spatial:
            regions = cv.filter_spatial(regions, spatial, bounds)
        if ordinal is not None and regions:
            reg = cv.select_ordinal(regions, ordinal)
            if reg:
                # Below MEDIUM_CONFIDENCE deliberately: live-tested against a real page (YouTube search
                # results), this misfired on the FIRST attempt (matched browser-chrome text even with a
                # title-bar exclusion in place). The heuristic is real and sometimes right, but not proven
                # reliable enough to auto-execute -- so by default the cascade/engine refuse rather than
                # guess, and only act on a CV-only result when nothing else in the cascade found anything
                # more confident. Raise this once cluster_cards() earns more trust (see its own docstring).
                return GroundResult(True, *reg.center, confidence=0.45, description=target, bbox=reg.rect,
                                    reason=f"ordinal selection over {len(regions)} clustered region(s) "
                                          "(unverified heuristic -- see cv_heuristics.py)", method="cv")
        if spatial and regions and ordinal is None:
            reg = regions[0]
            return GroundResult(True, *reg.center, confidence=0.45, description=target, bbox=reg.rect,
                                reason=f"spatial filter ({', '.join(sorted(spatial))}) "
                                      "(unverified heuristic -- see cv_heuristics.py)", method="cv")
        return None

    def _vlm_ground(self, target: str, context: str, hwnd) -> GroundResult | None:
        if self.vlm is None or not self.vlm.available:
            return None
        return self.vlm.ground(target, context, hwnd)

    def _fallback_to_vlm(self, target: str, context: str, hwnd, why: str) -> GroundResult:
        r = self._vlm_ground(target, context, hwnd)
        return r if (r and r.found) else GroundResult(False, description=target, reason=why, method="cascade")


def _fuzzy_word_match(words: list[tuple[str, tuple[int, int, int, int]]], target: str) -> GroundResult | None:
    q = " ".join(target.lower().split())
    best: tuple[float, str, tuple[int, int, int, int]] | None = None
    for text, rect in words:
        ratio = SequenceMatcher(None, q, text.lower()).ratio()
        if ratio >= 0.7 and (best is None or ratio > best[0]):
            best = (ratio, text, rect)
    if not best:
        return None
    ratio, text, rect = best
    cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    return GroundResult(True, cx, cy, confidence=min(0.85, ratio), description=text, bbox=rect,
                        reason=f"fuzzy OCR match ({ratio:.2f})", method="ocr_fuzzy")
