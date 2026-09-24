"""Visual grounding cascade tests: CV heuristics, GroundResult validation, cascade confidence gating,
VisualGroundingEngine's safety gate (never clicks below the confidence threshold), VLM response parsing.
No real screenshot/OCR/model needed -- fakes throughout, same convention as test_ocr_perception.py.
"""
from __future__ import annotations

from cua.actions.registry import default_registry
from cua.actions.visual import VisualGroundingEngine
from cua.perception import cv_heuristics as cv
from cua.perception.grounder import GroundingCascade, HIGH_CONFIDENCE, MEDIUM_CONFIDENCE
from cua.perception.ocr import match_words
from cua.perception.vision import GroundResult, NullGrounder
from cua.perception.vlm import _parse_point
from cua.types import TaskPlan, WinInfo
from tests.fakes import make_agent

R = default_registry()


# ---- cv_heuristics --------------------------------------------------------------------------
def test_parse_ordinal():
    assert cv.parse_ordinal("click the first channel") == 0
    assert cv.parse_ordinal("the third result") == 2
    assert cv.parse_ordinal("the last video") == -1
    assert cv.parse_ordinal("click the button") is None


def test_parse_spatial():
    assert cv.parse_spatial("button in the top-right") == {"top", "right"}
    assert cv.parse_spatial("the search icon") == set()


def test_cluster_cards_groups_by_proximity_and_orders_top_to_bottom():
    words = [
        ("Channel", (10, 10, 80, 30)), ("One", (85, 10, 110, 30)),         # card 1 (top)
        ("Channel", (10, 200, 80, 220)), ("Two", (85, 200, 110, 220)),     # card 2 (bottom)
    ]
    regions = cv.cluster_cards(words)
    assert len(regions) == 2
    assert regions[0].rect[1] < regions[1].rect[1]      # top card ordered first


def test_cluster_cards_excludes_top_margin_when_bounds_given():
    # top word sits in the top 8% of a 0..1000 window (title bar / browser chrome); bottom word is page content
    words = [("Chrome", (10, 5, 80, 20)), ("Video", (10, 500, 80, 520))]
    without_bounds = cv.cluster_cards(words)
    assert len(without_bounds) == 2                    # no exclusion without `bounds`

    with_bounds = cv.cluster_cards(words, bounds=(0, 0, 1000, 1000))
    assert len(with_bounds) == 1 and with_bounds[0].rect[1] == 500     # the title-bar word was dropped


def test_select_ordinal_picks_correct_region():
    regions = [cv.Region((0, 0, 10, 10)), cv.Region((0, 20, 10, 30)), cv.Region((0, 40, 10, 50))]
    assert cv.select_ordinal(regions, 0) is regions[0]
    assert cv.select_ordinal(regions, -1) is regions[2]
    assert cv.select_ordinal(regions, 99) is None


def test_filter_spatial_keeps_only_matching_region():
    bounds = (0, 0, 1000, 1000)
    top_left = cv.Region((10, 10, 50, 50))
    bottom_right = cv.Region((900, 900, 950, 950))
    assert cv.filter_spatial([top_left, bottom_right], {"top", "right"}, bounds) == []
    assert cv.filter_spatial([top_left, bottom_right], {"bottom", "right"}, bounds) == [bottom_right]
    assert cv.filter_spatial([top_left, bottom_right], set(), bounds) == [top_left, bottom_right]


# ---- GroundResult / NullGrounder -------------------------------------------------------------
def test_ground_result_to_dict():
    r = GroundResult(True, 10, 20, 0.9, "Save button", (0, 0, 20, 20), "exact match", "ocr_exact")
    d = r.to_dict()
    assert d["found"] is True and d["x"] == 10 and d["confidence"] == 0.9 and d["method"] == "ocr_exact"


def test_null_grounder_always_fails_closed():
    g = NullGrounder()
    assert not g.available
    assert not g.ground("anything").found


# ---- GroundingCascade -----------------------------------------------------------------------
class FakeOcrEngine:
    def __init__(self, words, available=True):
        self.words, self.available = words, available

    def recognize(self, hwnd=None):
        return self.words

    def find_text(self, text, hwnd=None):
        return match_words(self.words, text)


class FakeVLM:
    def __init__(self, result, available=True):
        self._result, self.available, self.calls = result, available, 0

    def ground(self, target, context="", hwnd=None):
        self.calls += 1
        return self._result


def test_cascade_exact_ocr_match_wins_first():
    ocr = FakeOcrEngine([("Subscribe", (100, 100, 200, 120))])
    r = GroundingCascade(ocr).ground("Subscribe")
    assert r.found and r.method == "ocr_exact" and r.confidence >= HIGH_CONFIDENCE


def test_cascade_fuzzy_ocr_match_when_no_exact():
    ocr = FakeOcrEngine([("Subscrbe", (100, 100, 200, 120))])       # typo
    r = GroundingCascade(ocr).ground("Subscribe")
    assert r.found and r.method == "ocr_fuzzy"


def test_cascade_cv_ordinal_selection_for_first_channel():
    # y-positions clear of the top-margin exclusion zone (window title bar / browser chrome) and well
    # separated from each other -- cluster_cards() excludes the top ~14% of the real screen height by design
    words = [
        ("Channel", (10, 200, 80, 220)), ("Alpha", (85, 200, 140, 220)),
        ("Channel", (10, 500, 80, 520)), ("Beta", (85, 500, 140, 520)),
    ]
    r = GroundingCascade(FakeOcrEngine(words)).ground("first channel")
    # CV-only ordinal selection is capped BELOW MEDIUM_CONFIDENCE by design (see grounder.py) -- live testing
    # against a real page showed it isn't reliable enough to auto-execute, so it's returned but the caller
    # (VisualGroundingEngine) must refuse to act on it without a more confident method agreeing.
    assert r.found and r.method == "cv" and r.confidence < MEDIUM_CONFIDENCE
    assert r.y < 300          # still picked the TOP card (near y=200), not the bottom one (near y=500)


def test_cascade_falls_through_to_vlm_when_ocr_and_cv_fail():
    vlm_result = GroundResult(True, 500, 300, 0.3, "guess", method="vlm")
    vlm = FakeVLM(vlm_result)
    r = GroundingCascade(FakeOcrEngine([]), vlm).ground("mystery icon")
    assert r.found and r.method == "vlm" and vlm.calls == 1


def test_cascade_returns_not_found_when_everything_fails():
    r = GroundingCascade(FakeOcrEngine([]), vlm=None).ground("nothing here")
    assert not r.found and r.method == "cascade"


# ---- VLM response parsing ---------------------------------------------------------------------
def test_parse_point_valid_formats():
    assert _parse_point("512, 300", 1000, 1000) == (512, 300)
    assert _parse_point("x=512 y=300", 1000, 1000) == (512, 300)


def test_parse_point_rejects_out_of_bounds():
    assert _parse_point("2000, 300", 1000, 1000) is None


def test_parse_point_rejects_unparseable_text():
    assert _parse_point("I don't know", 1000, 1000) is None


# ---- VisualGroundingEngine: the safety gate ---------------------------------------------------
class FakeCascade:
    def __init__(self, result):
        self._result = result

    def ground(self, target, context="", hwnd=None):
        return self._result


def _run_click_ui(cascade_result, name="first channel"):
    engine = VisualGroundingEngine(FakeCascade(cascade_result))
    a, d, p = make_agent(engines=[engine])
    p.windows_.append(WinInfo(1, "chrome", "c", 1, "chrome.exe"))
    a.planner = type("P", (), {"plan": lambda s, t, c: TaskPlan("x", [R.make_step("click_ui", name=name)]),
                               "replan": lambda *x: None})()
    return a.run("x"), d


def test_high_confidence_grounding_executes_the_click():
    result = GroundResult(True, 400, 300, 0.9, "first channel", method="ocr_exact")
    r, d = _run_click_ui(result)
    assert r.ok and ("click", 400, 300, "left", False) in d.calls


def test_medium_confidence_grounding_still_executes():
    result = GroundResult(True, 400, 300, 0.55, "first channel", method="cv")
    r, d = _run_click_ui(result)
    assert r.ok and ("click", 400, 300, "left", False) in d.calls


def test_low_confidence_grounding_refuses_to_click():
    result = GroundResult(True, 400, 300, 0.2, "guess", method="vlm")
    r, d = _run_click_ui(result)
    assert not r.ok and d.calls == []             # never clicked
    assert "confidence too low" in r.steps[0].detail


def test_not_found_grounding_refuses_to_click():
    result = GroundResult(False, description="x", reason="nowhere to be found")
    r, d = _run_click_ui(result)
    assert not r.ok and d.calls == []


def test_grounding_result_recorded_in_task_context_for_diagnostics():
    result = GroundResult(True, 400, 300, 0.9, "first channel", method="ocr_exact")
    engine = VisualGroundingEngine(FakeCascade(result))
    a, d, p = make_agent(engines=[engine])
    p.windows_.append(WinInfo(1, "chrome", "c", 1, "chrome.exe"))
    a.planner = type("P", (), {"plan": lambda s, t, c: TaskPlan("x", [R.make_step("click_ui", name="first channel")]),
                               "replan": lambda *x: None})()
    memory: dict = {}
    a.run("x", context=memory)
    assert memory["last_ground_result"]["method"] == "ocr_exact" and memory["last_ground_result"]["x"] == 400
