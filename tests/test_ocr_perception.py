from cua.actions.ocr import OcrActionEngine
from cua.actions.registry import default_registry
from cua.actions.uia import UIAEngine
from cua.perception.ocr import NullOcr, TextHit, match_words
from cua.types import EngineResult, TaskPlan, WinInfo
from tests.fakes import ScriptedEngine, make_agent

R = default_registry()
WORDS = [("Save", (10, 10, 50, 30)), ("As", (55, 10, 70, 30)), ("Cancel", (200, 10, 260, 30)),
         ("Save", (10, 100, 50, 120))]


def test_match_words_finds_phrases_with_union_rect_and_case_insensitively():
    h = match_words(WORDS, "save as")
    assert h[0].text == "Save As" and h[0].rect == (10, 10, 70, 30) and h[0].center == (40, 20)
    assert [x.text for x in match_words(WORDS, "cancel")] == ["Cancel"]
    assert match_words(WORDS, "nothing") == []


def test_prefers_shortest_match():
    assert match_words(WORDS, "save")[0].text == "Save"


class FakeOcr:
    available = True

    def __init__(self, hits): self.hits, self.queries = hits, []

    def find_text(self, text, hwnd=None):
        self.queries.append(text)
        return self.hits


def test_ocr_engine_clicks_found_text_and_reports_when_missing():
    a, d, p = make_agent(engines=[OcrActionEngine(FakeOcr([TextHit("Save", (10, 10, 50, 30))]))])
    p.windows_.append(WinInfo(1, "x", "c", 1, "x.exe"))
    a.planner = type("P", (), {"plan": lambda s, t, c: TaskPlan("x", [R.make_step("click_ui", name="Save")]),
                               "replan": lambda *x: None})()
    r = a.run("x")
    assert r.ok and r.steps[0].engine == "ocr" and ("click", 30, 20, "left", False) in d.calls
    a2, d2, p2 = make_agent(engines=[OcrActionEngine(FakeOcr([]))])
    p2.windows_.append(WinInfo(1, "x", "c", 1, "x.exe"))
    a2.planner = a.planner
    r2 = a2.run("x")
    assert not r2.ok and "not found by OCR" in r2.steps[0].detail and d2.calls == []


def test_ocr_engine_is_inactive_when_backend_unavailable_or_no_name():
    e = OcrActionEngine(NullOcr())
    assert not e.can_handle(R.make_step("click_ui", name="Save"))
    assert not OcrActionEngine(FakeOcr([])).can_handle(R.make_step("click_ui", control_type="button"))


def test_not_found_skips_retry_and_falls_through_to_next_engine():
    first = ScriptedEngine("uia", {"click_ui"}, [EngineResult(False, "element not found", retryable=False)])
    second = ScriptedEngine("ocr", {"click_ui"}, [EngineResult(True, "clicked")])
    a, _, p = make_agent(engines=[first, second])
    p.windows_.append(WinInfo(1, "x", "c", 1, "x.exe"))
    a.planner = type("P", (), {"plan": lambda s, t, c: TaskPlan("x", [R.make_step("click_ui", name="Save")]),
                               "replan": lambda *x: None})()
    r = a.run("x")
    assert r.ok and first.calls == 1 and second.calls == 1 and r.steps[0].engine == "ocr"
