"""Small representative corpus for the "dynamic computer use" milestone.

Unlike bench/planner_corpus.py (116 cases, one broad sweep of router-miss phrasing), this corpus targets the
8 categories called out for that milestone specifically: simple, multi-step, natural-language, omitted-context,
misspelled, multi-tool, recovery, and browser-navigation commands -- including the exact example commands from
that spec. Same (command, expected_steps, category) shape as planner_corpus.py; RECOVERY_CASES additionally
carries the failed Step + error text a replan() call needs.

Nothing here executes on the desktop -- see bench/dynamic_eval.py.
"""
from __future__ import annotations

from cua.types import Step

A = lambda *a: tuple(a)

# fmt: off
CASES: list[tuple[str, list[tuple[tuple[str, ...], list[str]]], str]] = [
    # ---- SIMPLE COMMANDS -----------------------------------------------------------
    ("Open Notepad",                             [(A("open_app"), ["notepad"])],                          "simple"),
    ("Open Chrome",                              [(A("open_app"), ["chrome"])],                            "simple"),
    ("Close Calculator",                         [(A("close_app"), ["calc", "calculator"])],               "simple"),
    ("Maximize Chrome",                          [(A("maximize_app"), ["chrome"])],                        "simple"),
    ("Take a screenshot",                        [(A("hotkey"), [])],                                      "simple"),

    # ---- MULTI-STEP COMMANDS --------------------------------------------------------
    ("Open Notepad and type hello",              [(A("open_app"), ["notepad"]), (A("type_text"), ["hello"])], "multi_step"),
    ("Open Chrome and search for SpaceX",        [(A("open_app"), ["chrome"]), (A("web_search"), ["spacex"])], "multi_step"),
    ("Open notepad, type hello, and save it",    [(A("open_app"), ["notepad"]), (A("type_text"), ["hello"]),
                                                   (A("hotkey"), ["s"])],                                    "multi_step"),
    ("Open Downloads and find the latest PDF",   [(A("open_folder"), ["downloads"]), (A("find_file"), ["pdf"])], "multi_step"),
    ("Make a python file and write hello world", [(A("create_file"), ["py"]), (A("open_file"), [])],        "multi_step"),

    # ---- NATURAL-LANGUAGE COMMANDS ---------------------------------------------------
    ("Type hello on Notepad",                    [(A("type_text"), ["hello"])],                            "natural_language"),
    ("I want to research the latest SpaceX launch. Open Chrome and search for it.",
                                                  [(A("open_app"), ["chrome"]), (A("web_search"), ["spacex"])], "natural_language"),
    ("Can you get Chrome running and look up the NVIDIA stock price",
                                                  [(A("open_app"), ["chrome"]), (A("web_search"), ["nvidia"])], "natural_language"),
    ("I need to jot down a quick note",          [(A("open_app"), ["notepad"])],                           "natural_language"),
    ("Pull up my downloads folder",              [(A("open_folder"), ["downloads"])],                      "natural_language"),

    # ---- OMITTED-CONTEXT COMMANDS ----------------------------------------------------
    ("Type hello",                               [(A("type_text"), ["hello"])],                            "omitted_context"),
    ("Search for SpaceX",                        [(A("web_search"), ["spacex"])],                          "omitted_context"),
    ("Save it",                                  [(A("hotkey"), ["s"])],                                   "omitted_context"),
    ("Maximize it",                              [(A("maximize_app"), [])],                                "omitted_context"),

    # ---- SPELLING MISTAKES ------------------------------------------------------------
    ("Open youtbue",                             [(A("navigate_url", "open_app", "web_search"), ["youtube"])], "spelling"),
    ("Open chrmoe",                              [(A("open_app"), ["chrome"])],                            "spelling"),
    ("open notepaad and type hi",                [(A("open_app"), ["notepad"]), (A("type_text"), ["hi"])], "spelling"),
    ("search on gogle for weather",              [(A("web_search"), ["weather"])],                         "spelling"),

    # ---- MULTIPLE TOOLS REQUIRED -------------------------------------------------------
    ("Find the latest PDF in Downloads and open it",
                                                  [(A("find_file"), ["pdf"]), (A("open_file"), [])],        "multi_tool"),
    ("Create a text file called notes.txt and open it in notepad",
                                                  [(A("create_file"), ["notes.txt"]), (A("open_file"), ["notepad"])], "multi_tool"),
    ("Open notepad, maximize it, and type hello",
                                                  [(A("open_app"), ["notepad"]), (A("maximize_app"), []),
                                                   (A("type_text"), ["hello"])],                             "multi_tool"),
    ("Copy report.pdf from downloads to documents",
                                                  [(A("copy_file"), ["report.pdf"])],                       "multi_tool"),

    # ---- BROWSER NAVIGATION -------------------------------------------------------------
    ("Open YouTube",                             [(A("navigate_url", "open_app", "web_search"), ["youtube"])], "browser_navigation"),
    ("Go to YouTube and search for latest SpaceX launch",
                                                  [(A("navigate_url", "open_app"), ["youtube"]),
                                                   (A("web_search"), ["spacex"])],                           "browser_navigation"),
    ("Search Wikipedia for artificial intelligence",
                                                  [(A("web_search"), ["artificial", "intelligence"])],       "browser_navigation"),
    ("Open github.com",                          [(A("navigate_url"), ["github"])],                        "browser_navigation"),
    ("Open Chrome, search for NVIDIA, and open the official website",
                                                  [(A("open_app"), ["chrome"]), (A("web_search"), ["nvidia"]),
                                                   (A("navigate_url", "click_ui", "open_file"), [])],        "browser_navigation"),
]
# fmt: on

CATEGORIES = sorted({c for _, _, c in CASES})


# ---- RECOVERY: needs .replan(text, context, failed_step, error), not .plan() ------------------
# fmt: off
RECOVERY_CASES: list[tuple[str, Step, str, list[tuple[tuple[str, ...], list[str]]]]] = [
    ("open notepad and type hello",
     Step("type_text", {"text": "hello", "app": "notepad"}), "no window for notepad",
     [(A("open_app"), ["notepad"]), (A("type_text"), ["hello"])]),

    ("open chrome and search for spacex",
     Step("web_search", {"query": "spacex", "engine": "google"}), "no window for chrome",
     [(A("open_app"), ["chrome"]), (A("web_search"), ["spacex"])]),

    ("click the save button",
     Step("click_ui", {"name": "Save"}), "element 'Save' not found",
     [(A("hotkey", "click_ui"), [])]),

    ("find and open the latest pdf in downloads",
     Step("find_file", {"extension": "pdf", "folder": "downloads"}), "no file matching in downloads",
     [(A("find_file", "open_folder"), [])]),
]
# fmt: on
