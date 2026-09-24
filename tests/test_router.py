import pytest

from cua.agent.router import RouterPlanner

R = RouterPlanner()


def actions(text):
    p = R.plan(text)
    return None if p is None else [s.action for s in p.steps]


def test_spec_example_open_chrome_and_search():
    p = R.plan("Open Chrome and search for the latest SpaceX launch.")
    assert [s.action for s in p.steps] == ["open_app", "web_search"]
    assert p.steps[1].args["browser"] == "chrome"
    assert p.steps[1].args["query"] == "the latest SpaceX launch"


def test_downloads_pdf_today():
    p = R.plan("Open my Downloads folder and find the PDF I downloaded today")
    assert [s.action for s in p.steps] == ["open_folder", "find_file"]
    f = p.steps[1].args
    assert f["extension"] == "pdf" and f["modified"] == "today" and f["folder"] == "downloads"


def test_vscode_new_python_file():
    p = R.plan("Open VS Code and create a new Python file")
    assert [s.action for s in p.steps] == ["open_app", "create_file", "open_file"]
    assert p.steps[1].args["path"] == "documents/untitled.py"


@pytest.mark.parametrize("text,exp", [
    ("open notepad", ["open_app"]),
    ("open report.pdf", ["find_file", "open_file"]),
    ("open report.pdf with chrome", ["find_file", "open_file"]),
    ("open my downloads folder and find the pdf i downloaded today and open it", ["open_folder", "find_file", "open_file"]),
    ("please launch the calculator", ["open_app"]),
    ("open notepad and type hello world", ["open_app", "hotkey", "type_text"]),
    ('type "buy milk and eggs" in notepad', ["type_text"]),
    ("close notepad", ["close_app"]),
    ("open bluetooth settings", ["open_settings"]),
    ("open github.com", ["navigate_url"]),
    ("google python tutorials", ["web_search"]),
    ("search youtube for lofi beats", ["web_search"]),
    ("search my downloads for invoice", ["find_file"]),
    ("press ctrl+s", ["hotkey"]),
    ("click the Save button", ["click_ui"]),
    ("switch to chrome", ["focus_app"]),
])
def test_recognised(text, exp):
    assert actions(text) == exp


@pytest.mark.parametrize("text", [
    "write me a poem about cats", "what's the weather", "open the latest report and summarize it",
    "book a flight to paris", "", "press banana", "close",
])
def test_misses_are_none_not_guesses(text):
    assert actions(text) is None


def test_quoted_text_not_split():
    p = R.plan('type "open the door and close it" into notepad')
    assert len(p.steps) == 1 and p.steps[0].args["text"] == "open the door and close it"


def test_search_engine_selection():
    assert R.plan("search youtube for cats").steps[0].args["engine"] == "youtube"
    assert R.plan("search for mars on wikipedia").steps[0].args["engine"] == "wikipedia"


# ---- dynamic phrasing -------------------------------------------------------------
def test_users_real_command_yt_search_without_connectors():
    p = R.plan("open chrome go to yt and search gz fanatic")
    assert [s.action for s in p.steps] == ["open_app", "web_search"]
    assert p.steps[1].args == {"query": "gz fanatic", "engine": "youtube", "browser": "chrome"}


@pytest.mark.parametrize("text,engine", [
    ("go to youtube and search lofi", "youtube"), ("open youtube search lofi", "youtube"),
    ("open chrome go to wikipedia search mars", "wikipedia"), ("go to yt search cats", "youtube"),
])
def test_search_stays_on_the_site_just_visited(text, engine):
    p = R.plan(text)
    assert p.steps[-1].action == "web_search" and p.steps[-1].args["engine"] == engine
    assert "navigate_url" not in [s.action for s in p.steps]      # folded into the search: no wasted page load


def test_site_aliases_and_bare_words():
    assert R.plan("go to yt").steps[0].args["url"] == "https://youtube.com"
    assert R.plan("go to reddit").steps[0].args["url"] == "https://reddit.com"
    assert R.plan("open youtube").steps[0].action == "navigate_url"
    assert R.plan("go to notepad").steps[0].action == "open_app"
    assert R.plan("go to example.org").steps[0].args["url"] == "https://example.org"


def test_implicit_split_does_not_cut_ordinary_text():
    assert R.plan('type "open chrome go to yt" in notepad').steps[-1].args["text"] == "open chrome go to yt"
    assert [s.action for s in R.plan("open chrome and go to github").steps] == ["open_app", "navigate_url"]
