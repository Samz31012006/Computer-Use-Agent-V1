import pytest

from cua.agent.router import RouterPlanner

R = RouterPlanner()


def acts(text):
    p = R.plan(text)
    return None if p is None else [s.action for s in p.steps]


@pytest.mark.parametrize("text,exp", [
    # synonyms
    ("bring up the calculator", ["open_app"]), ("fire up chrome", ["open_app"]), ("pull up notepad", ["open_app"]),
    ("get rid of notepad", ["close_app"]), ("kill chrome", ["close_app"]),
    # several targets at once
    ("open chrome and notepad", ["open_app", "open_app"]), ("open chrome, calculator and paint", ["open_app"] * 3),
    ("close notepad and calculator", ["close_app", "close_app"]),
    # connectors
    ("open notepad; type hello", ["open_app", "hotkey", "type_text"]), ("open chrome & go to github", ["open_app", "navigate_url"]),
    ("open chrome next go to yt", ["open_app", "navigate_url"]),
    # window state
    ("minimize chrome", ["minimize_app"]), ("maximise the notepad window", ["maximize_app"]), ("restore chrome", ["focus_app"]),
    # keyboard phrases
    ("go back", ["hotkey"]), ("refresh", ["hotkey"]), ("new tab", ["hotkey"]), ("save it", ["hotkey"]), ("zoom in", ["hotkey"]),
    ("select all", ["hotkey"]), ("undo", ["hotkey"]), ("take a screenshot", ["hotkey"]),
    ("open chrome and refresh", ["open_app", "hotkey"]),
    # scrolling
    ("scroll down", ["scroll"]), ("scroll up 5", ["scroll"]), ("scroll down a lot", ["scroll"]), ("scroll to the top", ["hotkey"]),
    # files
    ("copy report.pdf from downloads to documents", ["copy_file"]), ("move notes.txt to desktop", ["move_file"]),
    ("list my downloads", ["list_directory"]), ("what's in my documents", ["open_folder"]),
    # file creation with content
    ("create a python file called test.py and write hello world", ["create_file", "open_file"]),
    ("make a javascript file called app.js with hello world", ["create_file", "open_file"]),
    ("create a text file called todo.txt saying buy milk", ["create_file", "open_file"]),
    ("open vs code and create a python file called test.py and write a hello world program",
     ["open_app", "create_file", "open_file"]),
    # system toggle
    ("turn on dark mode", ["open_settings", "click_ui", "click_ui"]), ("switch to light mode", ["open_settings", "click_ui", "click_ui"]),
    ("enable dark theme", ["open_settings", "click_ui", "click_ui"]),
])
def test_wide_range_recognised(text, exp):
    assert acts(text) == exp


def test_hello_world_content_is_real_code_and_folded_into_one_create_step():
    p = R.plan("open vs code, create a python file called test.py, and write a hello world program")
    create = next(s for s in p.steps if s.action == "create_file")
    assert create.args["path"] == "documents/test.py" and create.args["content"] == 'print("Hello, World!")\n'
    assert [s.action for s in p.steps].count("create_file") == 1
    java = R.plan("create a java file called Greeter.java with hello world").steps[0].args["content"]
    assert "class Greeter" in java


def test_saying_content_and_default_names():
    assert R.plan("create a text file called todo.txt saying buy milk").steps[0].args["content"] == "buy milk\n"
    assert R.plan("create a python file").steps[0].args["path"] == "documents/untitled.py"


def test_code_files_open_in_vscode_text_files_in_default_app():
    assert R.plan("create a python file called a.py").steps[1].args.get("app") == "vscode"
    assert "app" not in R.plan("create a text file called a.txt").steps[1].args


def test_dark_mode_steps_target_settings_and_need_confirmation():
    from cua.safety.policy import needs_confirmation
    p = R.plan("turn on dark mode")
    assert p.steps[1].args == {"name": "Choose your mode", "app": "settings"} and p.steps[2].args["name"] == "Dark"
    assert needs_confirmation(p.steps[1]) and needs_confirmation(p.steps[2])     # system change: always asks


@pytest.mark.parametrize("text", [
    "write hello world", "open notepad and write me a poem about cats", "shut down the computer", "create a spreadsheet file", "summarize this page", "back to the future",
    "make the text bigger", "what is the capital of france", "refresh my memory about spain",
])
def test_still_declines_what_it_cannot_do_safely(text):
    assert acts(text) is None


def test_move_is_confirmed_copy_is_not():
    from cua.safety.policy import needs_confirmation
    assert needs_confirmation(R.plan("move notes.txt to desktop").steps[0])
    assert not needs_confirmation(R.plan("copy notes.txt to desktop").steps[0])


@pytest.mark.parametrize("text,exp", [
    ("open notepad and write hello", ["open_app", "hotkey", "type_text"]),
    ("open notepad and write stuff", ["open_app", "hotkey", "type_text"]),
    ("open notepad and write my shopping list: eggs, milk", ["open_app", "hotkey", "type_text"]),
    ("open new tab in notepad", ["open_app", "hotkey"]), ("open chrome and open a new tab", ["open_app", "hotkey"]),
    ("new tab", ["hotkey"]),
])
def test_writing_into_an_open_app_needs_no_model(text, exp):
    assert acts(text) == exp


def test_write_text_is_typed_literally():
    p = R.plan("open notepad and write hello there")
    assert p.steps[-1].args["text"] == "hello there" and p.steps[-1].args["app"] == "notepad"


# ---- bare targets: the name of a thing, with no verb -------------------------------------------
# People say "chrome" far more often than "open chrome", and every one of these used to miss the router
# entirely and get handed to the model planner (which, on the target machine, usually had no RAM to load).
@pytest.mark.parametrize("text,exp", [
    ("chrome", ["open_app"]), ("notepad", ["open_app"]), ("calculator", ["open_app"]),
    ("downloads", ["open_folder"]), ("documents", ["open_folder"]), ("my documents folder", ["open_folder"]),
    ("bluetooth settings", ["open_settings"]), ("wifi settings", ["open_settings"]), ("dark mode", ["open_settings"]),
    ("youtube", ["navigate_url"]),
])
def test_bare_target_opens_the_thing_named(text, exp):
    assert acts(text) == exp


@pytest.mark.parametrize("text", [
    # ordinary prose and real misheard transcripts (from logs/tasks.jsonl) must never launch anything
    "We're not violent.", "and not back.", "hello there", "how are you", "tell me a joke",
    "the weather is nice today", "what do you think about this", "nothing", "whatever", "okay so",
])
def test_bare_target_never_fires_on_prose_or_mishears(text):
    assert acts(text) is None


# ---- volume / media / lock: global media keys, no model needed -----------------------------------
@pytest.mark.parametrize("text,keys", [
    ("mute", "volume_mute"), ("mute the volume", "volume_mute"), ("unmute", "volume_mute"),
    ("pause", "media_play_pause"), ("pause the music", "media_play_pause"),
    ("next song", "media_next"), ("previous track", "media_prev"),
    ("lock the pc", "win+l"), ("lock", "win+l"),
])
def test_system_control_phrases(text, keys):
    p = R.plan(text)
    assert p is not None and p.steps[0].action == "hotkey" and p.steps[0].args["keys"] == keys


@pytest.mark.parametrize("text,key", [
    ("volume up", "volume_up"), ("turn the volume up", "volume_up"), ("turn up the volume", "volume_up"),
    ("louder", "volume_up"), ("volume down", "volume_down"), ("turn down the volume", "volume_down"),
    ("quieter", "volume_down"),
])
def test_volume_steps_repeat_the_media_key(text, key):
    p = R.plan(text)
    assert p is not None and all(s.args["keys"] == key for s in p.steps) and len(p.steps) > 1


def test_play_something_searches_youtube_but_bare_play_is_the_media_key():
    p = R.plan("play despacito on youtube")
    assert p.steps[0].action == "web_search" and p.steps[0].args["engine"] == "youtube"
    assert p.steps[0].args["query"] == "despacito"
    assert R.plan("play").steps[0].args["keys"] == "media_play_pause"
