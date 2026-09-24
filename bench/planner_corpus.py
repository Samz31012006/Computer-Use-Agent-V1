"""Extended planner evaluation corpus: 110+ commands phrased to MISS the deterministic router.

Each entry: (command, [(allowed_actions, [required_arg_substrings]), ...])
  allowed_actions  - tuple of tool names any of which is correct for that step
  required_arg_substrings - strings that must appear (case-insensitive) in that step's args JSON

Categories are labeled so benchmark reports can break accuracy down by type.
"""
from __future__ import annotations

A = lambda *a: tuple(a)

# fmt: off
CASES: list[tuple[str, list[tuple[tuple[str, ...], list[str]]], str]] = [
    # ---- APPLICATION CONTROL (15) ------------------------------------------------
    ("bring up the calculator",                [(A("open_app"), ["calc"])],                              "app_control"),
    ("get rid of notepad",                     [(A("close_app"), ["notepad"])],                          "app_control"),
    ("I need something to browse the web",     [(A("open_app"), ["chrome", "edge", "firefox", "browser"])], "app_control"),
    ("can you launch something for notes",     [(A("open_app"), ["notepad"])],                           "app_control"),
    ("get the calculator running for me",      [(A("open_app"), ["calc"])],                              "app_control"),
    ("I want to edit some code",               [(A("open_app"), ["vscode", "vs code", "code"])],         "app_control"),
    ("put away the text editor",               [(A("close_app"), ["notepad"])],                          "app_control"),
    ("shut the browser down",                  [(A("close_app"), ["chrome", "edge", "firefox", "browser"])], "app_control"),
    ("could you start paint for me",           [(A("open_app"), ["paint"])],                             "app_control"),
    ("launch the file manager",                [(A("open_app"), ["explorer"])],                          "app_control"),
    ("run the windows terminal",               [(A("open_app"), ["terminal"])],                          "app_control"),
    ("I need spotify playing",                 [(A("open_app"), ["spotify"])],                           "app_control"),
    ("dispose of the calculator app",          [(A("close_app"), ["calc"])],                             "app_control"),
    ("start up the edge browser",              [(A("open_app"), ["edge"])],                              "app_control"),
    ("get me a code editor",                   [(A("open_app"), ["vscode", "vs code", "code"])],         "app_control"),

    # ---- WINDOW CONTROL (12) -----------------------------------------------------
    ("switch over to chrome",                  [(A("focus_app", "switch_window", "open_app"), ["chrome"])], "window_control"),
    ("make the chrome window fill the screen",  [(A("maximize_app"), ["chrome"])],                       "window_control"),
    ("shrink notepad out of the way",          [(A("minimize_app"), ["notepad"])],                       "window_control"),
    ("put chrome in the background",           [(A("minimize_app"), ["chrome"])],                        "window_control"),
    ("bring the browser back up",              [(A("focus_app"), ["chrome", "edge", "firefox", "browser"])], "window_control"),
    ("I need to see the calculator again",     [(A("focus_app"), ["calc"])],                             "window_control"),
    ("make notepad smaller",                   [(A("minimize_app"), ["notepad"])],                       "window_control"),
    ("go to my other window",                  [(A("switch_window", "focus_app", "hotkey"), [])],        "window_control"),
    ("enlarge the vs code window",             [(A("maximize_app"), ["vscode", "vs code", "code"])],     "window_control"),
    ("show me what window is in front",        [(A("read_window_title"), [])],                           "window_control"),
    ("what application am I looking at",       [(A("read_window_title"), [])],                           "window_control"),
    ("check which window has focus",           [(A("read_window_title"), [])],                           "window_control"),

    # ---- KEYBOARD ACTIONS (12) ---------------------------------------------------
    ("save this document",                     [(A("hotkey"), ["ctrl+s"])],                              "keyboard"),
    ("hit the enter key",                      [(A("press_key", "hotkey"), ["enter"])],                  "keyboard"),
    ("undo my last change",                    [(A("hotkey"), ["ctrl+z"])],                              "keyboard"),
    ("select everything on screen",            [(A("hotkey"), ["ctrl+a"])],                              "keyboard"),
    ("copy what I have selected",              [(A("hotkey"), ["ctrl+c"])],                              "keyboard"),
    ("paste what I copied",                    [(A("hotkey"), ["ctrl+v"])],                              "keyboard"),
    ("I want to find something on this page",  [(A("hotkey"), ["ctrl+f"])],                              "keyboard"),
    ("take a snapshot of the screen",          [(A("hotkey"), ["win+shift+s", "printscreen", "prtsc"])],  "keyboard"),
    ("redo the thing I just undid",            [(A("hotkey"), ["ctrl+y"])],                              "keyboard"),
    ("press the escape button",                [(A("press_key", "hotkey"), ["esc"])],                    "keyboard"),
    ("hit tab to move to the next field",      [(A("press_key", "hotkey"), ["tab"])],                    "keyboard"),
    ("cut the selected text",                  [(A("hotkey"), ["ctrl+x"])],                              "keyboard"),

    # ---- FILE OPERATIONS (15) ----------------------------------------------------
    ("make me a text file called bench_note.txt saying buy milk",
     [(A("create_file"), ["bench_note.txt", "buy milk"])],                                               "file_ops"),
    ("write a python file called hello.py that prints hello world",
     [(A("create_file"), ["hello.py", "print"])],                                                        "file_ops"),
    ("copy report.pdf from downloads to documents",
     [(A("copy_file"), ["report.pdf"])],                                                                 "file_ops"),
    ("find the spreadsheet I got yesterday",   [(A("find_file"), ["yesterday"])],                        "file_ops"),
    ("locate my resume pdf",                   [(A("find_file"), ["resume", "pdf"])],                    "file_ops"),
    ("I need to find a zip file I downloaded recently",
     [(A("find_file"), ["zip", "download"])],                                                            "file_ops"),
    ("duplicate my notes to the desktop",      [(A("copy_file"), ["note", "desktop"])],                  "file_ops"),
    ("relocate the budget file to documents",  [(A("move_file"), ["budget", "document"])],               "file_ops"),
    ("produce a javascript file named app.js", [(A("create_file"), ["app.js"])],                         "file_ops"),
    ("make me a new html page called index.html",
     [(A("create_file"), ["index.html"])],                                                               "file_ops"),
    ("I lost a pdf somewhere on my computer",  [(A("find_file"), ["pdf"])],                              "file_ops"),
    ("search for any word documents I have",   [(A("find_file"), ["doc"])],                              "file_ops"),
    ("where is the invoice I downloaded today",
     [(A("find_file"), ["invoice", "today"])],                                                           "file_ops"),
    ("create a rust file called main.rs with hello world",
     [(A("create_file"), ["main.rs"])],                                                                  "file_ops"),
    ("write me a go file named server.go",     [(A("create_file"), ["server.go"])],                      "file_ops"),

    # ---- FOLDER OPERATIONS (10) --------------------------------------------------
    ("show me what's in my downloads",         [(A("list_directory", "open_folder"), ["download"])],     "folder_ops"),
    ("take me to my documents",                [(A("open_folder"), ["document"])],                       "folder_ops"),
    ("what files do I have on my desktop",     [(A("list_directory", "open_folder"), ["desktop"])],      "folder_ops"),
    ("browse my pictures folder",              [(A("open_folder"), ["picture"])],                        "folder_ops"),
    ("show me the contents of downloads",      [(A("list_directory", "open_folder"), ["download"])],     "folder_ops"),
    ("open up my music folder",                [(A("open_folder"), ["music"])],                          "folder_ops"),
    ("I want to see my documents",             [(A("open_folder"), ["document"])],                       "folder_ops"),
    ("list everything in my desktop folder",   [(A("list_directory"), ["desktop"])],                     "folder_ops"),
    ("let me look at my downloads",            [(A("open_folder", "list_directory"), ["download"])],     "folder_ops"),
    ("navigate to my videos folder",           [(A("open_folder"), ["video"])],                          "folder_ops"),

    # ---- BROWSER / NAVIGATION (14) -----------------------------------------------
    ("head over to wikipedia.org",             [(A("navigate_url"), ["wikipedia"])],                     "browser"),
    ("what is the weather like in Paris",      [(A("web_search"), ["paris"])],                           "browser"),
    ("look up funny cat videos on youtube",    [(A("web_search"), ["cat", "youtube"])],                  "browser"),
    ("I need to check my gmail",               [(A("navigate_url"), ["gmail", "google.com/mail", "mail.google"])], "browser"),
    ("search for the best restaurants near me",[(A("web_search"), ["restaurant"])],                      "browser"),
    ("go to the google homepage",              [(A("navigate_url"), ["google"])],                        "browser"),
    ("look up how to make pasta",              [(A("web_search"), ["pasta"])],                           "browser"),
    ("find news about artificial intelligence",[(A("web_search"), ["artificial intelligence", "ai"])],   "browser"),
    ("visit reddit.com",                       [(A("navigate_url"), ["reddit"])],                        "browser"),
    ("take me to github",                      [(A("navigate_url"), ["github"])],                        "browser"),
    ("search wikipedia for the roman empire",  [(A("web_search"), ["roman", "wikipedia"])],              "browser"),
    ("I want to watch something on netflix",   [(A("navigate_url"), ["netflix"])],                       "browser"),
    ("google how to fix a leaky faucet",       [(A("web_search"), ["faucet", "leak"])],                  "browser"),
    ("look for python tutorials online",       [(A("web_search"), ["python", "tutorial"])],              "browser"),

    # ---- UI INTERACTION (10) -----------------------------------------------------
    ("click the Save button",                  [(A("click_ui"), ["save"])],                              "ui_interaction"),
    ("press the OK button",                    [(A("click_ui"), ["ok"])],                                "ui_interaction"),
    ("type my name into the search box",       [(A("type_into_ui", "click_ui"), ["search"])],            "ui_interaction"),
    ("hit the Submit button",                  [(A("click_ui"), ["submit"])],                            "ui_interaction"),
    ("click on the Cancel option",             [(A("click_ui"), ["cancel"])],                            "ui_interaction"),
    ("press the Apply button",                 [(A("click_ui"), ["apply"])],                             "ui_interaction"),
    ("tap the Next button",                    [(A("click_ui"), ["next"])],                              "ui_interaction"),
    ("click the Close button",                 [(A("click_ui"), ["close"])],                             "ui_interaction"),
    ("read what's in the status bar",          [(A("read_ui_text"), [])],                                "ui_interaction"),
    ("what controls are visible right now",    [(A("inspect_controls"), [])],                            "ui_interaction"),

    # ---- SETTINGS (6) ------------------------------------------------------------
    ("open the wifi settings",                 [(A("open_settings"), ["wifi"])],                         "settings"),
    ("take me to display settings",            [(A("open_settings"), ["display"])],                      "settings"),
    ("open bluetooth preferences",             [(A("open_settings"), ["bluetooth"])],                    "settings"),
    ("I want to check for system updates",     [(A("open_settings"), ["update", "windowsupdate"])],     "settings"),
    ("show me the sound settings",             [(A("open_settings"), ["sound"])],                        "settings"),
    ("go to network configuration",            [(A("open_settings"), ["network", "wifi"])],              "settings"),

    # ---- MULTI-STEP TASKS (18) ---------------------------------------------------
    ("open chrome and find the latest SpaceX launch",
     [(A("open_app"), ["chrome"]), (A("web_search"), ["spacex"])],                                       "multi_step"),
    ("open notepad and write my shopping list: eggs, milk",
     [(A("open_app"), ["notepad"]), (A("type_text"), ["eggs"])],                                         "multi_step"),
    ("go to youtube and search for lofi music",
     [(A("web_search", "navigate_url"), ["lofi", "youtube"])],                                           "multi_step"),
    ("open vs code, create a file called test.py and put a hello world program in it",
     [(A("create_file"), ["test.py", "print"])],                                                         "multi_step"),
    ("launch chrome, navigate to github.com, then search for python projects",
     [(A("open_app"), ["chrome"]), (A("navigate_url"), ["github"]), (A("web_search", "type_text"), ["python"])], "multi_step"),
    ("open the calculator and compute 5 times 3",
     [(A("open_app"), ["calc"]), (A("click_ui"), ["5", "five"]), (A("click_ui"), ["multiply", "times"]),
      (A("click_ui"), ["3", "three"]), (A("click_ui"), ["equal"])],                                      "multi_step"),
    ("open notepad, type hello world, then save the file",
     [(A("open_app"), ["notepad"]), (A("type_text"), ["hello world"]), (A("hotkey"), ["ctrl+s"])],        "multi_step"),
    ("launch edge and go to bing.com",
     [(A("open_app"), ["edge"]), (A("navigate_url"), ["bing"])],                                         "multi_step"),
    ("start chrome and look up the score of last night's game",
     [(A("open_app"), ["chrome"]), (A("web_search"), ["score", "game"])],                                "multi_step"),
    ("open file explorer and go to my downloads",
     [(A("open_app"), ["explorer"]), (A("open_folder"), ["download"])],                                  "multi_step"),
    ("type hello world into notepad",
     [(A("type_text"), ["hello world"])],                                                                 "multi_step"),
    ("open settings and go to the display page",
     [(A("open_settings"), ["display"])],                                                                 "multi_step"),
    ("search for news and open the first result",
     [(A("web_search"), ["news"]), (A("click_ui", "click"), [])],                                        "multi_step"),
    ("find my report in downloads and open it",
     [(A("find_file"), ["report", "download"]), (A("open_file"), [])],                                   "multi_step"),
    ("create a new text file called notes.txt and type meeting at 3pm",
     [(A("create_file"), ["notes.txt"]), (A("type_text", "open_file"), ["meeting", "3pm"])],              "multi_step"),
    ("open chrome, go to youtube, and search for funny videos",
     [(A("open_app"), ["chrome"]), (A("navigate_url", "web_search"), ["youtube"]),
      (A("web_search", "type_text"), ["funny"])],                                                         "multi_step"),
    ("find the latest pdf in my downloads and copy it to documents",
     [(A("find_file"), ["pdf", "download"]), (A("copy_file"), ["document"])],                             "multi_step"),
    ("open notepad and press ctrl+a then delete",
     [(A("open_app"), ["notepad"]), (A("hotkey"), ["ctrl+a"]), (A("press_key", "hotkey"), ["delete"])],   "multi_step"),

    # ---- SCROLL (4) --------------------------------------------------------------
    ("scroll down a bit",                      [(A("scroll"), ["-"])],                                   "scroll"),
    ("scroll up several lines",                [(A("scroll"), [])],                                      "scroll"),
    ("page down in this document",             [(A("scroll", "press_key", "hotkey"), [])],               "scroll"),
    ("go to the bottom of this page",          [(A("hotkey", "scroll"), ["ctrl+end", "end"])],           "scroll"),
]
# fmt: on

CATEGORIES = sorted({c for _, _, c in CASES})
