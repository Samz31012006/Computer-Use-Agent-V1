"""Router coverage corpus. Each entry: (command, expected action sequence or None for 'router should decline').

CAVEAT: written by the same author as the router, so hit-rate here is an OPTIMISTIC upper bound.
The honest number comes from real usage: the CLI appends every command + router hit/miss to logs/tasks.jsonl.
Out-of-scope entries are deliberately included so the metric can go down.
"""
L, U, S, T, F, O, C, H, K, W, N = ("open_app", "navigate_url", "web_search", "type_text", "find_file", "open_folder",
                                   "close_app", "hotkey", "click_ui", "open_settings", "create_file")

IN_SCOPE = [
    ("open chrome", [L]), ("open report.pdf", [F, "open_file"]), ("open notes.txt with vs code", [F, "open_file"]), ("launch notepad", [L]), ("start the calculator", [L]), ("open vs code", [L]),
    ("open file explorer", [L]), ("open paint", [L]), ("open spotify", [L]), ("please open firefox", [L]),
    ("can you open edge for me", [L]),
    ("open my downloads folder", [O]), ("open documents", [O]), ("open the desktop folder", [O]),
    ("open pictures", [O]),
    ("open bluetooth settings", [W]), ("open wifi settings", [W]), ("open display settings", [W]),
    ("open windows update settings", [W]), ("open settings", [L]),
    ("open github.com", [U]), ("go to youtube.com", [U]), ("visit https://example.com", [U]),
    ("open chrome and search for the latest spacex launch", [L, S]),
    ("open chrome and search for python tutorials", [L, S]),
    ("search for best pizza near me", [S]), ("google rust vs go", [S]), ("look up the capital of peru", [S]),
    ("search youtube for lofi beats", [S]), ("search wikipedia for mars", [S]),
    ("open my downloads folder and find the pdf i downloaded today", [O, F]),
    ("find the pdf i downloaded today", [F]), ("find my resume in documents", [F]),
    ("find the latest excel file in downloads", [F]), ("search my downloads for invoice", [F]),
    ("open notepad and type hello world", [L, H, T]), ('type "meeting at 3pm" in notepad', [T]),
    ("open notepad, type todo list and press ctrl+s", [L, H, T, H]),
    ("close notepad", [C]), ("quit chrome", [C]), ("close the calculator", [C]),
    ("press ctrl+s", [H]), ("press alt+tab", [H]), ("hit enter", [H]),
    ("click the save button", [K]), ("click seven", [K]),
    ("open vs code and create a new python file", [L, N, "open_file"]),
    ("create a new javascript file named app.js in vs code", [N, "open_file"]),
    ("switch to chrome", ["focus_app"]),
    ("open chrome go to yt and search gz fanatic", [L, S]), ("bring up the calculator", [L]), ("get rid of notepad", [C]),
    ("open chrome and notepad", [L, L]), ("minimize chrome", ["minimize_app"]), ("maximize notepad", ["maximize_app"]),
    ("scroll down", ["scroll"]), ("go back", [H]), ("refresh", [H]), ("save it", [H]), ("zoom in", [H]),
    ("copy report.pdf from downloads to documents", ["copy_file"]), ("list my downloads", ["list_directory"]),
    ("create a python file called test.py and write hello world", [N, "open_file"]),
    ("turn on dark mode", [W, K, K]), ("go to reddit", [U]), ("open youtube", [U]),
    ("open calculator and click seven and click plus and click eight and click equals", [L, K, K, K, K]),
]

OUT_OF_SCOPE = [
    "write me a poem about the ocean", "what's the weather tomorrow", "summarize the pdf i just downloaded",
    "book a flight to paris", "open the latest report and email it to my boss", "make the text on screen bigger",
    "turn down the volume", "rename all files in downloads to lowercase",
    "log into my bank", "open the second search result", "click the blue button at the top right",
    "reply to the last message", "what is on my screen", "organize my desktop",
]
