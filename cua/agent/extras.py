"""Broader deterministic understanding: everything the router handles beyond the core open/search/type/click set.

Each handler returns a list of Steps, or NOT_MINE so the router keeps trying its other patterns. Everything is built
through the ToolRegistry (validated, verified, safety-classified), so extending coverage never weakens safety.
"""
from __future__ import annotations

import re

from cua.catalog import folder_key, resolve_app, resolve_site
from cua.types import Step

NOT_MINE = object()

# ---- phrase tables ------------------------------------------------------------------
_SYNONYMS = [
    (r"\b(?:bring up|pull up|fire up|boot up|spin up|load up|start up|get me|give me)\b", "open"),
    (r"\b(?:get rid of|kill|terminate|dismiss|shut)\b(?! down)", "close"),
    (r"\bminimi[sz]e\b", "minimize"), (r"\bmaximi[sz]e\b", "maximize"),
    (r"\b(?:look for|hunt for|track down)\b", "find"), (r"\bpop open\b", "open"),
    (r"\b(?:head over to|take me to|navigate over to)\b", "go to"),
]
_SYN = [(re.compile(p, re.I), r) for p, r in _SYNONYMS]

HOTKEY_PHRASES = {
    "go back": "alt+left", "back": "alt+left", "go forward": "alt+right", "forward": "alt+right",
    "refresh": "f5", "reload": "f5", "refresh the page": "f5", "reload the page": "f5", "new tab": "ctrl+t",
    "close tab": "ctrl+w", "close this tab": "ctrl+w", "reopen tab": "ctrl+shift+t", "next tab": "ctrl+tab",
    "previous tab": "ctrl+shift+tab", "new window": "ctrl+n", "zoom in": "ctrl+plus", "zoom out": "ctrl+minus",
    "reset zoom": "ctrl+0", "select all": "ctrl+a", "copy": "ctrl+c", "copy that": "ctrl+c", "copy this": "ctrl+c",
    "paste": "ctrl+v", "paste it": "ctrl+v", "cut": "ctrl+x", "undo": "ctrl+z", "redo": "ctrl+y", "save": "ctrl+s",
    "save it": "ctrl+s", "save this": "ctrl+s", "save the file": "ctrl+s", "save the document": "ctrl+s",
    "print": "ctrl+p", "find on page": "ctrl+f", "show desktop": "win+d", "task view": "win+tab",
    "take a screenshot": "win+shift+s", "screenshot": "win+shift+s", "full screen": "f11", "fullscreen": "f11",
    "bookmark this": "ctrl+d", "bookmark this page": "ctrl+d", "open downloads": "ctrl+j",
}

_EXT_OF = {"python": "py", "py": "py", "javascript": "js", "js": "js", "node": "js", "typescript": "ts", "ts": "ts",
           "html": "html", "css": "css", "json": "json", "markdown": "md", "md": "md", "text": "txt", "txt": "txt",
           "c": "c", "c++": "cpp", "cpp": "cpp", "java": "java", "rust": "rs", "go": "go", "golang": "go",
           "bash": "sh", "shell": "sh", "powershell": "ps1", "sql": "sql", "csv": "csv", "xml": "xml", "yaml": "yaml"}
_CODE_EXT = {"py", "js", "ts", "html", "css", "json", "c", "cpp", "java", "rs", "go", "sh", "ps1", "sql", "xml", "yaml", "md"}


def hello_world(ext: str, stem: str = "Main") -> str | None:
    """A correct 'hello world' for common languages, so file-creation requests need no model at all."""
    t = {
        "py": 'print("Hello, World!")\n',
        "js": 'console.log("Hello, World!");\n',
        "ts": 'console.log("Hello, World!");\n',
        "html": '<!DOCTYPE html>\n<html>\n  <head><title>Hello</title></head>\n  <body><h1>Hello, World!</h1></body>\n</html>\n',
        "c": '#include <stdio.h>\n\nint main(void) {\n    printf("Hello, World!\\n");\n    return 0;\n}\n',
        "cpp": '#include <iostream>\n\nint main() {\n    std::cout << "Hello, World!" << std::endl;\n    return 0;\n}\n',
        "java": f'public class {stem} {{\n    public static void main(String[] args) {{\n        System.out.println("Hello, World!");\n    }}\n}}\n',
        "rs": 'fn main() {\n    println!("Hello, World!");\n}\n',
        "go": 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("Hello, World!")\n}\n',
        "sh": '#!/usr/bin/env bash\necho "Hello, World!"\n',
        "ps1": 'Write-Host "Hello, World!"\n',
        "sql": "SELECT 'Hello, World!';\n",
        "txt": "Hello, World!\n", "md": "# Hello, World!\n",
    }
    return t.get(ext)


def normalize(text: str) -> str:
    for rx, rep in _SYN:
        text = rx.sub(rep, text)
    return text


def _names(s: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*(?:,|\band\b|&)\s*", s) if p.strip()]


# ---- handlers ---------------------------------------------------------------------------
def handle(router, c: str, rc: dict):
    low = re.sub(r"\s+", " ", c.strip().lower())
    low_clean = re.sub(r"^(?:the|a|my)\s+", "", low)

    # keyboard phrases: "go back", "refresh", "save it", "zoom in", ...
    if low in HOTKEY_PHRASES or low_clean in HOTKEY_PHRASES:
        return [router.mk("hotkey", keys=HOTKEY_PHRASES.get(low) or HOTKEY_PHRASES[low_clean], app=rc.get("app"))]

    # "open a new tab (in notepad)": a browser tab is ctrl+t, Notepad's is ctrl+n
    m = re.match(r"^(?:open\s+)?(?:a\s+)?new tab(?:\s+(?:in|on)\s+(?:the\s+)?(.+))?$", low)
    if m:
        app = resolve_app(m.group(1)) if m.group(1) else None
        key = app.key if app else rc.get("app")
        steps = [] if (not app or rc.get("app") == app.key) else router._open(app.key, app.key, rc)
        return steps + [router.mk("hotkey", keys="ctrl+n" if key == "notepad" else "ctrl+t", app=key)]

    # window state
    m = re.match(r"^(minimize|maximize|restore)\s+(?:the\s+|my\s+)?(.+?)(?:\s+window)?$", low)
    if m:
        name = m.group(2)
        if not (resolve_app(name) or re.fullmatch(r"[\w .-]{2,30}", name)):
            return NOT_MINE
        tool = {"minimize": "minimize_app", "maximize": "maximize_app", "restore": "focus_app"}[m.group(1)]
        return [router.mk(tool, name=(resolve_app(name).key if resolve_app(name) else name))]

    # scrolling
    m = re.match(r"^scroll\s+(up|down)(?:\s+(?:by\s+)?(\d+))?(?:\s+(a bit|a little|a lot|way))?$", low)
    if m:
        n = int(m.group(2)) if m.group(2) else (10 if m.group(3) == "a lot" else 3)
        return [router.mk("scroll", amount=n if m.group(1) == "up" else -n, app=rc.get("app"))]
    m = re.match(r"^(?:scroll|go|jump)\s+to\s+the\s+(top|bottom)$", low)
    if m:
        return [router.mk("hotkey", keys="ctrl+home" if m.group(1) == "top" else "ctrl+end", app=rc.get("app"))]

    # copy / move a file
    m = re.match(r"^(copy|move)\s+(?:the\s+)?(.+?)\s+(?:from\s+(?:my\s+)?(\w+)\s+)?to\s+(?:my\s+|the\s+)?(.+)$", low)
    if m:
        verb, name, src_folder, dst = m.groups()
        src = f"{src_folder}/{name}" if src_folder and folder_key(src_folder) else name
        return [router.mk("copy_file" if verb == "copy" else "move_file", src=src, dst=re.sub(r" folder$", "", dst))]

    # list a folder's contents as text (opening it visually is 'open'/'show')
    m = re.match(r"^(?:list|enumerate)\s+(?:the\s+)?(?:files\s+(?:in|of)\s+)?(?:my\s+)?(\w+)(?:\s+folder)?$", low)
    if m and folder_key(m.group(1)):
        return [router.mk("list_directory", path=folder_key(m.group(1)))]
    m = re.match(r"^(?:show|what'?s|what is)\s+(?:me\s+)?(?:what'?s\s+)?(?:in|inside)\s+(?:my\s+)?(\w+)(?:\s+folder)?$", low)
    if m and folder_key(m.group(1)):
        return [router._folder_step({"folder": folder_key(m.group(1))}, m.group(1), rc)]

    # window queries
    if low in ("what window is open", "what's the window title", "read the window title", "what is the title"):
        return [router.mk("read_window_title")]

    # volume / media / lock: the most common "basic" things to ask a computer, and all of them were previously
    # a router miss -> handed to the LLM -> (on this machine) no RAM -> total failure. They're plain global
    # media keys, so they need no model and no window targeting at all.
    steps = _system_control(router, low_clean)
    if steps is not NOT_MINE:
        return steps

    # system toggle: dark / light mode through the Settings UI (confirmation is enforced by the safety policy)
    m = re.match(r"^(?:turn on|enable|switch to|use|set|activate|change to)\s+(dark|light)\s+(?:mode|theme)$", low) or \
        re.match(r"^(?:turn|switch|set)\s+(?:on\s+)?(dark|light)\s+(?:mode|theme)(?:\s+on)?$", low) or \
        re.match(r"^(?:make it|go)\s+(dark|light)$", low)
    if m:
        rc["app"] = "settings"
        return [router.mk("open_settings", page="colors"),
                router.mk("click_ui", name="Choose your mode", app="settings"),
                router.mk("click_ui", name=m.group(1).capitalize(), app="settings")]

    # open / close several things at once: "open chrome and notepad"
    m = re.match(r"^(open|close)\s+(.+)$", low)
    if m and re.search(r"\s(?:and|&)\s|,", m.group(2)):
        parts = _names(m.group(2))
        if len(parts) > 1 and all(resolve_app(p) or resolve_site(p) for p in parts):
            steps = []
            for p in parts:
                if m.group(1) == "close":
                    steps.append(router.mk("close_app", name=resolve_app(p).key if resolve_app(p) else p))
                else:
                    steps += router._open(p, p, rc) or []
            return steps

    # create a file (optionally with content): "make a python file called test.py with hello world in vs code"
    m = re.match(r"^(?:create|make|write|new|generate)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:new\s+)?(?:simple\s+)?"
                 r"(?:(?P<lang>[\w+#]+)\s+)?(?:file|script|program|document|note)"
                 r"(?:\s+(?:called|named)\s+(?P<name>[\w.-]+))?(?P<tail>.*)$", low)
    if m:
        return _create_file(router, m, c, rc)

    # add content to the file we just created: "write a hello world program", "put hello world in it"
    m = re.match(r"^(?:and\s+)?(?:write|put|add|type|insert)\s+(?:a\s+|the\s+)?(?:simple\s+)?hello,? world"
                 r"(?:\s+(?:program|script|app|code))?(?:\s+(?:in|into|to)\s+it)?$", low)
    if m and rc.get("last_file"):
        ext = rc["last_file"].rsplit(".", 1)[-1]
        body = hello_world(ext, rc["last_file"].rsplit("/", 1)[-1].rsplit(".", 1)[0])
        if body is not None:
            return [Step("_set_content", {"path": rc["last_file"], "content": body})]
    return NOT_MINE


_VOLUME_STEPS = 5      # one press is a 2% change on Windows; 5 is a noticeable but not jarring jump


def _system_control(router, low: str):
    """Volume, media transport and lock -- global media keys, no app targeting, no model."""
    _AUDIO = r"(?:volume|sound|audio)"
    m = (re.match(rf"^(?:turn|make|put|set)?\s*(?:the\s+)?{_AUDIO}\s*(up|down|louder|quieter|lower|higher)$", low)
         or re.match(rf"^(?:turn|make|put|set)\s+(up|down|louder|quieter|lower|higher)\s+(?:the\s+)?{_AUDIO}$", low)
         or re.match(r"^(?:(?:turn|make)\s+(?:it\s+)?)?(louder|quieter)$", low))
    if m:
        word = m.group(1)
        direction = "volume_up" if word in ("up", "louder", "higher") else "volume_down"
        return [router.mk("hotkey", keys=direction) for _ in range(_VOLUME_STEPS)]

    if re.fullmatch(r"(?:mute|unmute)(?:\s+(?:the\s+)?(?:volume|sound|audio|it))?", low) or \
       re.fullmatch(r"(?:turn\s+)?(?:the\s+)?(?:volume|sound|audio)\s+mute", low) or \
       re.fullmatch(r"(?:be\s+)?(?:quiet|silent)", low):
        return [router.mk("hotkey", keys="volume_mute")]

    # "play despacito on youtube" / "play lofi beats" -- a search on the site, not a media key. Checked before
    # the bare transport words below so "play" alone still means play/pause.
    m = re.match(r"^play\s+(?P<q>.+?)(?:\s+(?:on|in|using)\s+(?P<site>youtube|yt|spotify))?$", low)
    if m and m.group("q") not in ("it", "music", "the music", "song", "the song"):
        site = (m.group("site") or "youtube").lower()
        engine = "youtube" if site in ("youtube", "yt") else site
        if engine == "youtube":
            return [router.mk("web_search", query=m.group("q"), engine="youtube")]

    media = {"play": "media_play_pause", "pause": "media_play_pause", "resume": "media_play_pause",
             "play it": "media_play_pause", "pause it": "media_play_pause",
             "play the music": "media_play_pause", "pause the music": "media_play_pause",
             "play music": "media_play_pause", "pause music": "media_play_pause",
             "next track": "media_next", "next song": "media_next", "skip": "media_next",
             "skip song": "media_next", "skip the song": "media_next", "skip this song": "media_next",
             "previous track": "media_prev", "previous song": "media_prev", "last song": "media_prev",
             "stop the music": "media_stop", "stop music": "media_stop"}
    if low in media:
        return [router.mk("hotkey", keys=media[low])]

    if re.fullmatch(r"lock(?:\s+(?:the\s+)?(?:pc|computer|screen|laptop|it))?", low):
        return [router.mk("hotkey", keys="win+l")]
    return NOT_MINE


def _create_file(router, m, raw: str, rc: dict):
    lang, name, tail = m.group("lang"), m.group("name"), m.group("tail") or ""
    orig = re.search(r"(?:called|named)\s+([\w.-]+)", raw, re.I)      # keep the user's capitalisation (Greeter.java)
    if name and orig:
        name = orig.group(1)
    ext = _EXT_OF.get((lang or "").lower())
    if lang and ext is None and lang.lower() not in ("new", "a", "an", "the", "simple"):
        return NOT_MINE                                # "create a spreadsheet file": not something we can make blindly
    if name and "." in name:
        ext = name.rsplit(".", 1)[-1].lower()
    ext = ext or "txt"
    fn = name if name and "." in name else f"{name or 'untitled'}.{ext}"
    path = f"documents/{fn}"
    stem = fn.rsplit(".", 1)[0]
    content = ""
    hw = re.search(r"hello,? world", tail)
    quoted = re.search(r'(?:saying|says|containing|with(?: the)? (?:text|content)|that says)\s+"?([^"]+?)"?\s*(?:in\s+.*)?$', tail)
    if hw:
        content = hello_world(ext, stem) or ""
    elif quoted:
        content = quoted.group(1).strip() + "\n"
    app = None
    am = re.search(r"\b(?:in|with|using)\s+(vs ?code|visual studio code|notepad)\b", tail)
    if am:
        app = resolve_app(am.group(1)).key
    elif ext in _CODE_EXT:
        app = "vscode"
    rc["last_file"] = path
    steps = [router.mk("create_file", path=path, content=content or None)]
    steps.append(router.mk("open_file", path=path, app=app))
    return steps
