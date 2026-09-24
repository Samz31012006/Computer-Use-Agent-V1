"""Declarations of every tool the agent can use. Metadata only; implementations live in the engines.

Adding a capability = declare it here + implement it in an engine. The router and the LLM planner both pick from
this list, and neither can call anything that isn't in it.
"""
from __future__ import annotations

import re
from pathlib import Path

from cua.actions.base import ArgSpec as A, Safety, ToolError, ToolSpec as T
from cua.actions.paths import resolve_user_path
from cua.catalog import SEARCH_ENGINES, SETTINGS_PAGES, folder_key
from cua.computer.keyboard import normalize_keys

_URL_OK = re.compile(r"^https?://[^\s]+$", re.I)
_BAD_SCHEME = re.compile(r"^(javascript|data|file|vbscript|about|chrome|edge|ms-[\w-]+|mailto|tel|ftp|blob):", re.I)
_BROWSERS = ("chrome", "edge", "firefox")
APP = A("app", required=False, description="app to focus first")


# ---- argument normalisers ------------------------------------------------------
def _prep_url(a):
    u = a["url"]
    if _BAD_SCHEME.match(u):
        raise ToolError("navigate_url: only http(s) URLs are allowed")
    if not re.match(r"^[a-z][a-z0-9+.-]*://", u, re.I):
        u = "https://" + u
    if not _URL_OK.match(u):
        raise ToolError("navigate_url: only http(s) URLs are allowed")
    a["url"] = u
    return a


def _prep_keys(a):
    k = normalize_keys(a.get("keys") or a.get("key", ""))
    if not k:
        raise ToolError(f"unknown key(s): {a.get('keys') or a.get('key')}")
    a["keys" if "keys" in a else "key"] = k
    return a


def _prep_engine(a):
    a.setdefault("engine", "google")
    return a


def _prep_find(a):
    if a.get("extension"):
        a["extension"] = a["extension"].lower().lstrip(".")
    if not (a.get("name") or a.get("extension") or a.get("modified")):
        raise ToolError("find_file: give a name, an extension or a modified date")
    return a


def _prep_wait(a):
    a["seconds"] = max(0.0, min(30.0, a.get("seconds", 1.0)))
    return a


def _prep_folder(a):
    if not (a.get("folder") or a.get("path")):
        raise ToolError("open_folder: give folder or path")
    if a.get("folder") and not folder_key(a["folder"]):
        # a planner may put a path in 'folder'
        a["path"], a["folder"] = a.pop("folder"), None
        a = {k: v for k, v in a.items() if v is not None}
    elif a.get("folder"):
        a["folder"] = folder_key(a["folder"])
    return a


def _prep_ui(a):
    if not (a.get("name") or a.get("control_type")):
        raise ToolError("give a name or a control_type")
    return a


# ---- verification builders (args -> predicate spec) ----------------------------
def _v_open_app(a): return {"kind": "app_window", "app": a["name"], "timeout": 10}
def _v_focus(a): return {"kind": "foreground_app", "app": a["name"], "timeout": 3}
def _v_minimize(a): return {"kind": "window_state", "app": a["name"], "state": "minimized", "timeout": 2}
def _v_maximize(a): return {"kind": "window_state", "app": a["name"], "state": "maximized", "timeout": 2}
def _v_close(a): return {"kind": "last_hwnd_gone", "timeout": 5}
def _v_type(a):
    return {"kind": "text_in_window", "contains": a["text"][:40], "app": a.get("app"), "timeout": 3} \
        if len(a["text"]) <= 400 and "\n" not in a["text"][:40] else None
def _v_find_window(a): return {"kind": "window_title", "any_of": [a["title"]], "timeout": 2}
def _v_switch(a): return {"kind": "foreground_title", "any_of": [a["title"]], "timeout": 3}
def _v_click_ui(a):
    return {"kind": "element", "name": a["expect"], "timeout": 3} if a.get("expect") else None
def _v_type_ui(a): return {"kind": "element_text", "name": a["name"], "contains": a["text"][:40], "app": a.get("app"), "timeout": 3}
def _v_find_file(a): return {"kind": "file_exists", "from": "found_file", "timeout": 2}
def _v_open_file(a): return {"kind": "open_file", "timeout": 8}
def _v_create(a): return {"kind": "file_exists", "from": "created_file", "timeout": 2}
def _v_copy(a): return {"kind": "file_exists", "from": "dest_file", "timeout": 2}
def _v_open_folder(a):
    title = Path.home().name if a.get("folder") == "home" else (a.get("folder") or Path(a["path"]).name)
    return {"kind": "window_title", "any_of": [title], "app": "explorer", "timeout": 6}
def _v_url(a):
    host = re.sub(r"^https?://(www\.)?", "", a["url"]).split("/")[0]
    return {"kind": "window_title", "any_of": [host.split(".")[0]], "timeout": 10}
def _v_search(a):
    q = a["query"]
    toks = [t for t in re.findall(r"\w+", q.lower()) if len(t) >= 4][:3]
    return {"kind": "window_title", "any_of": [q.lower()] + toks + [SEARCH_ENGINES[a["engine"]][1][0]], "timeout": 12}
def _v_settings(a): return {"kind": "window_title", "any_of": ["Settings"], "app": "settings", "timeout": 6}


def _confirm_create(a):
    try:
        return f"overwrite {a['path']}" if resolve_user_path(a["path"]).exists() else None
    except ToolError:
        return None


def _confirm_copy(a):
    try:
        dst = resolve_user_path(a["dst"])
        if dst.is_dir():                       # copying INTO a folder: only a same-named file would be overwritten
            dst = dst / Path(a["src"]).name
        return f"overwrite {dst.name}" if dst.exists() else None
    except ToolError:
        return None


def build_tools() -> list[ToolSpec]:
    S, C = Safety.SAFE, Safety.CONFIRM
    engines = tuple(SEARCH_ENGINES)
    return [
        # -- applications
        T("open_app", "Open a Windows application by name", (A("name", description="e.g. chrome, notepad, calculator"),),
          verify=_v_open_app),
        T("close_app", "Close an application's top window", (A("name"),), safety=C, idempotent=False, verify=_v_close),
        T("focus_app", "Bring an open application to the front", (A("name"),), verify=_v_focus),
        T("minimize_app", "Minimize an application's window", (A("name"),), verify=_v_minimize),
        T("maximize_app", "Maximize an application's window", (A("name"),), verify=_v_maximize),
        # -- keyboard
        T("type_text", "Type text into the focused control", (A("text", strip=False), APP), idempotent=False, verify=_v_type),
        T("press_key", "Press one key, e.g. enter, tab, esc", (A("key"), APP), idempotent=False, prepare=_prep_keys),
        T("hotkey", "Press a key combination, e.g. ctrl+s", (A("keys"), APP), idempotent=False, prepare=_prep_keys),
        # -- mouse
        T("click", "Left-click at screen coordinates", (A("x", "int"), A("y", "int"), APP), idempotent=False),
        T("double_click", "Double-click at screen coordinates", (A("x", "int"), A("y", "int"), APP), idempotent=False),
        T("right_click", "Right-click at screen coordinates", (A("x", "int"), A("y", "int"), APP), idempotent=False),
        T("move_mouse", "Move the pointer to screen coordinates", (A("x", "int"), A("y", "int"))),
        T("scroll", "Scroll the wheel; positive is up, negative is down",
          (A("amount", "int"), A("x", "int", False), A("y", "int", False), APP), idempotent=False),
        # -- windows
        T("find_window", "Check that a window whose title contains the text exists", (A("title"),), verify=_v_find_window),
        T("switch_window", "Bring the window whose title contains the text to the front", (A("title"),),
          verify=_v_switch),
        T("read_window_title", "Read the title of the focused (or the named app's) window", (APP,)),
        T("inspect_controls", "List the clickable/typeable controls of a window", (APP,)),
        # -- UI elements (UI Automation)
        T("click_ui", "Click a UI element by name and/or control type",
          (A("name", required=False), A("control_type", required=False, description="button, edit, checkbox, ..."),
           APP, A("expect", required=False, description="element name that should appear after the click")),
          idempotent=False, verify=_v_click_ui, prepare=_prep_ui),
        T("type_into_ui", "Type text into a named UI element", (A("name"), A("text", strip=False), APP), idempotent=False,
          verify=_v_type_ui),
        T("read_ui_text", "Read the text/value of a named UI element", (A("name", required=False), APP)),
        T("find_ui", "Check that a UI element exists", (A("name", required=False), A("control_type", required=False), APP),
          prepare=_prep_ui),
        # -- files & folders
        T("find_file", "Find the newest file matching name/extension/modified",
          (A("name", required=False, description="words in the file name"), A("extension", required=False),
           A("folder", required=False, description="downloads, documents, desktop, ... or a path"),
           A("modified", required=False, description="today or yesterday", enum=("today", "yesterday"))),
          verify=_v_find_file, prepare=_prep_find),
        T("open_file", "Open a file (the last found file if no path), optionally with a given app",
          (A("path", required=False), A("app", required=False)), verify=_v_open_file),
        T("create_file", "Create a text/code file with content in the user's folders",
          (A("path", description="e.g. documents/test.py"), A("content", required=False, strip=False)), idempotent=False,
          verify=_v_create, confirm_if=_confirm_create),
        T("copy_file", "Copy a file", (A("src"), A("dst")), idempotent=False, verify=_v_copy, confirm_if=_confirm_copy),
        T("move_file", "Move a file", (A("src"), A("dst")), safety=C, idempotent=False, verify=_v_copy),
        T("open_folder", "Open a known folder (downloads, documents, ...) or a path in Explorer",
          (A("folder", required=False), A("path", required=False)), verify=_v_open_folder, prepare=_prep_folder,
          primary="folder"),
        T("list_directory", "List the files in a folder", (A("path", required=False, description="folder or path"),)),
        # -- browser / web
        T("navigate_url", "Open an http(s) URL in a browser", (A("url"), A("browser", required=False, enum=_BROWSERS)),
          verify=_v_url, prepare=_prep_url),
        T("web_search", "Search the web", (A("query"), A("engine", required=False, enum=engines),
                                           A("browser", required=False, enum=_BROWSERS)),
          verify=_v_search, prepare=_prep_engine),
        T("open_settings", "Open a Windows Settings page", (A("page", enum=tuple(SETTINGS_PAGES)),), verify=_v_settings),
        T("wait", "Wait a number of seconds", (A("seconds", "float"),), prepare=_prep_wait),
    ]
