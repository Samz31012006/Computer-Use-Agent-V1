"""Verification predicates. Each is (spec, ctx) -> (ok, detail) and is cheap: window titles, process state, files,
one scoped UIA lookup. None of them calls a model or takes a screenshot.

Spec kinds:
  app_window      {app}                         a window of the app exists
  foreground_app  {app}                         the app's window is in front
  foreground_title{any_of}                      the front window's title contains one of the strings
  window_title    {any_of, app?}                some (or the app's) window title contains one of the strings
  window_state    {app, state}                  minimized | maximized | normal
  last_hwnd_gone  {}                            the window recorded by close_app is gone
  file_exists     {path | path_text | from}     file exists (from = context key holding a path)
  open_file       {}                            the opened file's name shows in a window title
  element         {name, app?}                  UIA element with that name exists
  element_text    {name, contains, app?}        that element's text contains a string
  text_in_window  {contains, app?}              the window's text contains a string
  screen_text     {contains, app?}              OCR: the visible screen/window text contains a string --
                                                 last resort for canvas-rendered content UIA can't see; one
                                                 OCR pass, so only used when a verify spec explicitly asks
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import win32con
import win32gui

from cua.catalog import generic_app, resolve_app

Result = tuple[bool, str]


def _app(key: str):
    return resolve_app(key) or generic_app(key)


def target_hwnd(spec: dict, ctx) -> int | None:
    if spec.get("app"):
        last = ctx.state.context.get("last_hwnd")
        ws = ctx.perception.match(_app(spec["app"]))
        if last and any(w.hwnd == last for w in ws):
            return last
        return ws[0].hwnd if ws else None
    fg = ctx.perception.foreground()
    return fg.hwnd if fg else None


def app_window(spec, ctx) -> Result:
    ws = ctx.perception.match(_app(spec["app"]))
    return bool(ws), f"{len(ws)} window(s) of {spec['app']}"


def foreground_app(spec, ctx) -> Result:
    fg = ctx.perception.foreground()
    ws = {w.hwnd for w in ctx.perception.match(_app(spec["app"]))}
    return bool(fg and fg.hwnd in ws), f"foreground '{fg.title if fg else None}'"


def foreground_title(spec, ctx) -> Result:
    fg = ctx.perception.foreground()
    t = fg.title.lower() if fg else ""
    return any(s.lower() in t for s in spec["any_of"]), f"foreground '{t}'"


def window_title(spec, ctx) -> Result:
    ws = ctx.perception.match(_app(spec["app"])) if spec.get("app") else ctx.perception.windows()
    subs = [s.lower() for s in spec["any_of"]]
    for w in ws:
        if any(s in w.title.lower() for s in subs):
            return True, f"title '{w.title}'"
    return False, f"no title containing any of {spec['any_of']}"


def window_state(spec, ctx) -> Result:
    ws = ctx.perception.match(_app(spec["app"]))
    if not ws:
        return False, "no window"
    h = ws[0].hwnd
    state = ("minimized" if win32gui.IsIconic(h) else
             "maximized" if win32gui.GetWindowPlacement(h)[1] == win32con.SW_SHOWMAXIMIZED else "normal")
    return state == spec["state"], state


def last_hwnd_gone(spec, ctx) -> Result:
    h = ctx.state.context.get("closed_hwnd")
    return (not win32gui.IsWindow(h) or not win32gui.IsWindowVisible(h)), f"hwnd {h}"


def file_exists(spec, ctx) -> Result:
    path = spec.get("path") or ctx.state.context.get(spec.get("from", ""), "")
    if not path and spec.get("path_text"):
        from cua.actions.paths import resolve_user_path
        path = str(resolve_user_path(spec["path_text"], ctx.folders))
    return bool(path) and Path(path).exists(), str(path)


def open_file(spec, ctx) -> Result:
    p = ctx.state.context.get("opened_file")
    if not p:
        return True, "no file recorded"
    stem = Path(p).stem.lower()
    return any(stem in w.title.lower() for w in ctx.perception.windows()), f"window titled like '{stem}'"


def element(spec, ctx) -> Result:
    hwnd = target_hwnd(spec, ctx)
    if not hwnd:
        return False, "no target window"
    return ctx.perception.find_element(hwnd, spec["name"], 0) is not None, f"element '{spec['name']}'"


def element_text(spec, ctx) -> Result:
    hwnd = target_hwnd(spec, ctx)
    if not hwnd:
        return False, "no target window"
    txt = ctx.perception.element_text(hwnd, spec["name"])
    return spec["contains"] in txt, f"text len={len(txt)}"


def text_in_window(spec, ctx) -> Result:
    hwnd = target_hwnd(spec, ctx)
    if not hwnd:
        return False, "no target window"
    txt = ctx.perception.read_text(hwnd)
    return spec["contains"] in txt, f"text len={len(txt)}"


def screen_text(spec, ctx) -> Result:
    from cua.perception.ocr import WindowsOcr
    hwnd = target_hwnd(spec, ctx)
    ocr = WindowsOcr()
    if not ocr.available:
        return False, "OCR not available"
    words = ocr.recognize(hwnd)
    txt = " ".join(w[0] for w in words).lower()
    return spec["contains"].lower() in txt, f"OCR text len={len(txt)}"


PREDICATES: dict[str, Callable[[dict, object], Result]] = {
    "app_window": app_window, "foreground_app": foreground_app, "foreground_title": foreground_title,
    "window_title": window_title, "window_state": window_state, "last_hwnd_gone": last_hwnd_gone,
    "file_exists": file_exists, "open_file": open_file, "element": element, "element_text": element_text,
    "text_in_window": text_in_window, "screen_text": screen_text,
}
