"""SkillEngine: deterministic OS-level implementations of the declared tools (no UIA tree walking, no vision).

Span conventions (used by the benchmark): select = resolving a target, act = input/launch calls,
settle = waiting for a launched window to appear.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from cua.actions.base import ToolError
from cua.actions.paths import resolve_user_path
from cua.agent.state import Ctx, own_pids
from cua.catalog import (SETTINGS_PAGES, AppSpec, folder_key, generic_app, known_folder, resolve_app,
                         search_url)
from cua.types import EngineResult, Step

_EXT_ALIASES = {"jpg": ("jpg", "jpeg"), "jpeg": ("jpg", "jpeg"), "image": ("png", "jpg", "jpeg", "gif", "webp"),
                "doc": ("doc", "docx"), "docx": ("doc", "docx"), "xls": ("xls", "xlsx"), "xlsx": ("xls", "xlsx"),
                "ppt": ("ppt", "pptx"), "pptx": ("ppt", "pptx")}
_DEFAULT_ROOTS = ("desktop", "documents", "downloads")


def _app(name: str) -> AppSpec:
    return resolve_app(name) or generic_app(name)


def topmost_window(ctx: Ctx, name: str):
    """Topmost window of an app, never one owned by the agent's own host process."""
    ws = [w for w in ctx.perception.match(_app(name)) if w.pid not in own_pids()]
    return ws[0] if ws else None


class SkillEngine:
    name = "skill"

    def __init__(self):
        self.handlers = {n[1:]: getattr(self, n) for n in dir(self)
                         if n.startswith("_") and not n.startswith("__") and n[1:] in _TOOL_NAMES}

    def can_handle(self, step: Step) -> bool:
        return step.action in self.handlers

    def execute(self, step: Step, ctx: Ctx) -> EngineResult:
        return self.handlers[step.action](step.args, ctx)

    # ---- applications ----------------------------------------------------------
    def _open_app(self, a, ctx: Ctx, params: str | None = None) -> EngineResult:
        with ctx.log.span("select"):
            spec = _app(a["name"])
            before = {w.hwnd for w in ctx.perception.match(spec)}
        with ctx.log.span("act"):
            ctx.driver.launch(spec, params)
        hwnd = self._await_window(spec, before, ctx)
        if hwnd is None:
            return EngineResult(False, f"no window for {spec.key} after launch")
        with ctx.log.span("act"):
            ctx.driver.focus(hwnd)
        ctx.state.context.update(last_app=spec.key, last_hwnd=hwnd)
        return EngineResult(True, f"{spec.key} hwnd={hwnd}")

    def _await_window(self, spec: AppSpec, before: set[int], ctx: Ctx, timeout=10.0, grace=1.5):
        """A new window is accepted immediately; an existing one only after `grace` (single-instance apps)."""
        with ctx.log.span("settle"):
            t0, delay = time.monotonic(), 0.04
            while time.monotonic() - t0 < timeout:
                ctx.check_kill()
                ws = ctx.perception.match(spec)
                new = [w for w in ws if w.hwnd not in before]
                if new:
                    return new[0].hwnd
                if ws and time.monotonic() - t0 >= grace:
                    return ws[0].hwnd
                time.sleep(delay)
                delay = min(0.15, delay * 1.4)
        return None

    def _close_app(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            w = topmost_window(ctx, a["name"])      # one window only, never "all windows"
        if not w:
            return EngineResult(False, f"no window for {a['name']}")
        ctx.state.context["closed_hwnd"] = w.hwnd
        with ctx.log.span("act"):
            ctx.driver.close(w.hwnd)
        return EngineResult(True, f"closed '{w.title}'")

    def _focus_app(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            w = topmost_window(ctx, a["name"])
        if not w:
            return EngineResult(False, f"no window for {a['name']}")
        with ctx.log.span("act"):
            ok = ctx.driver.focus(w.hwnd)
        ctx.state.context.update(last_app=_app(a["name"]).key, last_hwnd=w.hwnd)
        return EngineResult(ok, w.title)

    def _window_op(self, a, ctx: Ctx, op: str) -> EngineResult:
        with ctx.log.span("select"):
            w = topmost_window(ctx, a["name"])
        if not w:
            return EngineResult(False, f"no window for {a['name']}")
        with ctx.log.span("act"):
            getattr(ctx.driver, op)(w.hwnd)
        return EngineResult(True, f"{op} '{w.title}'")

    def _minimize_app(self, a, ctx): return self._window_op(a, ctx, "minimize")
    def _maximize_app(self, a, ctx): return self._window_op(a, ctx, "maximize")

    # ---- windows ---------------------------------------------------------------
    def _find_window(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            t = a["title"].lower()
            hit = next((w for w in ctx.perception.windows() if t in w.title.lower()), None)
        if not hit:
            return EngineResult(False, f"no window titled like '{a['title']}'")
        ctx.state.context["found_window"] = hit.hwnd
        return EngineResult(True, hit.title)

    def _switch_window(self, a, ctx: Ctx) -> EngineResult:
        r = self._find_window(a, ctx)
        if not r.ok:
            return r
        with ctx.log.span("act"):
            ok = ctx.driver.focus(ctx.state.context["found_window"])
        ctx.state.context["last_hwnd"] = ctx.state.context["found_window"]
        return EngineResult(ok, r.detail)

    def _read_window_title(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            w = topmost_window(ctx, a["app"]) if a.get("app") else ctx.perception.foreground()
        if not w:
            return EngineResult(False, "no window")
        ctx.state.context["window_title"] = w.title
        return EngineResult(True, w.title)

    # ---- input -----------------------------------------------------------------
    def _focus_target(self, a, ctx: Ctx) -> EngineResult | int:
        """Resolve and focus the window that receives input, launching the target app first if it isn't
        already open ("type hello on notepad" works whether or not Notepad was already running). Refuses to
        touch the agent's own host window."""
        key = a.get("app") or None
        if key:
            with ctx.log.span("select"):
                ws = ctx.perception.match(_app(key))
            if not ws:
                launched = self._open_app({"name": key}, ctx)      # not open yet: launch it, then target it
                if not launched.ok:
                    return launched
                hwnd = ctx.state.context.get("last_hwnd")
            else:
                with ctx.log.span("select"):
                    last = ctx.state.context.get("last_hwnd")
                    hwnd = next((w.hwnd for w in ws if w.hwnd == last), ws[0].hwnd)
                with ctx.log.span("act"):
                    ctx.driver.focus(hwnd)
            fg = ctx.perception.foreground()
        else:
            with ctx.log.span("select"):
                fg = ctx.perception.foreground()
                hwnd = fg.hwnd if fg else 0
        if not fg or fg.hwnd != hwnd:
            return EngineResult(False, f"could not focus target (foreground='{fg.title if fg else None}')")
        if fg.pid in own_pids():
            return EngineResult(False, "refusing to send input to the agent's own terminal/host window")
        ready = getattr(ctx.perception, "input_ready", None)
        if ready:
            with ctx.log.span("settle"):
                if not ready(hwnd):
                    ctx.log.event("input_not_ready", hwnd=hwnd)   # proceed; the verifier will catch a lost keystroke
        ctx.state.context["input_hwnd"] = hwnd
        return hwnd

    def _typed(self, a, ctx: Ctx, fn) -> EngineResult:
        t = self._focus_target(a, ctx)
        if isinstance(t, EngineResult):
            return t
        with ctx.log.span("act"):
            fn()
        return EngineResult(True)

    def _type_text(self, a, ctx):
        r = self._typed(a, ctx, lambda: ctx.driver.type_text(a["text"]))
        return EngineResult(r.ok, r.detail or f"typed {len(a['text'])} chars")

    def _press_key(self, a, ctx): return self._typed(a, ctx, lambda: ctx.driver.hotkey(a["key"]))
    def _hotkey(self, a, ctx): return self._typed(a, ctx, lambda: ctx.driver.hotkey(a["keys"]))

    def _mouse(self, a, ctx: Ctx, **kw) -> EngineResult:
        if a.get("app"):
            r = self._focus_target(a, ctx)
            if isinstance(r, EngineResult):
                return r
        with ctx.log.span("act"):
            ctx.driver.click(a["x"], a["y"], **kw)
        return EngineResult(True, f"({a['x']},{a['y']})")

    def _click(self, a, ctx): return self._mouse(a, ctx)
    def _double_click(self, a, ctx): return self._mouse(a, ctx, double=True)
    def _right_click(self, a, ctx): return self._mouse(a, ctx, button="right")

    def _move_mouse(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("act"):
            ctx.driver.move(a["x"], a["y"])
        return EngineResult(True)

    def _scroll(self, a, ctx: Ctx) -> EngineResult:
        if a.get("app"):
            r = self._focus_target(a, ctx)
            if isinstance(r, EngineResult):
                return r
        with ctx.log.span("act"):
            ctx.driver.scroll(a["amount"], a.get("x"), a.get("y"))
        return EngineResult(True)

    # ---- files & folders -------------------------------------------------------
    def _open_folder(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            path = resolve_user_path(a["path"], ctx.folders) if a.get("path") else known_folder(a["folder"], ctx.folders)
            if not path.is_dir():
                return EngineResult(False, f"not a folder: {path}")
        with ctx.log.span("act"):
            ctx.driver.open_path(str(path))
        ctx.state.context.update(last_app="explorer", folder=str(path))
        return EngineResult(True, str(path))

    def _list_directory(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            p = resolve_user_path(a["path"], ctx.folders) if a.get("path") else Path(ctx.state.context.get("folder", "."))
            if not p.is_dir():
                return EngineResult(False, f"not a folder: {p}")
            names = sorted(e.name for e in os.scandir(p))[:50]
        ctx.state.context.update(listing=names, folder=str(p))
        return EngineResult(True, ", ".join(names[:30]))

    def _find_file(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            roots = self._roots(a.get("folder"), ctx)
            exts = set()
            if a.get("extension"):
                exts.update(_EXT_ALIASES.get(a["extension"], (a["extension"],)))
            since = {"today": date.today(), "yesterday": date.today() - timedelta(days=1)}.get(a.get("modified"))
            words = [w.lower() for w in re.findall(r"[\w.-]+", a.get("name", ""))]
            hit = _scan(roots, exts, words, since)
        if not hit:
            return EngineResult(False, f"no file matching {a} in {[str(r) for r in roots]}")
        ctx.state.context["found_file"] = str(hit)
        with ctx.log.span("act"):
            ctx.driver.reveal(str(hit))
        ctx.state.context.update(last_app="explorer", folder=str(hit.parent))
        return EngineResult(True, str(hit))

    @staticmethod
    def _roots(folder: str | None, ctx: Ctx) -> list[Path]:
        if not folder:
            return [known_folder(k, ctx.folders) for k in _DEFAULT_ROOTS]
        k = folder_key(folder)
        return [known_folder(k, ctx.folders)] if k else [resolve_user_path(folder, ctx.folders)]

    def _open_file(self, a, ctx: Ctx) -> EngineResult:
        path = resolve_user_path(a["path"], ctx.folders) if a.get("path") else Path(ctx.state.context.get("found_file", ""))
        if not path or not path.is_file():
            return EngineResult(False, "no file to open")
        ctx.state.context["opened_file"] = str(path)
        if a.get("app"):
            return self._open_app({"name": a["app"]}, ctx, params=str(path))
        with ctx.log.span("act"):
            ctx.driver.open_path(str(path))
        return EngineResult(True, str(path))

    def _create_file(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("select"):
            p = resolve_user_path(a["path"], ctx.folders)
        with ctx.log.span("act"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(a.get("content", ""), encoding="utf-8", newline="")
        ctx.state.context.update(created_file=str(p), found_file=str(p))
        return EngineResult(True, str(p))

    def _copy_move(self, a, ctx: Ctx, fn) -> EngineResult:
        with ctx.log.span("select"):
            src, dst = resolve_user_path(a["src"], ctx.folders), resolve_user_path(a["dst"], ctx.folders)
            if not src.exists():
                return EngineResult(False, f"source missing: {src}")
            if dst.is_dir():
                dst = dst / src.name
        with ctx.log.span("act"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            fn(str(src), str(dst))
        ctx.state.context["dest_file"] = str(dst)
        return EngineResult(True, str(dst))

    def _copy_file(self, a, ctx): return self._copy_move(a, ctx, shutil.copy2)
    def _move_file(self, a, ctx): return self._copy_move(a, ctx, shutil.move)

    # ---- browser / web / settings ----------------------------------------------
    def _navigate_url(self, a, ctx: Ctx) -> EngineResult:
        browser = _app(a["browser"]) if a.get("browser") else None
        with ctx.log.span("act"):
            ctx.driver.open_uri(a["url"], browser)
        if browser:
            ctx.state.context["last_app"] = browser.key
        return EngineResult(True, a["url"])

    def _web_search(self, a, ctx: Ctx) -> EngineResult:
        return self._navigate_url({"url": search_url(a["engine"], a["query"]), "browser": a.get("browser")}, ctx)

    def _open_settings(self, a, ctx: Ctx) -> EngineResult:
        with ctx.log.span("act"):
            ctx.driver.open_uri(SETTINGS_PAGES[a["page"]])
        ctx.state.context["last_app"] = "settings"
        return EngineResult(True, a["page"])

    def _wait(self, a, ctx: Ctx) -> EngineResult:
        ctx.sleep(float(a.get("seconds", 1)))
        return EngineResult(True)


_TOOL_NAMES = {"open_app", "close_app", "focus_app", "minimize_app", "maximize_app", "type_text", "press_key", "hotkey",
               "click", "double_click", "right_click", "move_mouse", "scroll", "find_window", "switch_window",
               "read_window_title", "find_file", "open_file", "create_file", "copy_file",
               "move_file", "open_folder", "list_directory", "navigate_url", "web_search", "open_settings", "wait"}


def _scan(roots, exts, words, since, max_depth=3, max_entries=30000):
    """Newest matching file under roots; bounded by depth and entry count so a huge tree can't stall the agent."""
    best, best_m, seen = None, -1.0, 0
    for root in roots:
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                it = list(os.scandir(d))
            except OSError:
                continue
            for e in it:
                seen += 1
                if seen > max_entries:
                    return best
                if e.name.startswith(".") or e.name.startswith("~$"):
                    continue
                if e.is_dir(follow_symlinks=False):
                    if depth < max_depth:
                        stack.append((Path(e.path), depth + 1))
                    continue
                low = e.name.lower()
                if exts and low.rsplit(".", 1)[-1] not in exts:
                    continue
                if words and not all(w in low for w in words):
                    continue
                m = e.stat().st_mtime
                if since and datetime.fromtimestamp(m).date() < since:
                    continue
                if m > best_m:
                    best, best_m = Path(e.path), m
    return best
