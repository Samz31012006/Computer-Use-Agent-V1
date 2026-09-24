"""UIA-based PerceptionEngine: window enumeration (cheap Win32) and element lookup (scoped UIA)."""
from __future__ import annotations

import ctypes
import re
import time
from ctypes import wintypes
from typing import Callable

import psutil
import win32gui
import win32process

from cua.catalog import AppSpec
from cua.types import ElementRef, WinInfo


def _cloaked(hwnd: int) -> bool:
    v = wintypes.DWORD(0)
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(v), ctypes.sizeof(v))
    return bool(v.value)


class UIAPerception:
    def __init__(self, on_time: Callable[[float], None] | None = None):
        self._proc_cache: dict[int, str] = {}
        self._on_time = on_time or (lambda ms: None)

    def _timed(self, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            self._on_time((time.perf_counter() - t0) * 1000)

    def _proc(self, pid: int) -> str:
        if pid not in self._proc_cache:
            try:
                self._proc_cache[pid] = psutil.Process(pid).name()
            except psutil.Error:
                self._proc_cache[pid] = ""
        return self._proc_cache[pid]

    def _info(self, hwnd: int) -> WinInfo:
        pid = win32process.GetWindowThreadProcessId(hwnd)[1]
        return WinInfo(hwnd, win32gui.GetWindowText(hwnd), win32gui.GetClassName(hwnd), pid, self._proc(pid))

    # ---- windows (top of z-order first) ----------------------------------
    def windows(self) -> list[WinInfo]:
        def run():
            out: list[WinInfo] = []

            def cb(h, _):
                if win32gui.IsWindowVisible(h) and win32gui.GetWindowText(h) and not _cloaked(h):
                    out.append(self._info(h))
                return True
            win32gui.EnumWindows(cb, None)
            return [w for w in out if w.title != "Program Manager"]
        return self._timed(run)

    def foreground(self) -> WinInfo | None:
        h = win32gui.GetForegroundWindow()
        return self._timed(lambda: self._info(h)) if h else None

    def match(self, app: AppSpec) -> list[WinInfo]:
        def ok(w: WinInfo) -> bool:
            if app.cls and w.cls not in app.cls:
                return False
            if app.procs and w.proc.lower() in {p.lower() for p in app.procs}:
                return True
            return bool(app.title_re and re.search(app.title_re, w.title, re.I))
        return [w for w in self.windows() if ok(w)]

    # ---- elements --------------------------------------------------------
    def find_element(self, hwnd: int, name: str, timeout: float = 0.0) -> ElementRef | None:
        def run():
            import uiautomation as auto
            root = auto.ControlFromHandle(hwnd)
            esc = re.escape(name.strip())
            for kw in ({"RegexName": f"(?i)^{esc}$"}, {"RegexName": f"(?i).*{esc}.*"}):   # exact first, then substring
                c = auto.Control(searchFromControl=root, **kw)
                if c.Exists(timeout, 0.05):
                    r = c.BoundingRectangle
                    return ElementRef(c.Name, c.ControlTypeName, (r.left, r.top, r.right, r.bottom), c)
            return None
        return self._timed(run)

    def input_ready(self, hwnd: int, timeout: float = 1.0) -> bool:
        """True once keyboard focus is inside `hwnd`. A window can be visible before it accepts input."""
        def run():
            import uiautomation as auto
            end = time.monotonic() + timeout
            while True:
                try:
                    top = auto.GetFocusedControl().GetTopLevelControl()
                    if top is not None and top.NativeWindowHandle == hwnd:
                        return True
                except Exception:
                    pass
                if time.monotonic() >= end:
                    return False
                time.sleep(0.03)
        return self._timed(run)

    def read_text(self, hwnd: int) -> str:
        def run():
            import uiautomation as auto
            root = auto.ControlFromHandle(hwnd)
            for ctype in (auto.DocumentControl, auto.EditControl):
                c = ctype(searchFromControl=root)
                if c.Exists(0, 0):
                    for getter in ("GetTextPattern", "GetValuePattern"):
                        try:
                            p = getattr(c, getter)()
                            if p:
                                return p.DocumentRange.GetText(-1) if getter == "GetTextPattern" else p.Value
                        except Exception:
                            continue
            return ""
        return self._timed(run)

    # ---- bounded tree queries: never walk an unbounded UIA tree (it can hang on huge apps) ----------
    _INTERACTIVE = ("Button", "Edit", "CheckBox", "RadioButton", "ComboBox", "MenuItem", "TabItem", "ListItem",
                    "Hyperlink", "Document", "Slider", "SplitButton")

    def _walk(self, hwnd: int, max_nodes: int = 600, max_depth: int = 10):
        import uiautomation as auto
        root = auto.ControlFromHandle(hwnd)
        for i, (c, depth) in enumerate(auto.WalkControl(root, includeTop=False, maxDepth=max_depth)):
            if i >= max_nodes:
                return
            yield c

    @staticmethod
    def _ref(c) -> ElementRef:
        r = c.BoundingRectangle
        return ElementRef(c.Name, c.ControlTypeName, (r.left, r.top, r.right, r.bottom), c)

    def find_elements(self, hwnd: int, name: str | None = None, control_type: str | None = None,
                      limit: int = 20) -> list[ElementRef]:
        def run():
            out, n, ct = [], (name or "").lower(), (control_type or "").lower()
            for c in self._walk(hwnd):
                if n and n not in c.Name.lower():
                    continue
                if ct and not c.ControlTypeName.lower().startswith(ct):
                    continue
                if c.BoundingRectangle.width() <= 0:
                    continue
                out.append(self._ref(c))
                if len(out) >= limit:
                    break
            return out
        return self._timed(run)

    def inspect(self, hwnd: int, limit: int = 40) -> list[dict]:
        """Compact list of named, interactive controls: what a planner needs to choose a click_ui target."""
        def run():
            out = []
            for c in self._walk(hwnd):
                t = c.ControlTypeName.replace("Control", "")
                if t in self._INTERACTIVE and c.Name and c.BoundingRectangle.width() > 0:
                    out.append({"name": c.Name[:60], "type": t})
                    if len(out) >= limit:
                        break
            return out
        return self._timed(run)

    def element_text(self, hwnd: int, name: str) -> str:
        ref = self.find_element(hwnd, name, 0)
        if ref is None:
            return ""
        def run():
            c = ref.native
            for getter in ("GetValuePattern", "GetTextPattern"):
                try:
                    p = getattr(c, getter)()
                    if p:
                        return p.Value if getter == "GetValuePattern" else p.DocumentRange.GetText(-1)
                except Exception:
                    continue
            return c.Name
        return self._timed(run)

    def set_value(self, hwnd: int, name: str, text: str) -> bool:
        ref = self.find_element(hwnd, name, 0)
        if ref is None:
            return False
        def run():
            try:
                ref.native.GetValuePattern().SetValue(text)
                return True
            except Exception:
                return False
        return self._timed(run)
