"""Window control. focus() walks a ladder of strategies because Windows' foreground lock makes any single one flaky."""
from __future__ import annotations

import ctypes
import time

import win32api
import win32con
import win32gui
import win32process

user32 = ctypes.windll.user32


def _settled_foreground(hwnd: int, wait: float = 0.15) -> bool:
    end = time.monotonic() + wait
    while time.monotonic() < end:
        if win32gui.GetForegroundWindow() == hwnd:
            return True
        time.sleep(0.02)
    return win32gui.GetForegroundWindow() == hwnd


class Windows:
    def focus(self, hwnd: int) -> bool:
        """Restore if minimized, then try: plain -> Alt-tap -> AttachThreadInput -> minimize/restore."""
        if not win32gui.IsWindow(hwnd):
            return False
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        for strategy in (self._plain, self._alt_tap, self._attach_thread, self._minimize_restore):
            try:
                strategy(hwnd)
            except Exception:
                continue
            if _settled_foreground(hwnd):
                return True
        return False

    @staticmethod
    def _plain(hwnd):
        win32gui.SetForegroundWindow(hwnd)

    @staticmethod
    def _alt_tap(hwnd):
        user32.keybd_event(0x12, 0, 0, 0)          # an Alt tap lifts the foreground lock
        user32.keybd_event(0x12, 0, 2, 0)
        win32gui.SetForegroundWindow(hwnd)

    @staticmethod
    def _attach_thread(hwnd):
        fg = win32gui.GetForegroundWindow()
        t_fg = win32process.GetWindowThreadProcessId(fg)[0] if fg else 0
        t_me = win32api.GetCurrentThreadId()
        if t_fg and t_fg != t_me:
            user32.AttachThreadInput(t_me, t_fg, True)
        try:
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
        finally:
            if t_fg and t_fg != t_me:
                user32.AttachThreadInput(t_me, t_fg, False)

    @staticmethod
    def _minimize_restore(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)

    def minimize(self, hwnd: int) -> None:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

    def maximize(self, hwnd: int) -> None:
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

    def restore(self, hwnd: int) -> None:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    def close(self, hwnd: int) -> None:
        """Graceful close. WM_CLOSE first (app may prompt to save); if ignored, ask via UIA like a click.
        A pending dialog ('save changes?') is left for the user: we never answer it."""
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        end = time.monotonic() + 0.6
        while time.monotonic() < end:
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
                return
            if win32gui.GetWindow(hwnd, win32con.GW_ENABLEDPOPUP):
                return
            time.sleep(0.05)
        try:
            import uiautomation as auto
            auto.ControlFromHandle(hwnd).GetWindowPattern().Close()
        except Exception:
            pass   # the window vanishing mid-call raises a COM error; the verifier decides success
