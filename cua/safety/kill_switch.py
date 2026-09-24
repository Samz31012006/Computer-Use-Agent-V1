"""Emergency stop: global hotkey (default Ctrl+Alt+Q) that every loop in the agent checks."""
from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes

from cua.types import Aborted


MOD = {"ctrl": 0x2, "alt": 0x1, "shift": 0x4, "win": 0x8}


class KillSwitch:
    """Global hotkey (default Ctrl+Alt+Q). Sets an event that every loop in the agent checks.

    If the process hasn't acknowledged within `grace` seconds (stuck in a blocking call),
    the watchdog hard-exits it so the emergency stop is real, not advisory.
    """

    def __init__(self, hotkey: str | None = None, grace: float = 2.0):
        self.hotkey = (hotkey or os.environ.get("CUA_KILL_HOTKEY", "ctrl+alt+q")).lower()
        self.grace = grace
        self.event = threading.Event()
        self._ack = threading.Event()
        self._thread: threading.Thread | None = None
        self._tid = 0
        self.registered = threading.Event()
        self.error: str | None = None
        self.hard_exit = True            # tests turn this off
        self.busy = threading.Event()    # set by Agent.run; the hotkey only fires while a task is running

    def start(self) -> "KillSwitch":
        self._thread = threading.Thread(target=self._loop, daemon=True, name="kill-hotkey")
        self._thread.start()
        self.registered.wait(2)
        return self

    def stop(self):
        if self._tid:
            ctypes.windll.user32.PostThreadMessageW(self._tid, 0x0012, 0, 0)  # WM_QUIT

    def _loop(self):
        u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
        self._tid = k32.GetCurrentThreadId()
        parts = self.hotkey.split("+")
        mods = sum(MOD[p] for p in parts[:-1]) | 0x4000  # MOD_NOREPEAT
        key = parts[-1]
        vk = ord(key.upper()) if len(key) == 1 else {"esc": 0x1B, "backspace": 0x08, "pause": 0x13}[key]
        if not u32.RegisterHotKey(None, 1, mods, vk):
            self.error = f"RegisterHotKey failed ({ctypes.GetLastError()}); hotkey {self.hotkey} in use?"
            self.registered.set()
            return
        self.registered.set()
        msg = wintypes.MSG()
        while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312 and self.busy.is_set():   # WM_HOTKEY; idle presses are ignored
                self.trigger()
        u32.UnregisterHotKey(None, 1)

    def trigger(self):
        self._ack.clear()
        self.event.set()
        if self.hard_exit:
            threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        if not self._ack.wait(self.grace):
            os._exit(2)

    def check(self):
        if self.event.is_set():
            raise Aborted("kill hotkey pressed")

    def acknowledge(self):
        """Call once the current task has unwound; re-arms the switch."""
        self.event.clear()
        self._ack.set()
