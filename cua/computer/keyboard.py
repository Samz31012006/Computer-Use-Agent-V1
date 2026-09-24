"""SendInput keyboard: unicode typing and hotkeys. No window logic here."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Callable

from cua.types import DriverError

user32 = ctypes.windll.user32

_KEYEVENTF_KEYUP, _KEYEVENTF_UNICODE = 0x2, 0x4
_VK = {"ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B, "enter": 0x0D, "return": 0x0D,
       "tab": 0x09, "esc": 0x1B, "escape": 0x1B, "space": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
       "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27, "home": 0x24, "end": 0x23,
       "pgup": 0x21, "pgdn": 0x22, "pageup": 0x21, "pagedown": 0x22, "insert": 0x2D,
       "plus": 0xBB, "equals": 0xBB, "minus": 0xBD,
       # media / volume keys: these are what "turn the volume up", "mute", "pause the music" actually send.
       # They work globally regardless of which window has focus, which is exactly the behaviour wanted here.
       "volume_up": 0xAF, "volume_down": 0xAE, "volume_mute": 0xAD,
       "media_play_pause": 0xB3, "media_stop": 0xB2, "media_next": 0xB0, "media_prev": 0xB1}
_VK.update({f"f{i}": 0x6F + i for i in range(1, 13)})
_MODS = {0x11, 0x12, 0x10, 0x5B}


def vk_for(name: str) -> int | None:
    n = name.strip().lower()
    if n in _VK:
        return _VK[n]
    if len(n) == 1 and n.isalnum():
        return ord(n.upper())
    return None


def normalize_keys(keys: str) -> str | None:
    """'Control + S' -> 'ctrl+s'; None if any part is not a known key."""
    parts = [p.strip().lower().replace("control", "ctrl").replace("escape", "esc") for p in keys.replace(" ", "+").split("+") if p.strip()]
    return "+".join(parts) if parts and all(vk_for(p) is not None for p in parts) else None


class _KI(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _MI(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _IU(ctypes.Union):
    _fields_ = [("ki", _KI), ("mi", _MI)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _IU)]


def _send(events: list[_INPUT]):
    arr = (_INPUT * len(events))(*events)
    if user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT)) != len(events):
        raise DriverError("SendInput blocked (UIPI / secure desktop?)")


def _key(vk: int, up: bool) -> _INPUT:
    return _INPUT(1, _IU(ki=_KI(vk, 0, _KEYEVENTF_KEYUP if up else 0, 0, 0)))


def _uni(ch: str, up: bool) -> _INPUT:
    return _INPUT(1, _IU(ki=_KI(0, ord(ch), _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if up else 0), 0, 0)))


class Keyboard:
    def __init__(self, kill_check: Callable[[], None] = lambda: None):
        self._kill_check = kill_check

    def type_text(self, text: str) -> None:
        chunk: list[_INPUT] = []
        for ch in text:
            self._kill_check()
            if ch in "\r\n":
                chunk += [_key(0x0D, False), _key(0x0D, True)]
            else:
                chunk += [_uni(ch, False), _uni(ch, True)]
            if len(chunk) >= 40:
                _send(chunk)
                chunk = []
        if chunk:
            _send(chunk)

    def hotkey(self, keys: str) -> None:
        vks = []
        for part in keys.lower().split("+"):
            vk = vk_for(part)
            if vk is None:
                raise DriverError(f"unknown key '{part}'")
            vks.append(vk)
        mods = [v for v in vks if v in _MODS]
        main = [v for v in vks if v not in _MODS]
        try:
            _send([_key(v, False) for v in mods] + [e for v in main for e in (_key(v, False), _key(v, True))])
        finally:
            _send([_key(v, True) for v in reversed(mods)])   # never leave a modifier stuck down

    press = hotkey   # a single key is a degenerate hotkey
