"""WindowsDriver: the ComputerDriver implementation. Launching/opening lives here; input and window
control are delegated to keyboard.py / mouse.py / windows.py. No observation logic."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import win32api
import win32con

from cua.catalog import AppSpec
from cua.computer.keyboard import Keyboard
from cua.computer.mouse import Mouse
from cua.computer.windows import Windows
from cua.types import DriverError


class ComputerDriver(Protocol):
    """Side effects on the OS. Anything implementing these can replace WindowsDriver (tests use fakes)."""
    def launch(self, app: Any, params: str | None = None) -> None: ...
    def open_uri(self, uri: str, browser: Any | None = None) -> None: ...
    def open_path(self, path: str) -> None: ...
    def reveal(self, path: str) -> None: ...
    def focus(self, hwnd: int) -> bool: ...
    def close(self, hwnd: int) -> None: ...
    def minimize(self, hwnd: int) -> None: ...
    def maximize(self, hwnd: int) -> None: ...
    def type_text(self, text: str) -> None: ...
    def hotkey(self, keys: str) -> None: ...
    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> None: ...
    def move(self, x: int, y: int) -> None: ...
    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> None: ...
    def sleep(self, seconds: float) -> None: ...


class WindowsDriver:
    def __init__(self, kill_check: Callable[[], None] = lambda: None, cache_dir: str | Path = ".cache"):
        self._kill_check = kill_check
        self._cache = Path(cache_dir)
        self.keyboard, self.mouse, self.windows = Keyboard(kill_check), Mouse(), Windows()

    # ---- launching / opening ---------------------------------------------
    def launch(self, app: AppSpec, params: str | None = None) -> None:
        kind, target = app.launch
        try:
            if kind == "shell":
                win32api.ShellExecute(0, "open", target, params, None, win32con.SW_SHOWNORMAL)
            elif kind == "cmd":   # .cmd shims such as VS Code's `code`
                subprocess.Popen(["cmd", "/c", target, *([params] if params else [])],
                                 creationflags=subprocess.CREATE_NO_WINDOW)
            elif kind == "exec":  # target is an argv list
                subprocess.Popen([*target, *([params] if params else [])], creationflags=subprocess.CREATE_NO_WINDOW)
            elif kind == "startapps":
                app_id = self._startapps_lookup(target)
                if not app_id:
                    raise DriverError(f"no installed app matching '{target}'")
                subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
            else:
                raise DriverError(f"unknown launch kind {kind}")
        except DriverError:
            raise
        except Exception as e:
            raise DriverError(f"launch {app.key}: {e}") from e

    def open_uri(self, uri: str, browser: AppSpec | None = None) -> None:
        try:
            if browser is not None and browser.launch[0] == "shell":
                win32api.ShellExecute(0, "open", browser.launch[1], uri, None, win32con.SW_SHOWNORMAL)
            else:
                win32api.ShellExecute(0, "open", uri, None, None, win32con.SW_SHOWNORMAL)
        except Exception as e:
            raise DriverError(f"open {uri}: {e}") from e

    def open_path(self, path: str, app: AppSpec | None = None) -> None:
        try:
            if app is not None and app.launch[0] == "cmd":
                subprocess.Popen(["cmd", "/c", app.launch[1], str(path)], creationflags=subprocess.CREATE_NO_WINDOW)
            elif app is not None and app.launch[0] == "exec":
                subprocess.Popen([*app.launch[1], str(path)], creationflags=subprocess.CREATE_NO_WINDOW)
            elif app is not None and app.launch[0] == "shell":
                win32api.ShellExecute(0, "open", app.launch[1], f'"{path}"', None, win32con.SW_SHOWNORMAL)
            else:
                win32api.ShellExecute(0, "open", str(path), None, None, win32con.SW_SHOWNORMAL)
        except Exception as e:
            raise DriverError(f"open {path}: {e}") from e

    def reveal(self, path: str) -> None:
        subprocess.Popen(f'explorer.exe /select,"{path}"')

    def _startapps_lookup(self, name: str) -> str | None:
        """Start-menu app lookup, cached on disk for a day (Get-StartApps costs ~1 s)."""
        cache = self._cache / "startapps.json"
        apps = None
        if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
            apps = json.loads(cache.read_text(encoding="utf-8"))
        if apps is None:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-StartApps | ConvertTo-Json"],
                                 capture_output=True, text=True, timeout=30).stdout
            apps = json.loads(out or "[]")
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps(apps), encoding="utf-8")
        n = name.lower()
        exact = [a for a in apps if a["Name"].lower() == n]
        part = [a for a in apps if n in a["Name"].lower()]
        hit = (exact or sorted(part, key=lambda a: len(a["Name"])) or [None])[0]
        return hit["AppID"] if hit else None

    # ---- delegated --------------------------------------------------------
    def focus(self, hwnd): return self.windows.focus(hwnd)
    def close(self, hwnd): self.windows.close(hwnd)
    def minimize(self, hwnd): self.windows.minimize(hwnd)
    def maximize(self, hwnd): self.windows.maximize(hwnd)
    def type_text(self, text): self.keyboard.type_text(text)
    def hotkey(self, keys): self.keyboard.hotkey(keys)
    def click(self, x, y, button="left", double=False): self.mouse.click(x, y, button, double)
    def move(self, x, y): self.mouse.move(x, y)
    def scroll(self, amount, x=None, y=None): self.mouse.scroll(amount, x, y)

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._kill_check()
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))
