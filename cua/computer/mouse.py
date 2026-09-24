"""Mouse via win32 mouse_event. Coordinates are physical screen pixels."""
from __future__ import annotations

import time

import win32api
import win32con


class Mouse:
    def move(self, x: int, y: int) -> None:
        win32api.SetCursorPos((int(x), int(y)))

    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> None:
        self.move(x, y)
        down, up = ((win32con.MOUSEEVENTF_RIGHTDOWN, win32con.MOUSEEVENTF_RIGHTUP) if button == "right"
                    else (win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_LEFTUP))
        for i in range(2 if double else 1):
            win32api.mouse_event(down, 0, 0, 0, 0)
            win32api.mouse_event(up, 0, 0, 0, 0)
            if double and i == 0:
                time.sleep(0.05)

    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> None:
        """Positive = up, negative = down; one unit is one wheel notch."""
        if x is not None and y is not None:
            self.move(x, y)
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, int(amount) * 120, 0)
