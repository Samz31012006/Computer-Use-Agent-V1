"""Perception seams. Order of preference: structured OS state -> UI Automation -> OCR -> (optional) vision model.
Nothing here takes a screenshot unless a fallback layer (OCR / vision) is explicitly invoked."""
from __future__ import annotations

from typing import Any, Protocol

from cua.types import ElementRef, WinInfo


class PerceptionEngine(Protocol):
    def windows(self) -> list[WinInfo]: ...
    def foreground(self) -> WinInfo | None: ...
    def match(self, app: Any) -> list[WinInfo]: ...
    def find_element(self, hwnd: int, name: str, timeout: float = 0.0) -> ElementRef | None: ...
    def read_text(self, hwnd: int) -> str: ...


def describe_state(perception, max_windows: int = 6, width: int = 40) -> str:
    """Compact, model-friendly summary of the desktop (~40 tokens). Cheap: one EnumWindows pass, no UIA walk."""
    try:
        fg = perception.foreground()
        titles = [w.title[:width] for w in perception.windows() if fg is None or w.hwnd != fg.hwnd][:max_windows]
    except Exception:
        return ""
    parts = [f"foreground: {fg.title[:width]!r}" if fg else "foreground: none"]
    if titles:
        parts.append("other windows: " + "; ".join(titles))
    return " | ".join(parts)
