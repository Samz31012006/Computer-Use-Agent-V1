"""OCR fallback perception (used only when UI Automation cannot see a control: canvas apps, games, images).
Interface + null backend now; the Windows.Media.Ocr backend (built into Windows 11) is added when approved."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TextHit:
    text: str
    rect: tuple[int, int, int, int]      # left, top, right, bottom in screen pixels
    confidence: float = 1.0

    @property
    def center(self) -> tuple[int, int]:
        return (self.rect[0] + self.rect[2]) // 2, (self.rect[1] + self.rect[3]) // 2


class OcrEngine(Protocol):
    available: bool

    def find_text(self, text: str, hwnd: int | None = None) -> list[TextHit]:
        """Capture the window (or screen) once and return matches for `text`, best first."""


class NullOcr:
    available = False

    def find_text(self, text: str, hwnd: int | None = None) -> list[TextHit]:
        return []


class WindowsOcr:
    """Windows.Media.Ocr (built into Windows 11, offline, no model download). One capture per call, cropped to the
    target window. The recognizer is created lazily and kept: it is small and costs nothing while idle."""

    def __init__(self):
        self._engine = None
        self._checked = False
        self._ok = False

    @property
    def available(self) -> bool:
        if not self._checked:
            self._checked = True
            try:
                from winrt.windows.media.ocr import OcrEngine as _E
                self._engine = _E.try_create_from_user_profile_languages()
                self._ok = self._engine is not None
            except Exception:
                self._ok = False
        return self._ok

    def recognize(self, hwnd: int | None = None) -> list[tuple[str, tuple[int, int, int, int]]]:
        """All recognised words with screen-pixel rectangles."""
        import asyncio

        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter

        from cua.perception.capture import capture, window_rect
        if not self.available:
            return []
        rect = window_rect(hwnd)
        w, h, pixels = capture(rect, hwnd)
        px = bytearray(pixels)
        px[3::4] = b"\xff" * (w * h)          # GDI leaves alpha at 0; an opaque image is what the recogniser expects
        dw = DataWriter()
        dw.write_bytes(px)
        bmp = SoftwareBitmap.create_copy_from_buffer(dw.detach_buffer(), BitmapPixelFormat.BGRA8, w, h)
        async def _go():
            return await self._engine.recognize_async(bmp)
        result = asyncio.run(_go())
        words = []
        for line in result.lines:
            for wd in line.words:
                r = wd.bounding_rect
                words.append((wd.text, (rect[0] + int(r.x), rect[1] + int(r.y),
                                        rect[0] + int(r.x + r.width), rect[1] + int(r.y + r.height))))
        return words

    def find_text(self, text: str, hwnd: int | None = None) -> list[TextHit]:
        return match_words(self.recognize(hwnd), text)


def match_words(words: list[tuple[str, tuple[int, int, int, int]]], query: str) -> list[TextHit]:
    """Tightest runs of consecutive words containing the query (case-insensitive), best (shortest) first.
    A run that merely contains a tighter match is dropped."""
    q = " ".join(query.lower().split())
    spans: list[tuple[int, int]] = []
    for i in range(len(words)):
        joined = ""
        for j in range(i, min(len(words), i + len(q.split()) + 2)):
            joined = (joined + " " + words[j][0]).strip().lower()
            if q in joined:
                spans.append((i, j))
                break
    spans = [(i, j) for i, j in spans if not any((i2 >= i and j2 <= j) and (i2, j2) != (i, j) for i2, j2 in spans)]
    hits = []
    for i, j in sorted(spans, key=lambda t: t[1] - t[0]):
        rs = [w[1] for w in words[i:j + 1]]
        rect = (min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs))
        hits.append(TextHit(" ".join(w[0] for w in words[i:j + 1]), rect))
    return hits
