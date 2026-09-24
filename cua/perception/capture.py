"""Screen capture via plain GDI (no Pillow / no extra dependency). Used only by the OCR/vision fallbacks:
the deterministic paths never take screenshots.

DPI: SetProcessDpiAwareness below makes GetWindowRect/GetSystemMetrics return PHYSICAL pixels -- the same
space BitBlt/PrintWindow copy from and SetCursorPos clicks into -- so no separate DPI-scaling math is needed
anywhere else in this module or its callers; coordinates are consistent end to end.

Multi-monitor: window_rect(hwnd) already returns the right rectangle regardless of which monitor a window is
on (GetWindowRect is monitor-agnostic). monitor_rects() and virtual_screen_rect() are for callers that need
the FULL desktop (every monitor) or a specific one, e.g. "no window is focused, ground somewhere on screen"."""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import win32gui

user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
_SRCCOPY = 0x00CC0020
_SM_XVIRTUALSCREEN, _SM_YVIRTUALSCREEN = 76, 77
_SM_CXVIRTUALSCREEN, _SM_CYVIRTUALSCREEN = 78, 79

try:                                   # coordinates from GetWindowRect must match the pixels BitBlt copies
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


_H = ctypes.c_void_p     # handles are pointer-sized; without argtypes ctypes truncates them to 32 bits (intermittent OverflowError)
user32.GetDC.restype, user32.GetDC.argtypes = _H, [_H]
user32.ReleaseDC.argtypes = [_H, _H]
user32.PrintWindow.restype, user32.PrintWindow.argtypes = wintypes.BOOL, [_H, _H, wintypes.UINT]
gdi32.CreateCompatibleDC.restype, gdi32.CreateCompatibleDC.argtypes = _H, [_H]
gdi32.CreateCompatibleBitmap.restype, gdi32.CreateCompatibleBitmap.argtypes = _H, [_H, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype, gdi32.SelectObject.argtypes = _H, [_H, _H]
gdi32.BitBlt.restype = wintypes.BOOL
gdi32.BitBlt.argtypes = [_H, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _H, ctypes.c_int, ctypes.c_int,
                         wintypes.DWORD]
gdi32.GetDIBits.restype = ctypes.c_int
gdi32.GetDIBits.argtypes = [_H, _H, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
gdi32.DeleteObject.argtypes = [_H]
gdi32.DeleteDC.argtypes = [_H]


class _BMIH(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def window_rect(hwnd: int | None) -> tuple[int, int, int, int]:
    if hwnd:
        return win32gui.GetWindowRect(hwnd)
    return virtual_screen_rect()


def virtual_screen_rect() -> tuple[int, int, int, int]:
    """The full desktop across every monitor (left may be negative if a monitor is placed left of primary)."""
    x = user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
    y = user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
    w = user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
    h = user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)
    return x, y, x + w, y + h


def monitor_rects() -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) of every connected monitor, in the same physical-pixel space as capture()."""
    rects: list[tuple[int, int, int, int]] = []
    MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                         ctypes.POINTER(wintypes.RECT), ctypes.c_double)

    def _cb(hmon, hdc, rect, data):
        r = rect.contents
        rects.append((r.left, r.top, r.right, r.bottom))
        return True

    user32.EnumDisplayMonitors(None, None, MonitorEnumProc(_cb), 0)
    return rects


def capture(rect: tuple[int, int, int, int], hwnd: int | None = None) -> tuple[int, int, bytes]:
    """BGRA pixels of a screen rectangle, top-down. With hwnd, the window renders itself (PrintWindow), so the
    result is correct even if another window overlaps it or it has just been brought forward."""
    left, top, right, bottom = rect
    w, h = max(1, right - left), max(1, bottom - top)
    hdc = user32.GetDC(0)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mdc, bmp)
    try:
        printed = bool(hwnd) and user32.PrintWindow(hwnd, mdc, 2)      # 2 = PW_RENDERFULLCONTENT
        if not printed:
            gdi32.BitBlt(mdc, 0, 0, w, h, hdc, left, top, _SRCCOPY)
        hdr = _BMIH(ctypes.sizeof(_BMIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(hdr), 0)
        return w, h, buf.raw
    finally:
        gdi32.SelectObject(mdc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, hdc)


def capture_timed(rect: tuple[int, int, int, int], hwnd: int | None = None) -> tuple[int, int, bytes, float]:
    """capture() plus wall-clock latency in ms, for Logger spans / perf instrumentation (Part 14)."""
    t0 = time.perf_counter()
    w, h, px = capture(rect, hwnd)
    return w, h, px, (time.perf_counter() - t0) * 1000
