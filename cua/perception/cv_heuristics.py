"""Lightweight CV heuristics over OCR word positions (and, for icon-only targets, raw pixels) -- no OpenCV,
no deep learning. Handles the class of targets plain text matching can't: "first channel" (ordinal selection
over spatially-clustered result cards), "button in the top-right" (spatial filtering), "search icon" (a small
best-effort local-contrast detector for icon-sized regions when nothing has a name to match).

Deliberately modest, per the milestone's instruction not to build a general CV system: these are heuristics
that solve the specific examples in the spec, not a general object detector.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ORDINAL_WORDS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
                  "fifth": 4, "5th": 4, "last": -1}
_SPATIAL_WORDS = {"top", "bottom", "left", "right", "center", "middle"}


@dataclass
class Region:
    rect: tuple[int, int, int, int]     # left, top, right, bottom, screen pixels
    label: str = ""
    words: list[str] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        return (self.rect[0] + self.rect[2]) // 2, (self.rect[1] + self.rect[3]) // 2

    @property
    def area(self) -> int:
        return max(0, self.rect[2] - self.rect[0]) * max(0, self.rect[3] - self.rect[1])


def parse_ordinal(text: str) -> int | None:
    """'first channel' -> 0, 'third result' -> 2, 'last video' -> -1, else None."""
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[w]
    return None


def parse_spatial(text: str) -> set[str]:
    """'button in the top-right' -> {'top', 'right'}."""
    return set(re.findall(r"[a-z]+", text.lower())) & _SPATIAL_WORDS


def cluster_cards(words: list[tuple[str, tuple[int, int, int, int]]], row_gap: int = 40,
                  col_gap: int = 80, bounds: tuple[int, int, int, int] | None = None,
                  top_margin_frac: float = 0.14) -> list[Region]:
    """Group OCR words into rough 'cards' (e.g. video/channel result tiles) by proximity, ordered top-to-
    bottom then left-to-right -- enough for 'first result', 'second card', without a real layout model.
    A genuine grid/list layout will usually cluster correctly; irregular pages may not, which is why this
    only ever RANKS candidates -- confidence stays capped so a bad cluster can't drive a confident wrong click.

    `bounds`, if given, excludes the top `top_margin_frac` of the captured area before clustering: measured
    on a real page (YouTube search results), the single biggest source of a wrong "first result" was picking
    up the BROWSER's own title bar / tab strip / address bar text, not page content -- true of virtually any
    windowed app, not just Chrome, so this is a general exclusion, not a Chrome-specific hack. Even with this,
    live testing on that same page still misfired once (see grounder.py's confidence for this method, capped
    deliberately below the auto-execute threshold as a result) -- raise this if false positives persist.
    """
    if not words:
        return []
    if bounds:
        top, bottom = bounds[1], bounds[3]
        cutoff = top + int((bottom - top) * top_margin_frac)
        words = [w for w in words if w[1][1] >= cutoff]
        if not words:
            return []
    boxes = sorted((w[1] for w in words), key=lambda r: (r[1], r[0]))
    clusters: list[list[tuple[int, int, int, int]]] = []
    for box in boxes:
        placed = False
        for cl in clusters:
            cx0, cy0, cx1, cy1 = cl[-1]
            if abs(box[1] - cy0) < row_gap and 0 <= box[0] - cx1 < col_gap:
                cl.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])
    regions = []
    for cl in clusters:
        rect = (min(b[0] for b in cl), min(b[1] for b in cl), max(b[2] for b in cl), max(b[3] for b in cl))
        regions.append(Region(rect))
    regions.sort(key=lambda r: (r.rect[1], r.rect[0]))
    return regions


def select_ordinal(regions: list[Region], index: int) -> Region | None:
    if not regions:
        return None
    try:
        return regions[index]
    except IndexError:
        return None


def filter_spatial(regions: list[Region], keywords: set[str], bounds: tuple[int, int, int, int]) -> list[Region]:
    """Keep only regions whose center falls in the screen area implied by top/bottom/left/right keywords."""
    if not keywords:
        return regions
    left, top, right, bottom = bounds
    midx, midy = (left + right) // 2, (top + bottom) // 2
    out = []
    for reg in regions:
        cx, cy = reg.center
        if "top" in keywords and cy > midy:
            continue
        if "bottom" in keywords and cy < midy:
            continue
        if "left" in keywords and cx > midx:
            continue
        if "right" in keywords and cx < midx:
            continue
        out.append(reg)
    return out


def find_icon_candidates(w: int, h: int, bgra: bytes, bounds: tuple[int, int, int, int],
                         cell: int = 24, min_edge_density: float = 18.0, max_candidates: int = 8) -> list[Region]:
    """Best-effort small-icon detector: grid-scan for cells with high local contrast (edges) and roughly
    square/small size -- icons tend to be small, busy (many edges) regions against comparatively flat
    surrounding chrome. Crude by design (no OpenCV): a ranking signal for the cascade, not a classifier, so
    it's only ever used to narrow candidates, never to click without OCR/CV agreement or explicit confirmation
    at low confidence.
    """
    import struct
    left, top, right, bottom = bounds
    scored: list[tuple[float, Region]] = []
    for gy in range(0, h - cell, cell):
        for gx in range(0, w - cell, cell):
            # sample a sparse grid of pixels in the cell, sum of absolute luma deltas between neighbors
            prev = None
            edge_sum = 0.0
            n = 0
            for dy in range(0, cell, 4):
                for dx in range(0, cell, 4):
                    idx = ((gy + dy) * w + (gx + dx)) * 4
                    if idx + 3 >= len(bgra):
                        continue
                    b, g, r = bgra[idx], bgra[idx + 1], bgra[idx + 2]
                    luma = 0.114 * b + 0.587 * g + 0.299 * r
                    if prev is not None:
                        edge_sum += abs(luma - prev)
                        n += 1
                    prev = luma
            if n == 0:
                continue
            density = edge_sum / n
            if density >= min_edge_density:
                rect = (left + gx, top + gy, left + gx + cell, top + gy + cell)
                scored.append((density, Region(rect, label="icon-candidate")))
    scored.sort(key=lambda t: -t[0])
    return [r for _, r in scored[:max_candidates]]
