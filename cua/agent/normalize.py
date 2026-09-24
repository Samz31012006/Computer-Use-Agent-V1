"""Lightweight fuzzy normalization against the existing app/site/settings catalog.

No hardcoded typo dictionary: candidates are the app keys/aliases, site names, and settings-page names already
declared in cua/catalog.py. `difflib.get_close_matches` (stdlib) scores a misspelled word against that pool.
A single clearly-best match is safe to substitute automatically; several close, distinct matches are genuinely
ambiguous and the caller should ask rather than guess.
"""
from __future__ import annotations

from difflib import get_close_matches

from cua.catalog import APPS, SETTINGS_PAGES, SITES

_CUTOFF = 0.72


def _pool() -> dict[str, str]:
    """word -> canonical form, pooled from every catalog table that names an app/site/page."""
    pool: dict[str, str] = {}
    for spec in APPS:
        if spec.generic:
            continue
        pool[spec.key] = spec.key
        for alias in spec.aliases:
            pool[alias] = spec.key
    for name in SITES:
        pool[name] = name
    for name in SETTINGS_PAGES:
        pool[name] = name
    return pool


def fuzzy_resolve(word: str) -> tuple[str | None, list[str]]:
    """Best-effort correction of a single app/site/settings word against the catalog.

    Returns (corrected, alternatives):
      - exact/alias hit already: (word, [])                        -- nothing to correct
      - one clearly-best close match: (best, [])                    -- safe to substitute silently
      - several close, distinct matches: (None, [top few])          -- ambiguous, caller should ask
      - nothing close enough: (None, [])                            -- leave the input alone
    """
    w = word.strip().lower()
    pool = _pool()
    if not w or w in pool:
        return (word, []) if w else (None, [])
    hits = get_close_matches(w, pool.keys(), n=5, cutoff=_CUTOFF)
    if not hits:
        return None, []
    canon, seen = [], set()
    for h in hits:
        c = pool[h]
        if c not in seen:
            seen.add(c)
            canon.append(c)
    if len(canon) == 1:
        return canon[0], []
    return None, canon[:3]
