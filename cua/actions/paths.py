"""User-path resolution. Every file tool goes through here so nothing can touch paths outside the user's profile."""
from __future__ import annotations

import re
from pathlib import Path

from cua.actions.base import ToolError
from cua.catalog import FOLDER_KEYS, folder_key, known_folder


def resolve_user_path(text: str, folders: dict | None = None, default_folder: str = "documents") -> Path:
    """'downloads/a.pdf' | '~/x' | 'C:\\Users\\me\\x' | 'x.txt' (-> Documents). Must stay inside the user's profile."""
    t = text.strip().strip('"').strip("'")
    if not t:
        raise ToolError("empty path")
    parts = [p for p in re.split(r"[\\/]+", t) if p]
    key = folder_key(parts[0]) if parts and not re.fullmatch(r"[A-Za-z]:", parts[0]) else None
    if key:
        p = known_folder(key, folders).joinpath(*parts[1:])
    elif t.startswith("~"):
        p = Path.home() / t[1:].lstrip("\\/")
    elif Path(t).is_absolute():
        p = Path(t)
    else:
        p = known_folder(default_folder, folders) / t
    p = p.resolve()
    if not is_allowed(p, folders):
        raise ToolError(f"path outside the user profile is not allowed: {p}")
    return p


def allowed_roots(folders: dict | None = None) -> list[Path]:
    roots = [Path.home().resolve()]
    roots += [Path(v).resolve() for v in (folders or {}).values()]
    roots += [known_folder(k, folders).resolve() for k in FOLDER_KEYS if k != "home"]
    return roots


def is_allowed(p: Path, folders: dict | None = None) -> bool:
    return any(p == r or r in p.parents for r in allowed_roots(folders))
