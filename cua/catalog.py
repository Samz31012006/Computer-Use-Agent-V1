"""Static knowledge the deterministic path relies on: apps, folders, settings pages, search engines."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus


@dataclass(frozen=True)
class AppSpec:
    key: str
    launch: tuple                 # ("shell", target) | ("cmd", exe) | ("startapps", name)
    procs: tuple = ()             # process names that own the app's windows
    title_re: str | None = None   # or match by window title (UWP apps are hosted by ApplicationFrameHost)
    cls: tuple = ()               # optional window class filter
    aliases: tuple = ()
    generic: bool = False         # not in the catalog; resolved through Start-menu lookup


APPS: list[AppSpec] = [
    AppSpec("chrome", ("shell", "chrome"), ("chrome.exe",), r"Google Chrome$", (), ("google chrome", "chrome browser", "browser")),
    AppSpec("edge", ("shell", "msedge"), ("msedge.exe",), r"Edge$", (), ("microsoft edge",)),
    AppSpec("firefox", ("shell", "firefox"), ("firefox.exe",), r"Firefox$"),
    AppSpec("notepad", ("shell", "notepad.exe"), ("notepad.exe",), r"Notepad$"),
    AppSpec("calculator", ("shell", "calc.exe"), (), r"^Calculator$", (), ("calc",)),
    AppSpec("vscode", ("cmd", "code"), ("code.exe",), r"Visual Studio Code$", (), ("vs code", "visual studio code", "code")),
    AppSpec("explorer", ("shell", "explorer.exe"), ("explorer.exe",), None, ("CabinetWClass",), ("file explorer", "files", "windows explorer")),
    AppSpec("settings", ("shell", "ms-settings:"), (), r"^Settings$"),
    AppSpec("terminal", ("shell", "wt.exe"), ("WindowsTerminal.exe",), None, (), ("windows terminal",)),
    AppSpec("paint", ("shell", "mspaint.exe"), (), r"Paint$"),
]

_BY_NAME = {}
for _a in APPS:
    _BY_NAME[_a.key] = _a
    for _al in _a.aliases:
        _BY_NAME[_al] = _a


def register(spec: AppSpec) -> None:
    """Add an app at runtime (used by the benchmark's throwaway typing target)."""
    APPS.append(spec)
    _BY_NAME[spec.key] = spec
    for al in spec.aliases:
        _BY_NAME[al] = spec


def resolve_app(name: str) -> AppSpec | None:
    n = re.sub(r"^(the|my|a)\s+|\s+(app|application|program)$", "", name.strip().lower())
    return _BY_NAME.get(n)


def generic_app(name: str) -> AppSpec:
    n = name.strip()
    return AppSpec(n.lower(), ("startapps", n), (), re.escape(n), (), (), generic=True)


_START_APPS: list[str] | None = None


def installed_apps(cache_dir: str | Path = ".cache") -> list[str]:
    """Names from the cached Start-menu index, or [] if it hasn't been built yet.

    Read-only and cache-only on purpose: this is consulted while PLANNING, which must stay instant and must
    never shell out to PowerShell (the driver builds/refreshes the cache when it actually launches something).
    No cache simply means bare app names aren't recognised yet -- the assistant asks instead of guessing."""
    global _START_APPS
    if _START_APPS is None:
        try:
            import json
            data = json.loads((Path(cache_dir) / "startapps.json").read_text(encoding="utf-8"))
            _START_APPS = [a["Name"] for a in data if isinstance(a, dict) and a.get("Name")]
        except Exception:
            _START_APPS = []
    return _START_APPS


def is_installed_app(name: str) -> bool:
    """True only for a confident match against a genuinely installed app: the whole phrase must equal the
    app's name or appear in it as a run of whole words ("task manager" -> "Task Manager"). Deliberately
    strict -- this is what lets a bare noun launch something, so a misheard phrase must not match anything."""
    n = " ".join(name.strip().lower().split())
    if len(n) < 3:
        return False
    for app in installed_apps():
        low = app.lower()
        if low == n or re.search(rf"(?:^|\s){re.escape(n)}(?:\s|$)", low):
            return True
    return False


FOLDER_KEYS = ("downloads", "documents", "desktop", "pictures", "music", "videos", "home")
_FOLDER_ALIASES = {"download": "downloads", "document": "documents", "docs": "documents",
                   "picture": "pictures", "photos": "pictures", "video": "videos",
                   "user": "home", "home": "home", "my computer": "home"}


def folder_key(word: str) -> str | None:
    w = word.strip().lower()
    w = _FOLDER_ALIASES.get(w, w)
    return w if w in FOLDER_KEYS else None


def known_folder(key: str, overrides: dict[str, Path] | None = None) -> Path:
    if overrides and key in overrides:
        return Path(overrides[key])
    if key == "home":
        return Path.home()
    try:  # honours OneDrive-redirected folders
        from win32com.shell import shell, shellcon
        fid = {"downloads": shellcon.FOLDERID_Downloads, "documents": shellcon.FOLDERID_Documents,
               "desktop": shellcon.FOLDERID_Desktop, "pictures": shellcon.FOLDERID_Pictures,
               "music": shellcon.FOLDERID_Music, "videos": shellcon.FOLDERID_Videos}[key]
        return Path(shell.SHGetKnownFolderPath(fid, 0, None))
    except Exception:
        return Path.home() / key.capitalize()


SETTINGS_PAGES = {
    "bluetooth": "ms-settings:bluetooth", "wifi": "ms-settings:network-wifi", "wi-fi": "ms-settings:network-wifi",
    "network": "ms-settings:network", "display": "ms-settings:display", "sound": "ms-settings:sound",
    "battery": "ms-settings:batterysaver", "power": "ms-settings:powersleep", "sleep": "ms-settings:powersleep",
    "apps": "ms-settings:appsfeatures", "windows update": "ms-settings:windowsupdate", "update": "ms-settings:windowsupdate",
    "personalization": "ms-settings:personalization", "background": "ms-settings:personalization-background",
    "storage": "ms-settings:storagesense", "privacy": "ms-settings:privacy", "accounts": "ms-settings:yourinfo",
    "time": "ms-settings:dateandtime", "date": "ms-settings:dateandtime", "notifications": "ms-settings:notifications",
    "default apps": "ms-settings:defaultapps", "about": "ms-settings:about", "night light": "ms-settings:nightlight",
    "colors": "ms-settings:colors", "dark mode": "ms-settings:colors", "theme": "ms-settings:themes",
    "mouse": "ms-settings:mousetouchpad", "keyboard": "ms-settings:easeofaccess-keyboard",
}

SEARCH_ENGINES = {
    "google": ("https://www.google.com/search?q=", ("google search",)),
    "youtube": ("https://www.youtube.com/results?search_query=", ("youtube",)),
    "bing": ("https://www.bing.com/search?q=", ("bing",)),
    "duckduckgo": ("https://duckduckgo.com/?q=", ("duckduckgo",)),
    "wikipedia": ("https://en.wikipedia.org/w/index.php?search=", ("wikipedia",)),
}


def search_url(engine: str, query: str) -> str:
    return SEARCH_ENGINES[engine][0] + quote_plus(query)


# Short names people actually say. Anything not listed here can still be reached with "go to <word>" (-> <word>.com)
# or a full domain; this table only exists for abbreviations that cannot be guessed.
SITES = {
    "yt": "youtube.com", "youtube": "youtube.com", "gmail": "mail.google.com", "github": "github.com",
    "reddit": "reddit.com", "twitter": "x.com", "x": "x.com", "wikipedia": "wikipedia.org", "wiki": "wikipedia.org",
    "netflix": "netflix.com", "amazon": "amazon.com", "maps": "maps.google.com", "google": "google.com",
    "linkedin": "linkedin.com", "facebook": "facebook.com", "insta": "instagram.com",
    "instagram": "instagram.com", "drive": "drive.google.com",
}
SITE_ENGINE = {"youtube.com": "youtube", "wikipedia.org": "wikipedia", "google.com": "google"}


def resolve_site(name: str) -> str | None:
    n = re.sub(r"^(the|my)\s+|\s+(website|site|page|app)$", "", name.strip().lower())
    return SITES.get(n)
