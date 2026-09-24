"""Deterministic Planner: regex + catalog -> TaskPlan. Returns None when ANY clause is not understood,
so the caller can hand the whole command to a model planner. It never guesses.

How much of real usage this covers is a hypothesis; bench/corpus.py and the CLI's miss log measure it.
"""
from __future__ import annotations

import re
from pathlib import Path

from cua.catalog import (FOLDER_KEYS, SEARCH_ENGINES, SETTINGS_PAGES, SITE_ENGINE, folder_key, generic_app,
                         is_installed_app, resolve_app, resolve_site, search_url)
from cua.agent.extras import NOT_MINE, handle as handle_extra, normalize
from cua.agent.normalize import fuzzy_resolve
from cua.computer.keyboard import vk_for
from cua.actions.base import ToolError
from cua.actions.registry import ToolRegistry, default_registry
from cua.types import NeedsClarification, Step, TaskPlan

_VERBS = (r"open|launch|start|run|search|google|look up|close|quit|exit|type|enter|write|press|hit|find|locate|"
          r"go to|navigate|visit|click|tap|create|make|new|switch|focus|minimize|maximize|restore|scroll|copy|move|"
          r"list|show|put|add|turn|enable|save|refresh|reload|zoom|undo|redo|paste|cut|print|select|go back|go forward")
_SPLIT = re.compile(rf"\s*(?:,|;|&|\band then\b|\bthen\b|\band\b|\bafter that\b|\balso\b|\bnext\b|\bfinally\b|"
                    rf"\bafterwards\b)\s+(?=(?:{_VERBS})\b)", re.I)
# "open chrome go to yt search cats": a new verb right after a recognised target starts a new clause (no "and" needed)
_IMPLICIT = re.compile(r"^(?P<v>open|launch|start|run|go to|navigate to|visit)\s+(?P<t>.+?)\s+(?P<rest>(?:go to|navigate to|"
                       r"visit|search|google|look up|type|press|click|find)\b.*)$", re.I)
_GENERATION = re.compile(r"^(?:me\s+)?(?:an?\s+|the\s+|some\s+)?(?:poem|essay|story|email|letter|report|summary|song|joke|"
                         r"code|program|script|function|paragraph|article|blog|speech|caption|review)\b|\babout\b|\bthat\b",
                         re.I)
_LEAD = re.compile(r"^(?:hey |ok |okay )?(?:please |can you |could you |would you |i want you to |i'?d like you to |"
                   r"go ahead and |i need you to )+", re.I)
_TRAIL = re.compile(r"[\s.!?]*(?:,? ?(?:please|for me))?[\s.!?]*$", re.I)
_URL = re.compile(r"^(?:https?://\S+|(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|dev|gov|edu|co|ai|app)(?:/\S*)?)$", re.I)
_WINPATH = re.compile(r"^(?:[A-Za-z]:\\|\\\\|~[\\/])[^\"<>|?*]*$")
_EXTS = r"pdf|docx?|xlsx?|pptx?|txt|png|jpe?g|gif|zip|mp4|mp3|csv|py|js|json|md"
_KIND_WORDS = {"word document": ["docx"], "excel": ["xlsx"], "spreadsheet": ["xlsx"], "powerpoint": ["pptx"],
               "presentation": ["pptx"], "image": ["image"], "picture": ["image"], "photo": ["image"],
               "screenshot": ["png"], "python file": ["py"], "text file": ["txt"]}
_LANG_EXT = {"python": "py", "javascript": "js", "typescript": "ts", "html": "html", "css": "css", "json": "json",
             "markdown": "md", "text": "txt", "c": "c", "java": "java", "rust": "rs", "go": "go"}
_VISUAL_REF = re.compile(r"\b(top|bottom|left|right|corner|first|second|third|last|next|above|below|red|blue|green|"
                         r"yellow|black|white|icon|image|picture|screen)\b", re.I)
_STOP = {"the", "a", "an", "my", "me", "file", "files", "that", "i", "downloaded", "created", "saved", "modified",
         "today", "yesterday", "called", "named", "is", "one", "latest", "newest", "recent", "recently"}


class RouterPlanner:
    def __init__(self, registry: ToolRegistry | None = None):
        self.reg = registry or default_registry()

    def mk(self, tool: str, /, **kw) -> Step:
        return self.reg.make_step(tool, **kw)

    def plan(self, text: str, context: dict | None = None) -> TaskPlan | None:
        text, quotes = _mask(text)
        text = normalize(_TRAIL.sub("", _LEAD.sub("", text.strip())))
        if not text:
            return None
        # seed from carried-over session context (e.g. "open notepad" then, separately, "type hello") -- a
        # clause naming its own target still overrides this within the same command, same as before.
        rc: dict = {"app": context["last_app"]} if context and context.get("last_app") else {}
        steps: list[Step] = []
        clauses = [c for part in _SPLIT.split(text) for c in self._split_implicit(part)]
        for clause in clauses:
            clause = _TRAIL.sub("", clause.strip())
            try:
                got = self._clause(_unmask(clause, quotes), rc)
            except ToolError:
                return None                      # arguments the tools reject => not something the router may guess
            if got is None:
                return None
            steps.extend(got)
        steps = self._fold_site_search(steps)
        steps = self._fold_content(steps)
        return TaskPlan(_unmask(text, quotes), steps) if steps else None

    # ---- dynamic phrasing ---------------------------------------------------
    def _known_target(self, t: str) -> bool:
        t = t.strip()
        return bool(resolve_app(t) or resolve_site(t) or _URL.match(t))

    def _split_implicit(self, clause: str) -> list[str]:
        """'open chrome go to yt search cats' -> three clauses, without needing 'and'. Splits only where the words
        before the next verb are a target we recognise (app, site, URL), so ordinary text is never cut."""
        m = _IMPLICIT.match(clause.strip())
        if m and self._known_target(m.group("t")):
            return [f"{m.group('v')} {m.group('t')}"] + self._split_implicit(m.group("rest"))
        return [clause]

    @staticmethod
    def _fold_content(steps: list[Step]) -> list[Step]:
        """'create a python file ... and write hello world' -> the file is created with that content in one step."""
        out: list[Step] = []
        for s in steps:
            if s.action != "_set_content":
                out.append(s)
                continue
            target = next((o for o in reversed(out) if o.action == "create_file" and o.args["path"] == s.args["path"]), None)
            if target is None:
                return []                                  # nothing to attach to: refuse rather than guess
            target.args["content"] = s.args["content"]
        return out

    @staticmethod
    def _fold_site_search(steps: list[Step]) -> list[Step]:
        """navigate to a site then search that same site == one search on it (no wasted page load)."""
        out: list[Step] = []
        for i, s in enumerate(steps):
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            if s.note.startswith("site:") and nxt and nxt.action == "web_search" and nxt.args.get("engine") == s.note[5:]:
                continue
            out.append(s)
        return out

    def replan(self, text, context, failed, error):
        return None

    # ---- one clause ------------------------------------------------------
    def _clause(self, c: str, rc: dict) -> list[Step] | None:
        extra = handle_extra(self, c, rc)
        if extra is not NOT_MINE:
            return extra
        low = c.lower()
        m = re.match(r"^(open|launch|start|run)\s+(.+)$", low)
        if m:
            return self._open(m.group(2), c[m.start(2):], rc)
        m = re.match(r"^(?:go to|navigate to|visit|head to|head over to|take me to)\s+(.+)$", c, re.I)
        if m:
            return self._go(m.group(1).strip(), rc)
        m = re.match(r"^(close|quit|exit)\s+(.+)$", low)
        if m:
            return [self.mk("close_app", name=_clean_app(m.group(2)))]
        m = re.match(r"^(?:switch to|focus)\s+(.+)$", low)
        if m:
            return [self.mk("focus_app", name=_clean_app(m.group(1)))]
        m = re.match(r"^(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?(?:(\w+)\s+)?file(?:\s+(?:named|called)\s+(\S+))?"
                     r"(?:\s+(?:in|with|using)\s+(?:vs ?code|visual studio code|code))?$", low)
        if m:
            return self._new_file(m.group(1), m.group(2))
        m = re.match(r"^(find|locate|look for)\s+(.+)$", low)
        if m:
            return self._find(m.group(2), rc)
        m = re.match(r"^(?:search|look up|google)\b(.*)$", c, re.I)
        if m:
            return self._search(m.group(1).strip(), c, rc)
        m = re.match(r"^(type|enter|write)\s+(.+)$", c, re.I)
        if m:
            return self._type(m.group(1).lower(), m.group(2), rc)
        m = re.match(r"^(press|hit)\s+(.+)$", low)
        if m:
            keys = _keys(m.group(2))
            if keys:
                return [self.mk("hotkey", keys=keys, app=rc.get("app"))]
        m = re.match(r"^(click|tap|push|press|hit)\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button)?$", c, re.I)
        if m and m.group(1).lower() in ("click", "tap", "push"):
            if _VISUAL_REF.search(m.group(2)):     # positional/visual references need a grounder, not a name lookup
                return None
            return [self.mk("click_ui", name=m.group(2), app=rc.get("app"))]
        return self._bare_target(c, rc)

    def _bare_target(self, c: str, rc: dict) -> list[Step] | None:
        """No verb at all -- "chrome", "downloads", "bluetooth settings", "task manager".

        People say the name of the thing they want far more often than they say "open <thing>", and every one
        of those used to miss the router entirely and get handed to the model planner. Only accepted when the
        WHOLE command resolves to something concrete we already know (an installed app, a known site, a user
        folder, a Settings page), so ordinary prose is never mistaken for a launch request -- "hello there"
        resolves to nothing and still returns None. Runs last, so any explicit verb above always wins.
        """
        t = re.sub(r"^(?:the|my)\s+", "", c.strip(), flags=re.I).strip()
        if not t or len(t.split()) > 4:
            return None
        low = t.lower()
        # a bare "settings page" name: "bluetooth settings", "wifi", "dark mode" -> the Settings app page
        page = re.sub(r"\s+settings$", "", low)
        if page in SETTINGS_PAGES:
            rc["app"] = "settings"
            return [self.mk("open_settings", page=page)]
        if folder_key(re.sub(r"\s+(?:folder|directory)$", "", low)):
            fk = folder_key(re.sub(r"\s+(?:folder|directory)$", "", low))
            return [self._folder_step({"folder": fk}, fk, rc)]
        if resolve_app(low):
            return [self._launch(resolve_app(low).key, rc)]
        if resolve_site(low) or _URL.match(t):
            return self._go(t, rc)
        if is_installed_app(low):          # "task manager", "spotify" -- a real Start-menu app, not a guess
            return [self._launch(t, rc, generic=True)]
        return None

    def _open(self, target: str, raw: str, rc: dict) -> list[Step] | None:
        t = re.sub(r"^(the|my)\s+", "", target.strip())
        fm = re.match(rf"^(?:file\s+)?([\w .-]+?)\.({_EXTS})(?:\s+(?:with|in|using)\s+(.+))?$", t)
        if fm:                                    # open report.pdf [with chrome]
            app = resolve_app(fm.group(3)) if fm.group(3) else None
            if fm.group(3) and not app:
                return None
            return [self.mk("find_file", name=fm.group(1), extension=fm.group(2).lower(),
                            folder=rc.get("folder")), self.mk("open_file", app=app.key if app else None)]
        if t in ("it", "that", "this", "the file", "the pdf", "the document"):
            return [self.mk("open_file")]
        fk = folder_key(re.sub(r"\s+(folder|directory)$", "", t))
        if fk:
            return [self._folder_step({"folder": fk}, fk, rc)]
        if _WINPATH.match(raw.strip()):
            p = Path(raw.strip().replace("~", str(Path.home()), 1))
            if p.is_dir():
                return [self._folder_step({"path": str(p)}, p.name, rc)]
            return None
        settings_key = re.sub(r"\s+settings$", "", t)
        if t.endswith("settings") and settings_key in SETTINGS_PAGES:
            rc["app"] = "settings"
            return [self.mk("open_settings", page=settings_key)]
        if t == "settings":
            return [self._launch("settings", rc)]
        if _URL.match(t):
            return [self._url_step(t, rc)]
        app = resolve_app(t)
        if app:
            return [self._launch(app.key, rc)]
        if resolve_site(t):                       # "open youtube" means the website, not a Start-menu app
            return self._go(t, rc)
        corrected, alts = fuzzy_resolve(t)
        if alts:
            raise NeedsClarification(f"'{t}' is ambiguous -- did you mean one of: {', '.join(alts)}?", alts)
        if corrected:
            app = resolve_app(corrected)
            if app:
                return [self._launch(app.key, rc)]
            if resolve_site(corrected):
                return self._go(corrected, rc)
        words = t.split()
        if 1 <= len(words) <= 2 and not re.search(r"\b(file|folder|document|it|that|this|latest|new|recent)\b", t):
            return [self._launch(t, rc, generic=True)]
        return None

    def _launch(self, name: str, rc: dict, generic: bool = False) -> Step:
        spec = generic_app(name) if generic else resolve_app(name)
        rc["app"], rc["fresh"] = spec.key, spec.key
        return self.mk("open_app", name=name if generic else spec.key)

    def _folder_step(self, args: dict, title: str, rc: dict) -> Step:
        rc["app"], rc["folder"] = "explorer", args.get("folder") or args.get("path")
        return self.mk("open_folder", **args)

    def _url_step(self, url: str, rc: dict) -> Step:
        return self.mk("navigate_url", url=url, browser=self._browser(rc))

    def _go(self, target: str, rc: dict) -> list[Step] | None:
        """'go to X': a full URL, a known short name (yt), an installed app, or a bare word -> word.com."""
        t = re.sub(r"^(the|my)\s+|\s+(website|site|page)$", "", target.strip())
        host = target if _URL.match(target) else resolve_site(t)
        if host is None:
            app = resolve_app(t)
            if app:
                return [self._launch(app.key, rc)]
            corrected, alts = fuzzy_resolve(t)
            if alts:
                raise NeedsClarification(f"'{t}' is ambiguous -- did you mean one of: {', '.join(alts)}?", alts)
            if corrected:
                host = resolve_site(corrected)
                if host is None:
                    app = resolve_app(corrected)
                    if app:
                        return [self._launch(app.key, rc)]
            if host is None:
                if re.fullmatch(r"[a-z0-9-]{2,30}", t.lower()):
                    host = t.lower() + ".com"      # generic guess for any other single word; verified by the title check
                else:
                    return None
        step = self._url_step(host, rc)
        bare = re.sub(r"^https?://(www\.)?", "", step.args["url"]).split("/")[0]
        engine = SITE_ENGINE.get(bare)
        if engine:
            rc["engine"] = engine                  # a following "search X" searches THIS site
            step.note = f"site:{engine}"
        return [step]

    @staticmethod
    def _browser(rc: dict):
        return rc.get("app") if rc.get("app") in ("chrome", "edge", "firefox") else None

    def _search(self, rest: str, raw: str, rc: dict) -> list[Step] | None:
        engine = rc.get("engine", "google")
        low = rest.lower()
        m = re.match(r"^(youtube|google|bing|duckduckgo|wikipedia)\s+for\s+(.+)$", rest, re.I)
        if m:
            engine, query = m.group(1).lower(), m.group(2)
        else:
            m = re.match(r"^(?:the web |online |the internet )?(?:for\s+)?(.+?)(?:\s+(?:on|in|using|with)\s+"
                         r"(youtube|google|bing|duckduckgo|wikipedia))?$", rest, re.I)
            if not m:
                return None
            query, engine = m.group(1), (m.group(2) or engine).lower()
        # "search my downloads for X" is a file search, not a web search
        fm = re.match(r"^(?:in\s+)?(?:my\s+|the\s+)?(\w+)(?:\s+folder)?\s+for\s+(.+)$", query, re.I)
        if fm and folder_key(fm.group(1)):
            return self._find(f"{fm.group(2)} in {fm.group(1)}", rc)
        query = re.sub(r"^(?:for\s+)", "", query.strip())
        if not query or query.lower() in ("it", "that", "this"):
            return None
        return [self.mk("web_search", query=query, engine=engine, browser=self._browser(rc))]

    def _find(self, rest: str, rc: dict) -> list[Step] | None:
        s = rest
        roots, use_last = None, False
        m = re.search(r"\b(?:in|from|inside|under)\s+(?:my\s+|the\s+)?(\w+)(?:\s+folder)?\b", s)
        if m and folder_key(m.group(1)):
            roots = [folder_key(m.group(1))]
            s = s[:m.start()] + s[m.end():]
        elif rc.get("app") == "explorer":
            use_last = True
        when = "today" if "today" in s else "yesterday" if "yesterday" in s else None
        exts = [e.lower() for e in re.findall(rf"\b({_EXTS})\b", s)]
        for phrase, ee in _KIND_WORDS.items():
            if phrase in s:
                exts += ee
                s = s.replace(phrase, " ")
        if not roots and not use_last:
            roots = ["downloads"] if "downloaded" in s else ["desktop", "documents", "downloads"]
        words = [w for w in re.findall(r"[\w.-]+", s) if w not in _STOP and not re.fullmatch(_EXTS, w)]
        if not (exts or words or when):
            return None
        folder = roots[0] if roots and len(roots) == 1 else (rc.get("folder") if use_last else None)
        return [self.mk("find_file", name=" ".join(words) or None, extension=exts[0] if exts else None,
                        folder=folder, modified=when)]

    def _type(self, verb: str, rest: str, rc: dict) -> list[Step] | None:
        into, rest = rc.get("app"), rest.strip()
        m = re.fullmatch(r'"(.*)"(?:\s+(?:in|into|on)\s+(?:the\s+)?([\w ]+))?', rest, re.S)
        quoted = m is not None
        if m:
            text, target = m.group(1), m.group(2)
        else:
            m = re.fullmatch(r"(.*?)(?:\s+(?:in|into|on)\s+(?:the\s+)?([\w ]+))?", rest, re.S)
            text, target = m.group(1), m.group(2)
            if target and not resolve_app(target):       # "on the table" is part of the text
                text, target = rest, None
            text = re.sub(r"^text\s+", "", text)
        if verb == "write" and not quoted:
            # "write hello" right after opening an app means type it; "write me a poem about X" means generate it
            if not into or _GENERATION.search(text):
                return None
        if target:
            app = resolve_app(target)
            if not app:
                return None
            into = app.key
        if not text:
            return None
        steps = [self.mk("type_text", text=text, app=into)]
        if into == "notepad" and rc.get("fresh") == "notepad":
            # Win11 Notepad restores the previous session; never type into the user's old tab.
            steps.insert(0, self.mk("hotkey", keys="ctrl+n", app="notepad"))
        return steps

    def _new_file(self, lang: str | None, name: str | None) -> list[Step] | None:
        ext = _LANG_EXT.get((lang or "python").lower())
        if ext is None:
            return None
        path = f"documents/{name or 'untitled.' + ext}"
        return [self.mk("create_file", path=path, content=""), self.mk("open_file", path=path, app="vscode")]


# ---- helpers ----------------------------------------------------------------
def _clean_app(s: str) -> str:
    return re.sub(r"^(the|my)\s+|\s+(app|application|window)$", "", s.strip())


def _keys(s: str) -> str | None:
    s = s.strip().lower().replace("control", "ctrl").replace("escape", "esc")
    parts = [p for p in re.split(r"\s*(?:\+|\s)\s*", s) if p]
    return "+".join(parts) if parts and all(vk_for(p) is not None for p in parts) else None


def _mask(text: str):
    quotes: list[str] = []

    def sub(m):
        quotes.append(m.group(0))
        return f"\x00{len(quotes) - 1}\x00"
    return re.sub(r'"[^"]*"', sub, text), quotes


def _unmask(text: str, quotes: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: quotes[int(m.group(1))], text)
