"""LLMPlanner: natural language -> validated TaskPlan using a small local model.

The model only PLANS. It emits JSON naming tools from the registry; the JSON schema passed to the runtime makes
hallucinated tool names unrepresentable, Pydantic + the registry validate everything again afterwards, and the
safety policy still gates every step at execution time. It never produces code or shell commands.

Prompt strategy ('variant', chosen by measurement in bench/planner_eval.py):
  dynamic  - two worked examples picked by keyword from a pool, so the model copies a *relevant* shape
  static   - the same two examples every time
  zero     - no examples
"""
from __future__ import annotations

import json
import re
from typing import Callable

from cua.actions.registry import ToolRegistry
from cua.agent.plan import PlanError, parse_plan_json
from cua.config import Config
from cua.models.runtime import GenResult, ModelUnavailable, Runtime
from cua.perception.base import describe_state
from cua.types import Step, TaskPlan

_ALWAYS = ("open_app", "wait")

_GROUPS: list[tuple[set[str], tuple[str, ...]]] = [
    # (trigger words/phrases, tools in this group)
    ({"browse", "browser", "web", "search", "google", "look up", "weather", "news", "youtube", "find out",
      "online", "internet", "website", "url", "visit", "head", "go to", "navigate", "wiki", "bing",
      "gmail", "reddit", "netflix", "amazon", "site", "page"},
     ("navigate_url", "web_search")),

    ({"type", "write", "text", "enter", "dictate", "input", "note", "compose"},
     ("type_text",)),

    ({"key", "press", "hit", "hotkey", "shortcut", "save", "undo", "redo", "copy", "paste", "cut",
      "ctrl", "alt", "escape", "tab", "enter", "select all", "screenshot", "find on page", "bookmark"},
     ("hotkey", "press_key")),

    ({"close", "quit", "exit", "rid", "kill", "terminate", "dismiss", "put away", "shut"},
     ("close_app",)),

    ({"focus", "switch", "bring", "front", "see", "go to", "other window", "back to"},
     ("focus_app", "switch_window", "find_window", "read_window_title")),

    ({"minimize", "shrink", "hide", "background", "smaller", "out of the way", "tray"},
     ("minimize_app",)),

    ({"maximize", "fill", "enlarge", "bigger", "full screen", "expand"},
     ("maximize_app",)),

    ({"file", "find", "locate", "search for", "where", "newest", "latest", "yesterday", "today",
      "pdf", "doc", "xls", "zip", "resume", "invoice", "report", "spreadsheet", "lost"},
     ("find_file", "open_file")),

    ({"create", "make", "new file", "write a", "python", "javascript", "html", "code", "script",
      "program", "hello world", "rust", "java", "go file", "txt"},
     ("create_file", "open_file")),

    ({"copy", "duplicate", "move", "relocate", "transfer"},
     ("copy_file", "move_file")),

    ({"folder", "directory", "downloads", "documents", "desktop", "pictures", "videos", "music",
      "list", "contents", "what's in", "show me"},
     ("open_folder", "list_directory")),

    ({"click", "button", "press the", "tap", "select the", "ok", "cancel", "submit", "apply", "next"},
     ("click_ui",)),

    ({"scroll", "page down", "page up", "bottom", "top"},
     ("scroll",)),

    ({"settings", "wifi", "bluetooth", "display", "sound", "update", "network", "dark mode",
      "light mode", "personalization", "preferences", "configuration"},
     ("open_settings",)),

    ({"read", "inspect", "controls", "what controls", "status bar", "value", "text of"},
     ("inspect_controls", "read_ui_text", "find_ui", "type_into_ui")),

    ({"window", "title", "which window", "what app", "what application"},
     ("read_window_title",)),
]
_GROUP_WORDS: list[tuple[list[str], tuple[str, ...]]] = [(sorted(ws, key=len, reverse=True), ts) for ws, ts in _GROUPS]

_SYSTEM = ("You plan tasks on a Windows PC. Reply with ONE JSON object and nothing else, in the form "
           '{"steps": [{"action": tool_name, "arguments": {argument_name: value}}]}.')


def _plan(*steps) -> str:
    return json.dumps({"steps": [{"action": a, "arguments": g} for a, g in steps]}, separators=(",", ":"))


# (trigger words, request, plan). Values are deliberately unlike anything in the evaluation set.
_POOL = [
    ({"search", "look", "weather", "google", "youtube", "web", "videos", "news", "find out", "browser"},
     "open chrome and search for cats", _plan(("open_app", {"name": "chrome"}), ("web_search", {"query": "cats", "browser": "chrome"}))),
    ({"file", "text", "write", "make", "create", "save", "python", "code", "saying", "called"},
     "make notes.txt saying buy milk",
     _plan(("create_file", {"path": "documents/notes.txt", "content": "buy milk"}), ("open_file", {"path": "documents/notes.txt"}))),
    ({"close", "quit", "exit", "rid", "kill"}, "close the calculator", _plan(("close_app", {"name": "calculator"}))),
    ({"folder", "show", "downloads", "documents", "pictures", "take", "list", "directory"}, "show my pictures folder",
     _plan(("open_folder", {"folder": "pictures"}))),
    ({"find", "locate", "where", "newest", "latest", "yesterday", "today"}, "find the newest zip file from today",
     _plan(("find_file", {"extension": "zip", "modified": "today"}))),
    ({"type", "write", "enter", "dictate"}, "type hello in notepad",
     _plan(("open_app", {"name": "notepad"}), ("type_text", {"text": "hello", "app": "notepad"}))),
    ({"settings", "wifi", "bluetooth", "dark", "display", "sound", "mode"}, "open bluetooth settings",
     _plan(("open_settings", {"page": "bluetooth"}))),
    ({"go", "head", "visit", "website", "site", "url"}, "go to github.com", _plan(("navigate_url", {"url": "github.com"}))),
    ({"press", "key", "enter", "escape", "tab", "save", "copy", "paste", "undo"}, "press escape",
     _plan(("press_key", {"key": "esc"}))),
    ({"scroll", "down", "up"}, "scroll up", _plan(("scroll", {"amount": 3}))),
    ({"click", "button", "press", "select", "tap"}, "click the OK button", _plan(("click_ui", {"name": "OK"}))),
    ({"maximize", "minimize", "fill", "shrink", "window", "switch", "focus", "bring"}, "maximize firefox",
     _plan(("maximize_app", {"name": "firefox"}))),
    ({"copy", "move", "duplicate"}, "copy a.txt from desktop to pictures",
     _plan(("copy_file", {"src": "desktop/a.txt", "dst": "pictures"}))),
]
_WORD = re.compile(r"[a-z']+")


def pick_examples(text: str, k: int = 2) -> list[tuple[str, str]]:
    words = set(_WORD.findall(text.lower()))
    scored = sorted(((len(keys & words), i) for i, (keys, _, _) in enumerate(_POOL)), key=lambda t: (-t[0], t[1]))
    chosen = [i for s, i in scored if s > 0][:k] or [0, 1]
    return [(_POOL[i][1], _POOL[i][2]) for i in chosen]


def select_tools(text: str, registry: ToolRegistry) -> ToolRegistry:
    """Send only the tools that plausibly matter: fewer tools => shorter prompt => faster prefill, less confusion."""
    low = text.lower()
    names = set(_ALWAYS)
    for phrases, tools in _GROUP_WORDS:
        hits = sum(1 for phrase in phrases if phrase in low)
        if hits:
            names.update(tools)
    if len(names) <= len(_ALWAYS):
        names.update(("navigate_url", "web_search", "type_text", "hotkey", "click_ui",
                       "find_file", "open_file", "open_folder"))
    return registry.subset(names)


def build_prompt(text: str, tools: ToolRegistry, state: str, fmt: str, max_steps: int, failure: str | None = None,
                 variant: str = "dynamic") -> str:
    if variant == "zero":
        ex = ""
    else:
        pairs = pick_examples(text) if variant == "dynamic" else [(_POOL[0][1], _POOL[0][2]), (_POOL[1][1], _POOL[1][2])]
        ex = "Examples (different requests; use the words of THIS request, not the examples):\n" + "".join(
            f"Request: {r}\n{j}\n" for r, j in pairs)
    # static prefix first: llama-server caches the shared prompt prefix between calls
    body = (f"Tools:\n{tools.prompt()}\n"
            f"Rules: at most {max_steps} steps; shortest plan; open_app before typing or clicking in an app; use "
            "web_search/navigate_url for the web; create_file to write code then open_file(app=...); only these tools.\n"
            f"{ex}"
            + (f"State: {state}\n" if state else "")
            + (f"A previous attempt failed: {failure}\nGive a NEW plan for what remains.\n" if failure else "")
            + f"Request: {text}\n")
    if fmt in ("chatml", "qwen3"):
        # Qwen3 is a reasoning model: an empty think block selects its non-thinking mode (no wasted tokens)
        think = "<think>\n\n</think>\n\n" if fmt == "qwen3" else ""
        return (f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n"
                f"<|im_start|>assistant\n{think}")
    return f"{_SYSTEM}\n{body}"


def resolve_format(cfg: Config) -> str:
    if cfg.chat_format != "auto":
        return cfg.chat_format
    m = cfg.resolve_model()
    return "qwen3" if m is not None and "qwen3" in m.name.lower() else "chatml"


class LLMPlanner:
    def __init__(self, runtime: Runtime, registry: ToolRegistry, cfg: Config | None = None,
                 perception=None, abort: Callable[[], bool] | None = None):
        self.rt, self.reg, self.cfg = runtime, registry, cfg or Config()
        self.perception, self.abort = perception, abort
        self.last: GenResult | None = None       # for logging: latency/tokens of the most recent call

    def _state(self) -> str:
        return describe_state(self.perception) if self.perception is not None else ""

    @staticmethod
    def _context_hint(context: dict) -> str:
        """A few carried-over facts (active app, previous request), not a growing conversation history --
        see AgentService, which is what actually maintains this dict across tasks."""
        if not context:
            return ""
        bits = []
        if context.get("last_app"):
            bits.append(f"active app is {context['last_app']}")
        if context.get("last_task_text"):
            bits.append(f"previous request was \"{context['last_task_text']}\" ({'ok' if context.get('last_task_ok') else 'failed'})")
        return ("; ".join(bits) + ".") if bits else ""

    def _run(self, text: str, failure: str | None, context: dict) -> TaskPlan | None:
        tools = select_tools(text, self.reg)
        state = " ".join(s for s in (self._state(), self._context_hint(context)) if s)
        prompt = build_prompt(text, tools, state, resolve_format(self.cfg), self.cfg.max_steps, failure,
                              self.cfg.prompt_variant)
        schema = tools.json_schema(self.cfg.max_steps, per_tool=self.cfg.per_tool_schema)
        err = ""
        for attempt in range(1 + self.cfg.llm_retries):
            p = prompt if not err else prompt + f"Your last output was invalid ({err}). Output corrected JSON only.\n"
            try:
                gen = self.rt.generate(p, schema, self.cfg.max_new_tokens, self.abort)
            except ModelUnavailable as e:
                context["planner_error"] = str(e)
                return None
            self.last = gen
            context.setdefault("planner_calls", []).append(
                {"ms": round(gen.total_ms), "prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens,
                 "tok_s": round(gen.tok_s, 1), "attempt": attempt + 1})
            try:
                plan = parse_plan_json(gen.text, self.reg, source="llm")
                plan.goal = plan.goal or text
                return plan
            except PlanError as e:
                err = str(e)
                context["planner_error"] = err
        return None

    def plan(self, text: str, context: dict) -> TaskPlan | None:
        return self._run(text, None, context)

    def replan(self, text: str, context: dict, failed: Step, error: str) -> TaskPlan | None:
        args = ", ".join(f"{k}={v!r}" for k, v in failed.args.items())[:120]
        return self._run(text, f"{failed.action}({args}) -> {error[:160]}", context)
