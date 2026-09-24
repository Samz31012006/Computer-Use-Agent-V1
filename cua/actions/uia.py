"""UIAEngine: UI Automation implementations (named elements, control types, value patterns).

Also the *alternative mechanism* for type_text: `repair()` fixes text that SendInput partly lost, by checking the
element's actual value first and only setting what is missing, so it is safe to run after a failed verification.
"""
from __future__ import annotations

from cua.actions.skills import SkillEngine, _app, topmost_window
from cua.agent.state import Ctx
from cua.types import EngineResult, Step

_TOOLS = {"click_ui", "type_into_ui", "read_ui_text", "find_ui", "inspect_controls"}


class UIAEngine:
    name = "uia"

    def __init__(self):
        self._skills = SkillEngine()      # reused only for its _open_app, to auto-launch a missing target app

    def can_handle(self, step: Step) -> bool:
        return step.action in _TOOLS

    def execute(self, step: Step, ctx: Ctx) -> EngineResult:
        return getattr(self, "_" + step.action)(step.args, ctx)

    # ---- helpers ---------------------------------------------------------------
    def _target(self, a, ctx: Ctx) -> int | EngineResult:
        if a.get("app"):
            with ctx.log.span("select"):
                w = topmost_window(ctx, a["app"])
            if not w:
                launched = self._skills._open_app({"name": a["app"]}, ctx)   # not open yet: launch it first
                if not launched.ok:
                    return launched
                return ctx.state.context.get("last_hwnd")
            with ctx.log.span("select"):
                last = ctx.state.context.get("last_hwnd")
                hwnd = last if any(x.hwnd == last for x in ctx.perception.match(_app(a["app"]))) else w.hwnd
            with ctx.log.span("act"):
                ctx.driver.focus(hwnd)
            return hwnd
        with ctx.log.span("select"):
            fg = ctx.perception.foreground()
            if not fg:
                return EngineResult(False, "no foreground window")
            return fg.hwnd

    def _resolve(self, a, ctx: Ctx, hwnd: int):
        with ctx.log.span("select"):
            if a.get("control_type") or not a.get("name"):
                found = ctx.perception.find_elements(hwnd, a.get("name"), a.get("control_type"), limit=1)
                return found[0] if found else None
            return ctx.perception.find_element(hwnd, a["name"], timeout=float(a.get("timeout", 0.6)))

    # ---- tools -----------------------------------------------------------------
    def _click_ui(self, a, ctx: Ctx) -> EngineResult:
        hwnd = self._target(a, ctx)
        if isinstance(hwnd, EngineResult):
            return hwnd
        ref = self._resolve(a, ctx, hwnd)
        if ref is None:
            return EngineResult(False, f"element {a.get('name') or a.get('control_type')!r} not found", retryable=False)
        with ctx.log.span("act"):
            try:
                inv = ref.native.GetInvokePattern()
                if inv is None:
                    raise AttributeError
                inv.Invoke()
                how = "invoke"
            except Exception:
                cx, cy = (ref.rect[0] + ref.rect[2]) // 2, (ref.rect[1] + ref.rect[3]) // 2
                ctx.driver.click(cx, cy)
                how = "mouse"
        return EngineResult(True, f"{how} '{ref.name}'")

    def _type_into_ui(self, a, ctx: Ctx) -> EngineResult:
        hwnd = self._target(a, ctx)
        if isinstance(hwnd, EngineResult):
            return hwnd
        ref = self._resolve({"name": a["name"]}, ctx, hwnd)
        if ref is None:
            return EngineResult(False, f"element '{a['name']}' not found")
        with ctx.log.span("act"):
            try:
                ref.native.GetValuePattern().SetValue(a["text"])      # direct value set: no dropped keystrokes
                return EngineResult(True, f"set '{ref.name}'")
            except Exception:
                ctx.driver.click((ref.rect[0] + ref.rect[2]) // 2, (ref.rect[1] + ref.rect[3]) // 2)
                ctx.driver.type_text(a["text"])
        return EngineResult(True, f"typed into '{ref.name}'")

    def _read_ui_text(self, a, ctx: Ctx) -> EngineResult:
        hwnd = self._target(a, ctx)
        if isinstance(hwnd, EngineResult):
            return hwnd
        with ctx.log.span("select"):
            txt = ctx.perception.element_text(hwnd, a["name"]) if a.get("name") else ctx.perception.read_text(hwnd)
        ctx.state.context["read_text"] = txt
        return EngineResult(bool(txt), txt[:200] or "no text")

    def _find_ui(self, a, ctx: Ctx) -> EngineResult:
        hwnd = self._target(a, ctx)
        if isinstance(hwnd, EngineResult):
            return hwnd
        ref = self._resolve(a, ctx, hwnd)
        return EngineResult(ref is not None, f"{ref.role} '{ref.name}'" if ref else "not found")

    def _inspect_controls(self, a, ctx: Ctx) -> EngineResult:
        hwnd = self._target(a, ctx)
        if isinstance(hwnd, EngineResult):
            return hwnd
        with ctx.log.span("select"):
            items = ctx.perception.inspect(hwnd)
        ctx.state.context["controls"] = items
        return EngineResult(True, "; ".join(f"{i['type']}:{i['name']}" for i in items))

    # ---- repair (state-aware, safe after a failed verification) ------------------
    def repair(self, step: Step, ctx: Ctx) -> EngineResult | None:
        if step.action != "type_text":
            return None
        hwnd = ctx.state.context.get("input_hwnd")
        if not hwnd:
            return None
        text, cur = step.args["text"], ctx.perception.read_text(hwnd)
        if text in cur:
            return EngineResult(True, "already present")
        # keystrokes lost at the start: the field ends with a proper suffix of the text -> fill in the missing head
        for i in range(1, len(text)):
            if cur.endswith(text[i:]):
                fixed = cur[:len(cur) - len(text[i:])] + text
                with ctx.log.span("act"):
                    ok = _set_focused_value(fixed)
                return EngineResult(ok, f"repaired {i} lost char(s)" if ok else "cannot set value")
        return None


def _set_focused_value(value: str) -> bool:
    try:
        import uiautomation as auto
        c = auto.GetFocusedControl()
        c.GetValuePattern().SetValue(value)
        return True
    except Exception:
        return False

