"""Tool declarations and the ActionEngine seam.

A ToolSpec is pure metadata: what the tool is called, what it does, what arguments it takes, how risky it is and how
to verify it. The planner (router or LLM) only ever sees ToolSpecs; engines supply the implementation. That is what
lets a model choose tools dynamically without inventing any.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from cua.types import EngineResult, Step


class Safety(str, Enum):
    SAFE = "safe"          # runs without asking
    CONFIRM = "confirm"    # always needs the user's OK


class ToolError(ValueError):
    """Bad arguments for a tool (unknown arg, wrong type, missing required, out of enum)."""


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str = "str"                    # str | int | float | bool
    required: bool = True
    description: str = ""
    enum: tuple[str, ...] | None = None
    strip: bool = True                   # False for content/text where whitespace is meaningful


_CAST = {"str": str, "int": int, "float": float}


@dataclass
class ToolSpec:
    name: str
    description: str
    args: tuple[ArgSpec, ...] = ()
    safety: Safety = Safety.SAFE
    idempotent: bool = True              # safe to repeat after a failed *verification*
    verify: Callable[[dict], dict | None] | None = None        # args -> predicate spec (see verification/)
    confirm_if: Callable[[dict], str | None] | None = None     # args -> reason, for data-dependent risk
    prepare: Callable[[dict], dict] | None = None              # normalise/validate args after type checks
    primary: str | None = None           # arg that a planner's bare "target" maps to (default: first required)

    def __post_init__(self):
        if self.primary is None:
            req = [a.name for a in self.args if a.required]
            self.primary = req[0] if req else (self.args[0].name if self.args else None)

    def prompt_line(self) -> str:
        def one(a):
            v = a.name + ("" if a.required else "?")
            if a.enum:      # show allowed values: a small model picks from what it can see
                v += ": " + "|".join(a.enum[:6]) + ("|..." if len(a.enum) > 6 else "")
            elif a.type in ("int", "float"):
                v += f": {a.type}"
            return v
        sig = ", ".join(one(a) for a in self.args)
        return f"{self.name}({sig}): {self.description}"

    def check_args(self, raw: dict[str, Any]) -> dict[str, Any]:
        known = {a.name: a for a in self.args}
        extra = set(raw) - set(known)
        if extra:
            raise ToolError(f"{self.name}: unknown argument(s) {sorted(extra)}; allowed {sorted(known)}")
        out: dict[str, Any] = {}
        for a in self.args:
            v = raw.get(a.name)
            if v is None or v == "":
                if a.required:
                    raise ToolError(f"{self.name}: missing required argument '{a.name}'")
                continue
            try:
                if a.type == "bool":
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                else:
                    v = _CAST[a.type](v)
            except (TypeError, ValueError):
                raise ToolError(f"{self.name}: argument '{a.name}' must be {a.type}") from None
            if a.type == "str":
                v = v.strip() if a.strip else v
                if len(v) > 4000:
                    raise ToolError(f"{self.name}: argument '{a.name}' too long")
            if a.enum and str(v).lower() not in a.enum:
                raise ToolError(f"{self.name}: '{a.name}' must be one of {list(a.enum)}")
            out[a.name] = v.lower() if a.enum else v
        return self.prepare(out) if self.prepare else out


class ActionEngine(Protocol):
    """Implements some subset of tools. Engines are tried in order, so a later engine is the 'alternative
    mechanism' for tools an earlier one also handles (e.g. UIA ValuePattern as a fallback for typed text)."""
    name: str

    def can_handle(self, step: Step) -> bool: ...
    def execute(self, step: Step, ctx: Any) -> EngineResult: ...
