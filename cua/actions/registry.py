"""ToolRegistry: the single source of truth for what the agent can do."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from cua.actions.base import Safety, ToolError, ToolSpec
from cua.types import Step


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec] = ()):
        self._tools: dict[str, ToolSpec] = {}
        for t in tools:
            self.register(t)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def make_step(self, name: str, /, **args) -> Step:
        """Validate args against the declaration and attach the tool's own verification. Raises ToolError."""
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool '{name}'")
        clean = spec.check_args({k: v for k, v in args.items() if v is not None})
        return Step(name, clean, spec.verify(clean) if spec.verify else None)

    def subset(self, names: Iterable[str]) -> "ToolRegistry":
        return ToolRegistry(self._tools[n] for n in names if n in self._tools)

    def is_idempotent(self, name: str) -> bool:
        spec = self._tools.get(name)
        return spec.idempotent if spec else False      # unknown => never repeat

    def confirm_reason(self, step: Step) -> str | None:
        """Declared risk only: CONFIRM tools, or data-dependent risk (e.g. overwriting an existing file)."""
        spec = self._tools.get(step.action)
        if spec is None:
            return f"run unknown action {step.action}"
        if spec.safety is Safety.CONFIRM:
            return f"{step.action.replace('_', ' ')} {step.args.get('name') or step.args.get('src') or ''}".strip()
        return spec.confirm_if(step.args) if spec.confirm_if else None

    # ---- what the planner sees ---------------------------------------------
    def prompt(self) -> str:
        """Compact tool list: ~15 tokens per tool."""
        return "\n".join(t.prompt_line() for t in self._tools.values())

    @staticmethod
    def _arg_schema(a) -> dict:
        if a.enum:
            return {"enum": list(a.enum)}
        if a.type == "int":
            return {"type": "integer"}
        if a.type == "float":
            return {"type": "number"}
        if a.type == "bool":
            return {"type": "boolean"}
        return {"type": "string", "maxLength": 1500 if a.name in ("content", "text") else 200}

    def json_schema(self, max_steps: int = 8, per_tool: bool = True) -> dict:
        """Schema for constrained decoding. With per_tool, each step must be exactly one declared tool with exactly
        its declared arguments (names, types, enums), so hallucinated tools and malformed arguments are
        unrepresentable; validation still runs afterwards."""
        if per_tool:
            item = {"oneOf": [{
                "type": "object",
                "properties": {"action": {"const": t.name},
                               "arguments": {"type": "object", "additionalProperties": False,
                                             "properties": {a.name: self._arg_schema(a) for a in t.args},
                                             "required": [a.name for a in t.args if a.required]}},
                "required": ["action", "arguments"], "additionalProperties": False} for t in self._tools.values()]}
        else:
            item = {"type": "object", "properties": {"action": {"type": "string", "enum": self.names()},
                                                     "target": {"type": "string"}, "arguments": {"type": "object"}},
                    "required": ["action"]}
        return {"type": "object", "properties": {"steps": {"type": "array", "minItems": 1, "maxItems": max_steps,
                                                           "items": item}}, "required": ["steps"]}


@lru_cache(maxsize=1)
def default_registry() -> ToolRegistry:
    from cua.actions.tools import build_tools
    return ToolRegistry(build_tools())
