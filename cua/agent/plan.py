"""The plan schema (Pydantic) and its compilation into executable Steps.

Model output is untrusted: it is parsed as JSON, validated against PlanModel, then every step is checked against the
ToolRegistry (known tool, valid arguments) before anything can run. A plan never contains code or shell commands.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cua.actions.base import ToolError
from cua.actions.registry import ToolRegistry
from cua.types import TaskPlan


class PlanError(ValueError):
    """The planner produced something unusable. The message is short enough to feed back to the model."""


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    target: str | None = None                    # shorthand for the tool's primary argument
    arguments: dict[str, Any] = Field(default_factory=dict)
    verification: str | None = None              # optional DSL, e.g. "title:SpaceX" (see parse_verification)


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    goal: str = Field(default="", max_length=300)
    steps: list[PlanStep] = Field(min_length=1, max_length=20)
    success_condition: str | None = None
    max_steps: int = Field(default=12, ge=1, le=20)


_DSL = re.compile(r"^(title|window|element|file|foreground|screen)\s*:\s*(.+)$", re.I)


def parse_verification(text: str | None) -> dict | None:
    """'title:SpaceX' | 'window:notepad' | 'element:Save' | 'file:documents/a.txt' | 'foreground:chrome' |
    'screen:Liked' (OCR text present -- for canvas-rendered content UIA can't see, e.g. after a visually-
    grounded click)."""
    m = _DSL.match(text or "")
    if not m:
        return None
    kind, val = m.group(1).lower(), m.group(2).strip()
    return {"title": {"kind": "window_title", "any_of": [val], "timeout": 8},
            "window": {"kind": "app_window", "app": val, "timeout": 8},
            "element": {"kind": "element", "name": val, "timeout": 4},
            "file": {"kind": "file_exists", "path_text": val, "timeout": 3},
            "foreground": {"kind": "foreground_app", "app": val, "timeout": 3},
            "screen": {"kind": "screen_text", "contains": val, "timeout": 5}}[kind]


def compile_plan(model: PlanModel, registry: ToolRegistry, source: str = "llm") -> TaskPlan:
    if len(model.steps) > model.max_steps:
        raise PlanError(f"plan has {len(model.steps)} steps, limit is {model.max_steps}")
    steps = []
    for i, ps in enumerate(model.steps, 1):
        spec = registry.get(ps.action)
        if spec is None:
            raise PlanError(f"step {i}: unknown tool '{ps.action}'")
        args = dict(ps.arguments)
        if ps.target and spec.primary and spec.primary not in args:
            args[spec.primary] = ps.target
        try:
            step = registry.make_step(ps.action, **args)
        except ToolError as e:
            raise PlanError(f"step {i}: {e}") from e
        hint = parse_verification(ps.verification)
        if hint and step.verify is None:            # a model hint may add a check, never replace a built-in one
            step.verify = hint
        steps.append(step)
    return TaskPlan(goal=model.goal, steps=steps, source=source, success_condition=model.success_condition,
                    max_steps=model.max_steps)


def parse_plan_json(text: str, registry: ToolRegistry, source: str = "llm") -> TaskPlan:
    """Text from a model -> validated TaskPlan, or PlanError. Tolerates code fences and leading/trailing prose."""
    raw = text.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise PlanError("no JSON object in output")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise PlanError(f"invalid JSON: {e.msg}") from e
    try:
        model = PlanModel.model_validate(data)
    except ValidationError as e:
        err = e.errors()[0]
        raise PlanError(f"schema: {'.'.join(map(str, err['loc']))}: {err['msg']}") from e
    return compile_plan(model, registry, source)
