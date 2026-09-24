"""Planner seam + the router -> LLM chain."""
from __future__ import annotations

from typing import Protocol

from cua.types import Step, TaskPlan


class Planner(Protocol):
    def plan(self, text: str, context: dict) -> TaskPlan | None: ...
    def replan(self, text: str, context: dict, failed: Step, error: str) -> TaskPlan | None: ...


class ChainPlanner:
    """Fast deterministic router first; the model only for what the router does not understand.
    context['router_hit'] / context['planner'] record the decision for the logs and the benchmark."""

    def __init__(self, router: Planner, llm: Planner | None = None):
        self.router, self.llm = router, llm

    def plan(self, text: str, context: dict) -> TaskPlan | None:
        plan = self.router.plan(text, context)
        context["router_hit"] = plan is not None
        if plan is not None:
            context["planner"] = "router"
            return plan
        if self.llm is None:
            context["planner"] = "none"
            return None
        context["planner"] = "llm"
        return self.llm.plan(text, context)

    def replan(self, text: str, context: dict, failed: Step, error: str) -> TaskPlan | None:
        if self.llm is None:
            return None
        context["planner"] = "llm"
        return self.llm.replan(text, context, failed, error)
