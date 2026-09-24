"""Offline router coverage/accuracy over bench/corpus.py (touches nothing on the desktop)."""
from __future__ import annotations

import time

from cua.agent.router import RouterPlanner

from .corpus import IN_SCOPE, OUT_OF_SCOPE


def run() -> dict:
    r = RouterPlanner()
    hit = right = 0
    bad, t0 = [], time.perf_counter()
    for text, exp in IN_SCOPE:
        p = r.plan(text)
        got = [s.action for s in p.steps] if p else None
        hit += p is not None
        if got == exp:
            right += 1
        else:
            bad.append((text, exp, got))
    plan_ms = (time.perf_counter() - t0) * 1000 / (len(IN_SCOPE))
    false_pos = [t for t in OUT_OF_SCOPE if r.plan(t) is not None]
    n_all = len(IN_SCOPE) + len(OUT_OF_SCOPE)
    return {
        "in_scope_n": len(IN_SCOPE), "in_scope_hit": hit, "in_scope_exact": right,
        "out_of_scope_n": len(OUT_OF_SCOPE), "out_of_scope_false_plans": false_pos,
        "coverage_on_mixed_corpus": round((hit) / n_all, 3),   # share of ALL commands router claims
        "mean_plan_ms": round(plan_ms, 3), "mismatches": bad,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1))
