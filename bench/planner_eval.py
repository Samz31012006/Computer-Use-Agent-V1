"""Planner evaluation: how good is a (model, prompt-variant) pair at turning language into tool plans?

  python -m bench.planner_eval                                   # all variants, default model
  python -m bench.planner_eval --variants dynamic --model models/other.gguf
  python -m bench.planner_eval --category app_control,browser    # run specific categories
  python -m bench.planner_eval --compare models/A.gguf models/B.gguf

Nothing is executed on the desktop: plans are only parsed and scored. Each expected step is (allowed actions,
required argument substrings); a plan passes 'strict' if the expected steps appear in order with those arguments.
The set is deliberately phrased the way the router would MISS, and its example values differ from the prompt pool.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import psutil

from cua.actions.registry import default_registry
from cua.config import Config
from cua.models.local_llm import LLMPlanner, select_tools
from cua.models.runtime import LlamaServerRuntime

from bench.planner_corpus import CASES, CATEGORIES


def score(plan, expected) -> dict:
    if plan is None:
        return {"valid": False, "first": False, "actions": False, "strict": False, "extra_steps": 0}
    acts = [s.action for s in plan.steps]
    first = acts[0] in expected[0][0]
    i = j = 0
    for s in plan.steps:
        if i < len(expected) and s.action in expected[i][0]:
            i += 1
        if j < len(expected) and s.action in expected[j][0] and all(
                sub.lower() in json.dumps(s.args).lower() for sub in expected[j][1]):
            j += 1
    extra = max(0, len(plan.steps) - len(expected))
    return {"valid": True, "first": first, "actions": i == len(expected), "strict": j == len(expected),
            "extra_steps": extra}


def pct(rows, key):
    return round(100 * sum(r[key] for r in rows) / max(1, len(rows)))


def run_variant(variant: str, cases, cfg: Config, rt, reg, verbose: bool) -> dict:
    cfg.prompt_variant = variant
    pl = LLMPlanner(rt, reg, cfg)
    all_rows, by_cat = [], defaultdict(list)
    lat, toks, ptoks, ctok, extras, tool_counts = [], [], [], [], [], []

    for cmd, exp, cat in cases:
        ctx = {}
        tools = select_tools(cmd, reg)
        tool_counts.append(len(tools.names()))
        t = time.perf_counter()
        plan = pl.plan(cmd, ctx)
        ms = (time.perf_counter() - t) * 1000
        r = score(plan, exp)
        r["category"] = cat
        all_rows.append(r)
        by_cat[cat].append(r)
        lat.append(ms)
        extras.append(r["extra_steps"])
        for c in ctx.get("planner_calls", []):
            toks.append(c["tok_s"])
            ptoks.append(c["prompt_tokens"])
            ctok.append(c["completion_tokens"])
        if verbose or not r["strict"]:
            got = [(s.action, s.args) for s in plan.steps] if plan else ctx.get("planner_error")
            print(f"  [{variant}] {'OK ' if r['strict'] else 'MISS'} [{cat}] {cmd!r} -> {got}")

    stt = rt.status()
    proc = psutil.Process()
    summary = {
        "valid_pct": pct(all_rows, "valid"),
        "first_action_pct": pct(all_rows, "first"),
        "actions_pct": pct(all_rows, "actions"),
        "strict_pct": pct(all_rows, "strict"),
        "avg_extra_steps": round(st.mean(extras), 2) if extras else 0,
        "latency_ms_median": round(st.median(lat)),
        "latency_ms_mean": round(st.mean(lat)),
        "latency_ms_p95": round(sorted(lat)[max(0, int(len(lat) * .95) - 1)]),
        "tok_s": round(st.mean(toks), 1) if toks else 0,
        "prompt_tokens_avg": round(st.mean(ptoks)) if ptoks else 0,
        "completion_tokens_avg": round(st.mean(ctok)) if ctok else 0,
        "avg_tools_sent": round(st.mean(tool_counts), 1) if tool_counts else 0,
        "server_rss_mb": stt["rss_mb"],
        "agent_rss_mb": round(proc.memory_info().rss / 2**20),
        "load_ms": stt["load_ms"],
        "sys_avail_mb": round(psutil.virtual_memory().available / 2**20),
        "n": len(all_rows),
    }
    cat_summary = {}
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        cat_summary[cat] = {
            "n": len(rows),
            "strict_pct": pct(rows, "strict"),
            "actions_pct": pct(rows, "actions"),
            "valid_pct": pct(rows, "valid"),
        }
    summary["by_category"] = cat_summary
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="dynamic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default=None, help="comma-separated categories to run")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--retries", type=int, default=0, help="llm_retries (0 = single-shot quality)")
    ap.add_argument("--compare", nargs="*", help="model paths to compare")
    a = ap.parse_args()

    cases = [(cmd, exp, cat) for cmd, exp, cat in CASES]
    if a.category:
        cats = set(c.strip() for c in a.category.split(","))
        cases = [(cmd, exp, cat) for cmd, exp, cat in cases if cat in cats]
    if a.limit:
        cases = cases[:a.limit]

    models = a.compare if a.compare else [a.model]
    reg = default_registry()

    for model_path in models:
        cfg = Config(model_path=model_path, llm_retries=a.retries)
        rt = LlamaServerRuntime(cfg)
        model = cfg.resolve_model()
        if model is None:
            print("no model found; skipping")
            continue
        print(f"\n{'='*70}")
        print(f"model: {model.name} ({model.stat().st_size / 2**20:.0f} MB)")
        print(f"free RAM: {psutil.virtual_memory().available / 2**20:.0f} MB")
        print(f"cases: {len(cases)} across {len({c for _, _, c in cases})} categories")
        print(f"categories: {', '.join(sorted({c for _, _, c in cases}))}")
        print(f"{'='*70}\n")

        out = {
            "model": model.name,
            "size_mb": round(model.stat().st_size / 2**20),
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories_available": CATEGORIES,
            "variants": {},
        }

        for variant in a.variants.split(","):
            print(f"--- variant: {variant} ---")
            summary = run_variant(variant, cases, cfg, rt, reg, a.verbose)
            out["variants"][variant] = summary

            print(f"\n  Overall: valid={summary['valid_pct']}% first={summary['first_action_pct']}% "
                  f"actions={summary['actions_pct']}% strict={summary['strict_pct']}%")
            print(f"  Latency: median={summary['latency_ms_median']}ms mean={summary['latency_ms_mean']}ms "
                  f"p95={summary['latency_ms_p95']}ms")
            print(f"  Throughput: {summary['tok_s']} tok/s | prompt={summary['prompt_tokens_avg']} "
                  f"completion={summary['completion_tokens_avg']} tokens")
            print(f"  Tools sent: avg {summary['avg_tools_sent']} | Extra steps: avg {summary['avg_extra_steps']}")
            print(f"  RAM: server={summary['server_rss_mb']}MB agent={summary['agent_rss_mb']}MB "
                  f"sys_avail={summary['sys_avail_mb']}MB")
            print(f"\n  By category:")
            for cat, cs in summary["by_category"].items():
                bar = "#" * (cs["strict_pct"] // 5) + "." * (20 - cs["strict_pct"] // 5)
                print(f"    {cat:<20s} [{bar}] {cs['strict_pct']:3d}% strict  ({cs['n']} cases)")
            print()

        rt.unload()
        rd = Path("bench/results")
        rd.mkdir(parents=True, exist_ok=True)
        f = rd / f"planner_eval_{model.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        f.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
