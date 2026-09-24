"""Evaluate the LLM planner on the "dynamic computer use" corpus (bench/dynamic_corpus.py): simple, multi-step,
natural-language, omitted-context, misspelled, multi-tool, recovery, and browser-navigation commands.

  python -m bench.dynamic_eval                                  # default model
  python -m bench.dynamic_eval --model models/other.gguf
  python -m bench.dynamic_eval --compare models/A.gguf models/B.gguf

Nothing is executed on the desktop: plans are only parsed and scored, same convention as planner_eval.py (whose
scoring function this reuses).
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

from bench.dynamic_corpus import CASES, CATEGORIES, RECOVERY_CASES
from bench.planner_eval import pct, score


def run(cfg: Config, rt, reg, verbose: bool) -> dict:
    pl = LLMPlanner(rt, reg, cfg)
    all_rows, by_cat = [], defaultdict(list)
    lat, toks, tool_counts, failures = [], [], [], []

    for cmd, exp, cat in CASES:
        ctx = {}
        tool_counts.append(len(select_tools(cmd, reg).names()))
        t = time.perf_counter()
        plan = pl.plan(cmd, ctx)
        ms = (time.perf_counter() - t) * 1000
        r = score(plan, exp)
        r["category"] = cat
        all_rows.append(r)
        by_cat[cat].append(r)
        lat.append(ms)
        for c in ctx.get("planner_calls", []):
            toks.append(c["tok_s"])
        got = [(s.action, s.args) for s in plan.steps] if plan else ctx.get("planner_error")
        if not r["strict"]:
            failures.append({"cmd": cmd, "category": cat, "got": str(got)[:200]})
        if verbose or not r["strict"]:
            print(f"  {'OK ' if r['strict'] else 'MISS'} [{cat}] {cmd!r} -> {got}")

    # ---- recovery: replan() given a failed step + error, not plan() -----------------------
    rec_rows = []
    for cmd, failed, error, exp in RECOVERY_CASES:
        ctx = {}
        t = time.perf_counter()
        plan = pl.replan(cmd, ctx, failed, error)
        ms = (time.perf_counter() - t) * 1000
        r = score(plan, exp)
        r["category"] = "recovery"
        rec_rows.append(r)
        by_cat["recovery"].append(r)
        all_rows.append(r)
        lat.append(ms)
        got = [(s.action, s.args) for s in plan.steps] if plan else None
        if not r["strict"]:
            failures.append({"cmd": f"[replan] {cmd} (failed: {failed.action}, {error})", "category": "recovery",
                             "got": str(got)[:200]})
        if verbose or not r["strict"]:
            print(f"  {'OK ' if r['strict'] else 'MISS'} [recovery] {cmd!r} (after {failed.action} failed: "
                 f"{error}) -> {got}")

    stt = rt.status()
    proc = psutil.Process()
    summary = {
        "valid_pct": pct(all_rows, "valid"), "first_action_pct": pct(all_rows, "first"),
        "actions_pct": pct(all_rows, "actions"), "strict_pct": pct(all_rows, "strict"),
        "latency_ms_median": round(st.median(lat)) if lat else 0,
        "latency_ms_mean": round(st.mean(lat)) if lat else 0,
        "tok_s": round(st.mean(toks), 1) if toks else 0,
        "avg_tools_sent": round(st.mean(tool_counts), 1) if tool_counts else 0,
        "server_rss_mb": stt["rss_mb"], "agent_rss_mb": round(proc.memory_info().rss / 2**20),
        "load_ms": stt["load_ms"], "sys_avail_mb": round(psutil.virtual_memory().available / 2**20),
        "n": len(all_rows), "n_failures": len(failures), "failures": failures,
    }
    summary["by_category"] = {cat: {"n": len(rows), "strict_pct": pct(rows, "strict"), "actions_pct": pct(rows, "actions"),
                                    "valid_pct": pct(rows, "valid")} for cat, rows in sorted(by_cat.items())}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--compare", nargs="*", help="model paths to compare")
    ap.add_argument("--min-free-ram", type=int, default=None, help="override Config.min_free_ram_mb (diagnostic use)")
    a = ap.parse_args()

    models = a.compare if a.compare else [a.model]
    reg = default_registry()

    for model_path in models:
        cfg = Config(model_path=model_path)
        if a.min_free_ram is not None:
            cfg.min_free_ram_mb = a.min_free_ram
        rt = LlamaServerRuntime(cfg)
        model = cfg.resolve_model()
        if model is None:
            print(f"model not found: {model_path!r}; skipping")
            continue
        print(f"\n{'='*70}\nmodel: {model.name} ({model.stat().st_size / 2**20:.0f} MB)")
        print(f"free RAM: {psutil.virtual_memory().available / 2**20:.0f} MB")
        print(f"cases: {len(CASES) + len(RECOVERY_CASES)} across {len(CATEGORIES) + 1} categories "
             f"({', '.join(CATEGORIES)}, recovery)\n{'='*70}\n")

        summary = run(cfg, rt, reg, a.verbose)
        print(f"\n  Overall: valid={summary['valid_pct']}% first={summary['first_action_pct']}% "
             f"actions={summary['actions_pct']}% strict={summary['strict_pct']}%  ({summary['n_failures']} misses)")
        print(f"  Latency: median={summary['latency_ms_median']}ms mean={summary['latency_ms_mean']}ms")
        print(f"  Throughput: {summary['tok_s']} tok/s | Tools sent: avg {summary['avg_tools_sent']}")
        print(f"  RAM: server={summary['server_rss_mb']}MB agent={summary['agent_rss_mb']}MB "
             f"sys_avail={summary['sys_avail_mb']}MB | load={summary['load_ms']}ms")
        print(f"\n  By category:")
        for cat, cs in summary["by_category"].items():
            bar = "#" * (cs["strict_pct"] // 5) + "." * (20 - cs["strict_pct"] // 5)
            print(f"    {cat:<20s} [{bar}] {cs['strict_pct']:3d}% strict  ({cs['n']} cases)")

        rt.unload()
        rd = Path("bench/results")
        rd.mkdir(parents=True, exist_ok=True)
        out = {"model": model.name, "size_mb": round(model.stat().st_size / 2**20),
              "when": time.strftime("%Y-%m-%d %H:%M:%S"), **summary}
        f = rd / f"dynamic_eval_{model.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        f.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
