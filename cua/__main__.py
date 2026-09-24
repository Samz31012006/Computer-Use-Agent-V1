"""python -m cua "open chrome and search for the latest spacex launch"     (or no args for a REPL)
   --dry-run  print the plan only, touch nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from cua.agent.factory import build_agent
from cua.agent.router import RouterPlanner
from cua.safety.kill_switch import KillSwitch
from cua.safety.policy import cli_confirm


def _show(res):
    print(f"  {'OK' if res.ok else 'FAIL'} ({res.reason}) total={res.total_ms:.0f} ms  stages={res.stage_ms}")
    for s in res.steps:
        print(f"   - {s.step.action} {s.step.args} -> {'ok' if s.ok else 'FAIL'} [{s.engine} x{s.attempts}] {s.detail}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cua")
    ap.add_argument("command", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-planner", action="store_true", help="deterministic router only (no model)")
    ap.add_argument("--no-kill", action="store_true", help="do not register the emergency hotkey")
    args = ap.parse_args(argv)

    if args.dry_run:
        for text in [" ".join(args.command)] if args.command else iter(lambda: input("plan> "), "exit"):
            plan = RouterPlanner().plan(text)
            print(json.dumps(asdict(plan), indent=1) if plan else "  (router miss -> would need an LLM planner)")
        return 0

    kill = None
    if not args.no_kill:
        kill = KillSwitch().start()
        print(f"Emergency stop: {kill.hotkey.upper()}" if not kill.error else f"WARNING: {kill.error}")
    from cua.config import load_config
    cfg = load_config()
    cfg.planner_enabled = cfg.planner_enabled and not args.no_planner
    agent = build_agent(cfg, kill=kill, confirm=cli_confirm)
    if args.command:
        res = agent.run(" ".join(args.command))
        _show(res)
        return 0 if res.ok else 1
    print("Type a command, or 'exit'.")
    while True:
        try:
            text = input("cua> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("exit", "quit"):
            break
        if text:
            _show(agent.run(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
