"""Live benchmark: real commands, end to end, on this desktop.

  python -m bench.run --reps 3                    # router-only (LLM disabled)   -> "before"
  python -m bench.run --reps 3 --planner on       # router + local LLM planner   -> "after"
  python -m bench.run --only open_app,find_file

Measures per task: fast-path/planner latency, action latency, verification latency, total latency, CPU, RAM, success
and recovery. It opens and closes a throwaway text box, Calculator, Settings, Explorer windows and a Chrome tab; it
only closes windows it created and never touches your real Notepad. Files go to a temp dir, never your Documents.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil

from cua.agent.factory import build_agent
from cua.catalog import AppSpec, register, resolve_app
from cua.computer.driver import WindowsDriver
from cua.config import Config, load_config
from cua.perception.uia import UIAPerception
from cua.safety.kill_switch import KillSwitch
from cua.types import DriverError, TaskResult

from . import coverage

_PS1 = Path(__file__).with_name("benchpad.ps1").resolve()
register(AppSpec("benchpad", ("exec", ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy",
                                       "Bypass", "-File", str(_PS1)]), ("powershell.exe",), r"^CuaBenchPad"))
_closer = WindowsDriver()


# ---- instrumentation ---------------------------------------------------------------------
class Sampler:
    """100 ms samples of this process + children (CPU%, RSS) and system CPU% / available RAM."""

    def __init__(self):
        self.p = psutil.Process()
        self._stop = threading.Event()
        self.cpu, self.sys_cpu, self.rss, self.avail = [], [], [], []

    def _tree_rss(self):
        tot = 0
        for pr in [self.p, *self.p.children(recursive=True)]:
            try:
                tot += pr.memory_info().rss
            except psutil.Error:
                pass
        return tot

    def __enter__(self):
        self.p.cpu_percent(None); psutil.cpu_percent(None)
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(0.1):
            self.cpu.append(self.p.cpu_percent(None))
            self.sys_cpu.append(psutil.cpu_percent(None))
            self.rss.append(self._tree_rss())
            self.avail.append(psutil.virtual_memory().available)

    def __exit__(self, *a):
        self._stop.set(); self._t.join()

    def summary(self):
        f = lambda xs, fn: round(fn(xs), 1) if xs else 0.0
        return {"agent_cpu_mean_pct": f(self.cpu, st.mean), "sys_cpu_peak_pct": f(self.sys_cpu, max),
                "rss_peak_mb": f([r / 2**20 for r in self.rss], max), "sys_avail_min_mb": f([a / 2**20 for a in self.avail], min)}


class FlakyDriver(WindowsDriver):
    """Fault injection: the first N launches raise (exercises retry / replan on the real stack)."""

    def __init__(self, n=1, **kw):
        super().__init__(**kw); self.left = n

    def launch(self, app, params=None):
        if self.left > 0:
            self.left -= 1
            raise DriverError("injected launch failure")
        super().launch(app, params)


class DropCharDriver(WindowsDriver):
    """Fault injection: the first typed text loses its first character (the real failure seen in V0)."""

    def __init__(self, **kw):
        super().__init__(**kw); self.done = False

    def type_text(self, text):
        if not self.done:
            self.done, text = True, text[1:]
        super().type_text(text)


# ---- tasks ------------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    category: str
    command: str
    setup: str | None = None
    check: Callable | None = None            # (perception, tmp) -> bool, evaluated after the agent finishes
    inject: str | None = None                # "launch" | "drop"
    needs: tuple = ()                        # apps that must NOT already be open (we would touch the wrong window)
    confirm: bool = False                    # auto-approve destructive steps (closing a window this bench opened)
    llm: bool = False                        # only meaningful with the planner on (router is expected to miss)


def _uid() -> str:
    return "cua" + uuid.uuid4().hex[:8]


def build_tasks() -> list[Task]:
    calc_ok = lambda p, tmp: any(p.find_element(w.hwnd, "Display is 15", 1.0) for w in p.match(resolve_app("calculator")))
    return [
        Task("open_app", "open app", "open benchpad", needs=("benchpad",)),
        Task("close_app", "close app", "close benchpad", setup="open benchpad", needs=("benchpad",), confirm=True),
        Task("open_folder", "open folder", "open my downloads folder"),
        Task("find_file", "find file", "open my downloads folder and find the pdf i downloaded today"),
        Task("open_file", "open file", "open bench_open.txt with benchpad", needs=("benchpad",)),
        Task("type_text", "type text", "open benchpad and type {uid}", needs=("benchpad",)),
        Task("hotkey", "hotkey", "open benchpad and press ctrl+a", needs=("benchpad",)),
        Task("ui_click", "UI element click",
             "open calculator and click seven and click plus and click eight and click equals", needs=("calculator",),
             check=calc_ok),
        Task("browser_navigate", "browser navigation", "go to example.com"),
        Task("browser_search", "browser search", "Open Chrome and search for the latest SpaceX launch"),
        Task("navigate_settings", "navigate settings", "open bluetooth settings", needs=("settings",)),
        Task("multi_step", "multi-step", "open benchpad and type {uid} and press ctrl+a and press end", needs=("benchpad",)),
        Task("recover_launch", "recovery: failed launch", "open benchpad", inject="launch", needs=("benchpad",)),
        Task("recover_typing", "recovery: dropped keystroke", "open benchpad and type {uid}", inject="drop",
             needs=("benchpad",)),
        # --- router misses: only the LLM can plan these (run with --planner on)
        Task("llm_create_file", "LLM: create file", "make me a text file called bench_note.txt saying buy milk",
             check=lambda p, tmp: (tmp / "Documents" / "bench_note.txt").exists(), llm=True),
        Task("llm_open_app", "LLM: open app", "bring up the calculator", needs=("calculator",), llm=True),
        Task("llm_web", "LLM: web search", "what is the weather like in Paris", llm=True),
        Task("llm_folder", "LLM: open folder", "show me what's in my downloads", llm=True),
    ]


def new_windows(p, baseline):
    return [w for w in p.windows() if w.hwnd not in baseline]


def cleanup(p, baseline, tabs=()):
    """Close only windows created since baseline, of apps the bench may touch."""
    for w in new_windows(p, baseline):
        t, proc = w.title.lower(), w.proc.lower()
        ok = ((proc == "powershell.exe" and w.title.startswith("CuaBenchPad")) or t == "calculator" or t == "settings"
              or w.cls == "CabinetWClass" or (proc == "chrome.exe" and ("new tab" in t or any(m in t for m in tabs))))
        if ok:
            _closer.close(w.hwnd)
    time.sleep(0.2)


def close_our_tab(p, markers):
    """A search/navigation may open a tab in the user's existing window; close it only if it is in front and ours."""
    fg = p.foreground()
    if fg and fg.proc.lower() == "chrome.exe" and any(m in fg.title.lower() for m in markers):
        _closer.hotkey("ctrl+w")


_TAB_MARKERS = ("spacex", "example domain", "weather", "paris", "cats")


def make_agent(t: Task, mode: str, tmp: Path):
    cfg = load_config()
    cfg.planner_enabled = mode == "on"
    cfg.confirm_llm_plans = False        # the bench measures raw planner quality; the product asks first
    kw = {}
    if t.inject == "launch":
        kw["driver"] = FlakyDriver(1)
    elif t.inject == "drop":
        kw["driver"] = DropCharDriver()
    return build_agent(cfg, confirm=(lambda _: True) if t.confirm else (lambda _: False),
                       folders={"downloads": tmp / "Downloads", "documents": tmp / "Documents"}, log_dir=None, **kw)


def run_task(t: Task, reps: int, tmp: Path, mode: str) -> dict:
    p = UIAPerception()
    for key in t.needs:
        if p.match(resolve_app(key)):
            return {"task": t.name, "category": t.category, "skipped": f"{key} already open; skipped to avoid touching it"}
    rows = []
    for _ in range(reps):
        baseline = {w.hwnd for w in p.windows()}
        if t.setup:
            make_agent(Task("s", "s", t.setup, confirm=True), mode, tmp).run(t.setup)
        agent = make_agent(t, mode, tmp)
        cmd = t.command.format(uid=_uid())
        with Sampler() as s:
            res: TaskResult = agent.run(cmd)
        ok = res.ok and (t.check(agent.perception, tmp) if t.check else True)
        rows.append({"ok": ok, "reason": res.reason, "total_ms": res.total_ms, "stage_ms": res.stage_ms,
                     "recovered": res.recovered, "had_failure": res.had_failure, "planner": res.planner,
                     "router_hit": res.router_hit, "planner_calls": res.planner_calls, "replans": res.replans,
                     "retries": res.retries, "res": s.summary(), "fail_detail": [x.detail for x in res.steps if not x.ok]})
        close_our_tab(p, _TAB_MARKERS)
        cleanup(p, baseline, _TAB_MARKERS)
        if getattr(agent, "runtime", None) is not None:
            agent.runtime.unload() if mode == "off" else None
    return {"task": t.name, "category": t.category, "command": t.command, "rows": rows}


def kill_test() -> dict:
    """Real global hotkey while the agent types 4000 chars: measures hotkey -> abort latency."""
    p = UIAPerception()
    if p.match(resolve_app("benchpad")):
        return {"skipped": "benchpad already open"}
    baseline = {w.hwnd for w in p.windows()}
    ks = KillSwitch(); ks.hard_exit = False; ks.start()
    if ks.error:
        return {"error": ks.error}
    cfg = load_config(); cfg.planner_enabled = False
    agent = build_agent(cfg, kill=ks, log_dir=None)
    agent.run("open benchpad")
    done = {}

    def fire():
        time.sleep(0.6)
        done["t"] = time.perf_counter()
        agent.driver.hotkey("ctrl+alt+q")
    threading.Thread(target=fire).start()
    res = agent.run("type " + "x" * 4000 + " into benchpad")
    t_end = time.perf_counter()
    ks.stop(); cleanup(p, baseline)
    return {"aborted": res.reason.startswith("aborted"), "reason": res.reason,
            "hotkey_to_abort_ms": round((t_end - done["t"]) * 1000, 1) if "t" in done else None}


def summarise(r: dict) -> dict:
    if "rows" not in r:
        return r
    rows = r["rows"]
    tot = [x["total_ms"] for x in rows]
    stages: dict[str, list] = {}
    for x in rows:
        for k, v in x["stage_ms"].items():
            stages.setdefault(k, []).append(v)
    m = lambda k: round(st.mean(stages.get(k, [0])))
    with_fail = [x for x in rows if x["had_failure"]]
    calls = [c for x in rows for c in x["planner_calls"]]
    return {
        "task": r["task"], "category": r["category"], "n": len(rows),
        "success_rate": round(sum(x["ok"] for x in rows) / len(rows), 2),
        "recovery_rate": round(sum(x["recovered"] for x in with_fail) / len(with_fail), 2) if with_fail else None,
        "planner_used": sorted({x["planner"] for x in rows}),
        "total_ms_median": round(st.median(tot)), "total_ms_max": round(max(tot)),
        "planner_ms": m("plan") + m("replan"), "action_ms": m("select") + m("act") + m("settle"),
        "verify_ms": m("verify"), "perceive_ms": m("perceive"),
        "llm_tok_s": round(st.mean(c["tok_s"] for c in calls), 1) if calls else None,
        "llm_calls": len(calls),
        "agent_cpu_mean_pct": round(st.mean(x["res"]["agent_cpu_mean_pct"] for x in rows), 1),
        "sys_cpu_peak_pct": max(x["res"]["sys_cpu_peak_pct"] for x in rows),
        "rss_peak_mb": max(x["res"]["rss_peak_mb"] for x in rows),
        "sys_avail_min_mb": min(x["res"]["sys_avail_min_mb"] for x in rows),
        "failures": sorted({d for x in rows for d in x["fail_detail"]})[:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--planner", choices=["off", "on"], default="off")
    ap.add_argument("--no-kill-test", action="store_true")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="cua_bench_"))
    (tmp / "Downloads").mkdir(); (tmp / "Documents").mkdir()
    (tmp / "Downloads" / "cua_bench_report.pdf").write_bytes(b"%PDF-1.4 bench")
    (tmp / "Downloads" / "bench_open.txt").write_text("hello from the bench")
    (tmp / "Downloads" / "notes.txt").write_text("x")

    tasks = [t for t in build_tasks() if a.planner == "on" or not t.llm]
    if a.only:
        want = set(a.only.split(",")); tasks = [t for t in tasks if t.name in want]
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "planner": a.planner, "reps": a.reps,
           "system": {"ram_total_mb": round(psutil.virtual_memory().total / 2**20),
                      "ram_avail_at_start_mb": round(psutil.virtual_memory().available / 2**20)},
           "router_coverage": coverage.run(), "tasks": []}
    for t in tasks:
        print(f"== {t.name}: {t.command}", flush=True)
        s = summarise(run_task(t, a.reps, tmp, a.planner))
        out["tasks"].append(s)
        print("  ", {k: v for k, v in s.items() if k in ("success_rate", "recovery_rate", "total_ms_median", "planner_ms",
                                                            "action_ms", "verify_ms", "failures", "skipped")}, flush=True)
    if not a.no_kill_test:
        out["kill_test"] = kill_test()
        print("kill:", out["kill_test"])
    rd = Path("bench/results"); rd.mkdir(exist_ok=True, parents=True)
    f = rd / f"{time.strftime('%Y%m%d_%H%M%S')}_planner-{a.planner}.json"
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", f)


if __name__ == "__main__":
    main()
