"""Latency spans, events, and structured JSONL task records (incl. CPU / RAM)."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import psutil


class Logger:
    """Per-task collector. Spans accumulate wall-clock per stage; events are discrete facts (retry, replan, ...)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        self.stage_ms: dict[str, float] = defaultdict(float)

    @contextmanager
    def span(self, stage: str, **meta):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - t0) * 1000, **meta)

    def add(self, stage: str, ms: float, **meta):
        self.stage_ms[stage] += ms
        self.records.append({"stage": stage, "ms": round(ms, 2), **meta})

    def event(self, name: str, **meta):
        self.records.append({"event": name, "t": round(time.time(), 3), **meta})

    def flush(self, task_record: dict | None = None):
        """Append discrete events plus one structured task record. Per-span noise stays in memory."""
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for r in self.records:
                if "event" in r:
                    f.write(json.dumps(r, default=str) + "\n")
            if task_record:
                f.write(json.dumps({"event": "task", **task_record}, default=str) + "\n")


class ResourceProbe:
    """CPU (machine-normalised %) and RSS of the agent plus its children (the llama-server), over a task."""

    def __init__(self):
        self.p = psutil.Process()
        self.ncpu = psutil.cpu_count() or 1

    def _cpu_s(self) -> float:
        total = 0.0
        for proc in [self.p, *self.p.children(recursive=True)]:
            try:
                t = proc.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total

    def _rss_mb(self) -> float:
        total = 0
        for proc in [self.p, *self.p.children(recursive=True)]:
            try:
                total += proc.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**20

    def start(self) -> "ResourceProbe":
        self.t0, self.c0 = time.perf_counter(), self._cpu_s()
        return self

    def stop(self) -> dict:
        wall = max(time.perf_counter() - self.t0, 1e-6)
        return {"cpu_pct": round((self._cpu_s() - self.c0) / wall / self.ncpu * 100, 1),
                "rss_mb": round(self._rss_mb()),
                "avail_mb": round(psutil.virtual_memory().available / 2**20)}
