"""Local model runtime: llama.cpp's `llama-server` as a lazily started, idle-unloaded child process.

Why a server process rather than a Python binding: no compiler needed, the model's RAM is fully released on unload
(process exit), and a crash or OOM in the model can never take the agent down. Communication is plain HTTP on
localhost via the standard library, so this adds no Python dependencies.
"""
from __future__ import annotations

import base64
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import psutil

from cua.config import Config
from cua.types import Aborted


class ModelUnavailable(RuntimeError):
    """No usable model right now (not installed, not enough RAM, failed to start). The agent falls back to the router."""


@dataclass
class GenResult:
    text: str
    total_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_ms: float = 0.0
    gen_ms: float = 0.0

    @property
    def tok_s(self) -> float:
        return self.completion_tokens / (self.gen_ms / 1000) if self.gen_ms else 0.0


class Runtime(Protocol):
    def generate(self, prompt: str, schema: dict | None = None, max_tokens: int = 200,
                 abort: Callable[[], bool] | None = None) -> GenResult: ...
    def unload(self) -> None: ...
    def status(self) -> dict: ...


class LlamaServerRuntime:
    """One llama-server child process. Used both for the text planner (default) and, with `model_path`/
    `mmproj_path` overrides, for the separate VLM grounding runtime (cua/perception/vlm.py) -- same lazy-load,
    idle-unload and RAM-guard machinery, so nothing about process lifecycle is duplicated for the second model."""

    def __init__(self, cfg: Config, model_path: Path | None = None, mmproj_path: Path | None = None,
                 ctx_size: int | None = None, idle_unload_s: float | None = None, log_name: str = "llama-server.log"):
        self.cfg = cfg
        self._model_override = model_path
        self._mmproj_path = mmproj_path
        self._ctx_size = cfg.ctx_size if ctx_size is None else ctx_size
        self._idle_unload_s = cfg.idle_unload_s if idle_unload_s is None else idle_unload_s
        self._log_name = log_name
        self._proc: subprocess.Popen | None = None
        self._port = 0
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._busy = False
        self.load_ms = 0.0
        self._reaper: threading.Thread | None = None

    def _resolve_model(self) -> Path | None:
        return self._model_override or self.cfg.resolve_model()

    # ---- availability --------------------------------------------------------
    def unavailable_reason(self) -> str | None:
        if self.cfg.resolve_server() is None:
            return "llama-server not installed"
        model = self._resolve_model()
        if model is None:
            return "no .gguf model found"
        if self._mmproj_path is not None and not self._mmproj_path.is_file():
            return "mmproj file missing"
        if self._proc is None:      # only check memory when we would have to load
            free_mb = psutil.virtual_memory().available / 2**20
            if not self.cfg.model_fits(model, free_mb):
                need_mb = self.cfg.model_ram_need_mb(model) + self.cfg.min_free_ram_mb
                return (f"not enough free RAM ({free_mb:.0f} MB free, need ~{need_mb:.0f} MB for "
                        f"{model.name}) -- close some apps and try again")
        return None

    # ---- lifecycle -----------------------------------------------------------
    def _ensure_loaded(self):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            reason = self.unavailable_reason()
            if reason:
                raise ModelUnavailable(reason)
            server, model = self.cfg.resolve_server(), self._resolve_model()
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self._port = s.getsockname()[1]
            logf = open(Path(self.cfg.log_dir) / self._log_name, "ab") if Path(self.cfg.log_dir).is_dir() else subprocess.DEVNULL
            t0 = time.perf_counter()
            args = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(self._port),
                    "-c", str(self._ctx_size), "-t", str(self.cfg.threads), "-np", "1"]
            if self._mmproj_path:
                args += ["--mmproj", str(self._mmproj_path)]
            self._proc = subprocess.Popen(args, stdout=logf, stderr=logf, creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.time() + 90
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    self._proc = None
                    raise ModelUnavailable("llama-server exited during start-up (see logs/llama-server.log)")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self._port}/health", timeout=1) as r:
                        if r.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(0.15)
            else:
                self._kill_proc()
                raise ModelUnavailable("llama-server did not become ready in 90 s")
            self.load_ms = (time.perf_counter() - t0) * 1000
            self._start_reaper()

    def _kill_proc(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def unload(self) -> None:
        with self._lock:
            self._kill_proc()

    def _start_reaper(self):
        if self._reaper and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(target=self._reap, daemon=True, name="llm-idle-reaper")
        self._reaper.start()

    def _reap(self):
        while True:
            time.sleep(2)
            with self._lock:
                if self._proc is None:
                    return
                if not self._busy and time.time() - self._last_used > self._idle_unload_s:
                    self._kill_proc()
                    return

    def status(self) -> dict:
        loaded = self._proc is not None and self._proc.poll() is None
        rss = 0.0
        if loaded:
            try:
                rss = psutil.Process(self._proc.pid).memory_info().rss / 2**20
            except psutil.Error:
                pass
        return {"loaded": loaded, "rss_mb": round(rss), "load_ms": round(self.load_ms), "reason": self.unavailable_reason()}

    # ---- generation ----------------------------------------------------------
    def _post(self, path: str, body: dict, timeout: float, abort: Callable[[], bool] | None,
             abort_msg: str) -> dict:
        """POST body to path on the loaded server, pollable so `abort` can interrupt a decode (the only way
        to stop one is killing the server -- there's no cancel endpoint)."""
        req = urllib.request.Request(f"http://127.0.0.1:{self._port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        box: dict = {}

        def call():
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    box["data"] = json.loads(r.read())
            except Exception as e:
                box["err"] = e

        self._busy = True
        th = threading.Thread(target=call, daemon=True)
        th.start()
        try:
            while th.is_alive():
                th.join(0.05)
                if abort and abort():
                    self.unload()
                    raise Aborted(abort_msg)
        finally:
            self._busy, self._last_used = False, time.time()
        if "err" in box:
            raise ModelUnavailable(f"generation failed: {box['err']}")
        return box["data"]

    def generate(self, prompt: str, schema: dict | None = None, max_tokens: int = 200,
                 abort: Callable[[], bool] | None = None) -> GenResult:
        self._ensure_loaded()
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0, "cache_prompt": True, "stream": False}
        if schema:
            body["json_schema"] = schema
        t0 = time.perf_counter()
        d = self._post("/completion", body, 120, abort, "aborted during planning")
        tm = d.get("timings", {})
        return GenResult(d.get("content", ""), (time.perf_counter() - t0) * 1000, d.get("tokens_evaluated", 0),
                         d.get("tokens_predicted", 0), tm.get("prompt_ms", 0.0), tm.get("predicted_ms", 0.0))

    def generate_vision(self, prompt: str, image_png: bytes, max_tokens: int = 60,
                        abort: Callable[[], bool] | None = None) -> GenResult:
        """Like generate(), but for a model loaded with --mmproj: sends one image (PNG bytes) plus a text
        prompt to the OpenAI-compatible chat endpoint. Only meaningful on a runtime built with `mmproj_path`."""
        self._ensure_loaded()
        b64 = base64.b64encode(image_png).decode()
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt}]}],
               "max_tokens": max_tokens, "temperature": 0}
        t0 = time.perf_counter()
        d = self._post("/v1/chat/completions", body, 60, abort, "aborted during visual grounding")
        text = d["choices"][0]["message"]["content"]
        usage = d.get("usage", {})
        return GenResult(text, (time.perf_counter() - t0) * 1000, usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0))
