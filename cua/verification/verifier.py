"""PollingVerifier: poll a predicate with backoff until it holds or the step's timeout expires.
Backoff (40 ms -> 200 ms) keeps CPU low during long waits such as browser start-up."""
from __future__ import annotations

import time
from typing import Protocol

from cua.types import VerifyResult
from cua.verification.predicates import PREDICATES


class Verifier(Protocol):
    def check(self, spec: dict | None, ctx) -> VerifyResult: ...


class PollingVerifier:
    def __init__(self, default_timeout: float = 5.0, poll: float = 0.04, max_poll: float = 0.2):
        self.default_timeout, self.poll, self.max_poll = default_timeout, poll, max_poll

    def check(self, spec, ctx) -> VerifyResult:
        if not spec:
            return VerifyResult(True, "no check")
        fn = PREDICATES.get(spec["kind"])
        if fn is None:
            return VerifyResult(False, f"unknown verify kind {spec['kind']}")
        timeout = float(spec.get("timeout", self.default_timeout))
        t0 = time.perf_counter()
        delay, detail = self.poll, ""
        while True:
            ctx.check_kill()
            try:
                ok, detail = fn(spec, ctx)
            except Exception as e:          # UIA/COM hiccups (window closing mid-query) mean "not yet", not "crash"
                ok, detail = False, f"{type(e).__name__}: {e}"
            waited = (time.perf_counter() - t0) * 1000
            if ok:
                return VerifyResult(True, detail, waited)
            if waited >= timeout * 1000:
                return VerifyResult(False, f"timeout after {timeout}s: {detail}", waited)
            time.sleep(delay)
            delay = min(self.max_poll, delay * 1.5)
