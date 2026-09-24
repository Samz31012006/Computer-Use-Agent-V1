"""API server tests. No real agent, microphone, network exposure, or model needed.

AgentAPI is a thin transport wrapper around AgentService (cua/agent/service.py) -- see test_agent_service.py for
the shared-service behaviour itself (single agent, single worker, cancellation, shared events). These tests
cover the HTTP/WS-facing surface: task submission/lookup at the AgentAPI level, plus the auth system (login,
rate limiting, sessions, logout) and request validation exercised through a REAL ephemeral loopback HTTP server
(no real network exposure -- 127.0.0.1, an OS-assigned port, torn down at the end of each test).
"""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
from http.server import HTTPServer
from pathlib import Path

from cua.agent.service import AgentService
from cua.safety.kill_switch import KillSwitch
from cua.server.api import AgentAPI, TaskRecord, _ws_sync_handler, create_handler
from cua.server.auth import AuthStore


def safe_kill() -> KillSwitch:
    ks = KillSwitch()
    ks.hard_exit = False
    return ks


class FakeAgent:
    """Minimal stand-in for Agent: honors the same contract AgentService relies on (call on_event around
    run()), same as the real Agent does in cua/agent/agent.py."""

    def __init__(self, ok=True):
        self._ok = ok
        self.on_event = None
        self.calls = []

    def run(self, text, context=None):
        self.calls.append(text)
        from cua.types import TaskResult
        if self.on_event:
            self.on_event("task_start", text=text)
        result = TaskResult(text=text, ok=self._ok, plan=None, steps=[], reason="ok" if self._ok else "failed",
                            stage_ms={}, total_ms=100, planner="router", router_hit=True)
        if self.on_event:
            self.on_event("task_end", result=result)
        return result


class FakeWebSocket:
    """Stands in for a websockets.sync connection: recv() pops queued incoming messages, send()/ping() record
    what would go out."""

    def __init__(self, incoming: list[str]):
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.closed = False

    def recv(self, timeout=None):
        if not self._incoming:
            raise TimeoutError("no more incoming messages")
        return self._incoming.pop(0)

    def send(self, msg):
        self.sent.append(msg)

    def ping(self):
        if self.closed:
            raise ConnectionError("closed")


def make_auth() -> AuthStore:
    path = Path(tempfile.mktemp(suffix=".json"))
    return AuthStore(path=path)


def make_api(ok=True) -> tuple[AgentAPI, FakeAgent, AuthStore]:
    agent = FakeAgent(ok=ok)
    service = AgentService(agent=agent, kill=safe_kill())
    auth = make_auth()
    return AgentAPI(service, auth=auth), agent, auth


def wait_for(pred, timeout=2.0, interval=0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


class LiveServer:
    """A real HTTPServer bound to 127.0.0.1:0 (OS-assigned port), running create_handler(api). Used to exercise
    the actual request parsing / auth / routing code, not a reimplementation of it."""

    def __init__(self, api: AgentAPI):
        self.httpd = HTTPServer(("127.0.0.1", 0), create_handler(api))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers=headers or {})
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw
        return resp.status, parsed

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---- TaskRecord -----------------------------------------------------------
def test_task_record_lifecycle():
    rec = TaskRecord("open chrome")
    assert rec.status == "received"
    assert rec.id.startswith("task_") and len(rec.id) == 13
    d = rec.to_dict()
    assert d["text"] == "open chrome" and d["status"] == "received"


def test_task_record_with_result():
    from cua.types import TaskResult, StepResult, Step
    rec = TaskRecord("open chrome")
    rec.result = TaskResult(text="open chrome", ok=True, plan=None,
                            steps=[StepResult(Step("open_app", {"name": "chrome"}), True, "skills", 1, False, "ok", 50)],
                            reason="ok", stage_ms={"plan": 1}, total_ms=60,
                            planner="router", router_hit=True)
    rec.status = "completed"
    d = rec.to_dict()
    assert d["result"]["ok"] is True
    assert len(d["result"]["steps"]) == 1
    assert d["result"]["steps"][0]["action"] == "open_app"


# ---- AgentAPI core (object-level, no HTTP) ---------------------------------
def test_submit_and_get_task():
    api, _, _ = make_api()
    rec = api.submit_task("open notepad")
    assert rec.status == "received"
    assert api.get_task(rec.id) is rec
    assert api.get_task("nonexistent") is None


def test_task_executes():
    api, agent, _ = make_api()
    rec = api.submit_task("open chrome")
    assert wait_for(lambda: "open chrome" in agent.calls)
    assert wait_for(lambda: rec.status in ("completed", "executing"))


def test_failed_task():
    api, agent, _ = make_api(ok=False)
    rec = api.submit_task("do impossible thing")
    assert wait_for(lambda: rec.status == "failed")


def test_events_broadcast():
    api, _, _ = make_api()
    q = api.subscribe()
    api.submit_task("hello")
    time.sleep(0.1)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert any(e["type"] == "received" for e in events)
    api.unsubscribe(q)


def test_unsubscribe_removes_queue():
    api, _, _ = make_api()
    q = api.subscribe()
    assert q in api.service._subscribers
    api.unsubscribe(q)
    assert q not in api.service._subscribers


def test_multiple_tasks_execute_in_order():
    api, agent, _ = make_api()
    api.submit_task("first")
    api.submit_task("second")
    assert wait_for(lambda: agent.calls == ["first", "second"], timeout=3)


def test_cancel_unknown_task_returns_false():
    api, _, _ = make_api()
    assert api.cancel_task("task_doesnotexist") is False


def test_phone_task_uses_same_agent():
    api, agent, _ = make_api()
    rec = api.submit_task("open chrome")
    assert wait_for(lambda: agent.calls == ["open chrome"])
    assert rec.source == "phone"
    assert api.agent is agent


# ---- auth: login / rate limiting / sessions / logout -----------------------
def test_login_success():
    api, _, auth = make_api()
    session = auth.login(auth.generated_pin, "127.0.0.1")
    assert session is not None
    assert auth.sessions.validate(session)


def test_login_failure_does_not_create_a_session():
    api, _, auth = make_api()
    session = auth.login("0000000-wrong", "127.0.0.1")
    assert session is None
    assert len(auth.sessions) == 0


def test_login_rate_limiting_locks_out_after_repeated_failures():
    api, _, auth = make_api()
    for _ in range(5):
        assert auth.login("wrong", "9.9.9.9") is None
    assert auth.throttle.allowed("9.9.9.9") is False
    assert auth.login(auth.generated_pin, "9.9.9.9") is None      # even the RIGHT pin is locked out now
    assert auth.throttle.allowed("1.1.1.1") is True                # a different source is unaffected


def test_session_expires():
    api, _, auth = make_api()
    auth.sessions.ttl_s = 0.05
    session = auth.sessions.create()
    assert auth.sessions.validate(session)
    time.sleep(0.1)
    assert auth.sessions.validate(session) is False


def test_logout_revokes_session():
    api, _, auth = make_api()
    session = auth.login(auth.generated_pin, "127.0.0.1")
    assert auth.sessions.validate(session)
    auth.sessions.revoke(session)
    assert auth.sessions.validate(session) is False


def test_password_is_never_stored_in_plaintext():
    api, _, auth = make_api()
    raw = auth.path.read_text(encoding="utf-8")
    assert auth.generated_pin not in raw


def test_changing_password_revokes_existing_sessions():
    api, _, auth = make_api()
    session = auth.login(auth.generated_pin, "127.0.0.1")
    assert auth.sessions.validate(session)
    auth.set_password("a-new-password")
    assert auth.sessions.validate(session) is False
    assert auth.login("a-new-password", "127.0.0.1") is not None


# ---- HTTP transport: real loopback requests --------------------------------
def test_unauthenticated_http_request_is_rejected():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        status, body = srv.request("GET", "/status")
        assert status == 401 and body["error"] == "unauthorized"
    finally:
        srv.close()


def test_authenticated_http_request_succeeds():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        session = auth.login(auth.generated_pin, "127.0.0.1")
        status, body = srv.request("GET", "/status", headers={"Authorization": f"Bearer {session}"})
        assert status == 200 and "busy" in body
    finally:
        srv.close()


def test_credentials_are_rejected_from_query_string():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        session = auth.login(auth.generated_pin, "127.0.0.1")
        status, body = srv.request("GET", f"/status?token={session}&session={session}")
        assert status == 401                     # only the Authorization/X-Session header is honored
    finally:
        srv.close()


def test_login_then_submit_task_over_http():
    api, agent, auth = make_api()
    srv = LiveServer(api)
    try:
        status, body = srv.request("POST", "/auth/login", {"password": auth.generated_pin})
        assert status == 200 and "session" in body
        session = body["session"]

        status, body = srv.request("POST", "/task", {"text": "open chrome"},
                                   headers={"X-Session": session})
        assert status == 201 and body["id"].startswith("task_")
    finally:
        srv.close()


def test_login_wrong_password_over_http():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        status, body = srv.request("POST", "/auth/login", {"password": "not-it"})
        assert status == 401 and body["error"] == "invalid credentials"
    finally:
        srv.close()


def test_login_rate_limited_over_http():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        for _ in range(5):
            srv.request("POST", "/auth/login", {"password": "nope"})
        status, body = srv.request("POST", "/auth/login", {"password": "nope"})
        assert status == 429
    finally:
        srv.close()


def test_logout_over_http_invalidates_session():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        session = auth.login(auth.generated_pin, "127.0.0.1")
        status, _ = srv.request("POST", "/auth/logout", {}, headers={"X-Session": session})
        assert status == 200
        status, _ = srv.request("GET", "/status", headers={"X-Session": session})
        assert status == 401
    finally:
        srv.close()


def test_malformed_json_body_is_rejected_not_crashed():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        session = auth.login(auth.generated_pin, "127.0.0.1")
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
        conn.request("POST", "/task", body=b"{not valid json", headers={"X-Session": session})
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()
    finally:
        srv.close()


def test_non_string_task_text_is_rejected():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        session = auth.login(auth.generated_pin, "127.0.0.1")
        status, body = srv.request("POST", "/task", {"text": 12345}, headers={"X-Session": session})
        assert status == 400
    finally:
        srv.close()


def test_oversized_task_text_is_rejected():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        session = auth.login(auth.generated_pin, "127.0.0.1")
        status, body = srv.request("POST", "/task", {"text": "x" * 5000}, headers={"X-Session": session})
        assert status == 400
    finally:
        srv.close()


def test_static_page_is_public_but_status_is_not():
    api, _, auth = make_api()
    srv = LiveServer(api)
    try:
        status, _ = srv.request("GET", "/")
        assert status == 200
        status, _ = srv.request("GET", "/status")
        assert status == 401
    finally:
        srv.close()


# ---- WebSocket auth ----------------------------------------------------------
def test_unauthenticated_websocket_is_rejected():
    api, _, auth = make_api()
    ws = FakeWebSocket([json.dumps({"session": "not-a-real-session"})])
    _ws_sync_handler(api, ws)
    assert any("unauthorized" in m for m in ws.sent)
    assert len(api.service._subscribers) == 0


def test_authenticated_websocket_receives_broadcast_events():
    api, _, auth = make_api()
    session = auth.login(auth.generated_pin, "127.0.0.1")
    ws = FakeWebSocket([json.dumps({"session": session})])

    th = threading.Thread(target=_ws_sync_handler, args=(api, ws), daemon=True)
    th.start()
    assert wait_for(lambda: len(api.service._subscribers) == 1, timeout=1)
    assert wait_for(lambda: any('"connected"' in m for m in ws.sent), timeout=1)

    q = api.service._subscribers[0]
    q.put({"type": "test_event", "id": "task_x"})
    assert wait_for(lambda: any("test_event" in m for m in ws.sent), timeout=1)
