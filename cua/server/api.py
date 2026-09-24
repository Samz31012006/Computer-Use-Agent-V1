"""Local HTTPS + WebSocket API server for the agent.

This is a thin transport layer over `AgentService` (cua/agent/service.py), which is the one place task
execution, planning, the model runtime, and events actually live. The phone and the laptop UI both talk to the
same AgentService instance, so there is exactly one Agent and one model runtime whichever interfaces are active.

Auth: a persisted, hashed PIN exchanged for a short-lived session token (see cua/server/auth.py) -- never a
password or the PIN itself on any request after login, never in a URL, never logged.

Routes:
  POST /auth/login        {"password": "123456"}  -> {"session": "..."}  (rate-limited)
  POST /auth/logout       (session header)         -> {"ok": true}
  POST /task               {"text": "open chrome"} -> {"id": "...", "status": "received"}
  POST /task/{id}/cancel   -> {"ok": bool}
  GET  /task/{id}          -> full TaskResult JSON
  GET  /status             -> {"busy": bool, "voice": bool}
  GET  /voice/start        -> start listening (laptop voice)
  GET  /voice/stop         -> stop listening
  WebSocket /events        -> first message {"session": "..."}, then real-time task events
  GET  /                   -> phone PWA HTML (public: no secrets live in the page shell itself)

Everything except /, /static/*, /manifest.webmanifest, and /auth/login requires a valid session, presented as
`Authorization: Bearer <session>` or `X-Session: <session>` -- never as a query parameter.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from cua.agent.service import AgentService, TaskRecord  # noqa: F401  (TaskRecord re-exported for callers)
from cua.server.auth import AuthStore

_STATIC_DIR = Path(__file__).parent / "static"
_MAX_TASK_TEXT = 2000
_PUBLIC_PATHS = {"/", "/manifest.webmanifest"}


class AgentAPI:
    """Auth + HTTP/WS wiring around a shared AgentService. Owns no task state of its own."""

    def __init__(self, service: AgentService, voice=None, auth: AuthStore | None = None):
        self.service = service
        self.voice = voice
        self.auth = auth or AuthStore()

    @property
    def agent(self):
        return self.service.agent

    # ---- delegate task/event operations to the shared service ----------------------------
    def submit_task(self, text: str) -> TaskRecord:
        return self.service.submit_task(text, source="phone")

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self.service.get_task(task_id)

    def cancel_task(self, task_id: str) -> bool:
        return self.service.cancel_task(task_id)

    def subscribe(self):
        return self.service.subscribe()

    def unsubscribe(self, q):
        return self.service.unsubscribe(q)


def _session_from_headers(headers) -> str | None:
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return headers.get("X-Session") or None


def create_handler(api: AgentAPI):
    """Returns an http.server handler bound to `api`."""
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass                                       # never logs headers/bodies -- no risk of logging secrets

        def _client_key(self) -> str:
            return self.client_address[0] if self.client_address else "unknown"

        def _check_session(self) -> bool:
            token = _session_from_headers(self.headers)
            if not api.auth.sessions.validate(token):
                self._json({"error": "unauthorized"}, 401)
                return False
            return True

        def _read_json_body(self) -> tuple[dict | None, str | None]:
            """Returns (data, None) or (None, error_message). Never raises on malformed input."""
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return None, "invalid Content-Length"
            if length <= 0:
                return {}, None
            if length > 65536:
                return None, "body too large"
            try:
                raw = self.rfile.read(length)
                data = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                return None, "malformed JSON body"
            if not isinstance(data, dict):
                return None, "body must be a JSON object"
            return data, None

        def _json(self, data, code=200):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _static(self, path: str):
            fp = _STATIC_DIR / path.lstrip("/")
            if not fp.is_file() or not fp.resolve().is_relative_to(_STATIC_DIR.resolve()):
                self.send_error(404)
                return
            ct = {"html": "text/html", "js": "application/javascript", "css": "text/css",
                  "png": "image/png", "ico": "image/x-icon", "json": "application/json",
                  "webmanifest": "application/manifest+json"}.get(fp.suffix.lstrip("."), "application/octet-stream")
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Session")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._static("index.html")
                return
            if path.startswith("/static/"):
                self._static(path[7:])
                return
            if path == "/manifest.webmanifest":
                self._static("manifest.webmanifest")
                return
            if not self._check_session():
                return
            if path == "/status":
                self._json({
                    "busy": api.service.is_busy(),
                    "voice": api.voice is not None and hasattr(api.voice, "available") and api.voice.available,
                })
            elif path.startswith("/task/"):
                tid = path[6:]
                rec = api.get_task(tid)
                self._json(rec.to_dict()) if rec else self._json({"error": "not found"}, 404)
            elif path == "/voice/start":
                if api.voice and hasattr(api.voice, "start_listening"):
                    api.voice.start_listening()
                    self._json({"ok": True})
                else:
                    self._json({"error": "voice not available"}, 503)
            elif path == "/voice/stop":
                if api.voice and hasattr(api.voice, "stop_listening"):
                    text = api.voice.stop_listening()
                    self._json({"ok": True, "text": text})
                else:
                    self._json({"error": "voice not available"}, 503)
            else:
                self.send_error(404)

        def do_POST(self):
            path = self.path.split("?")[0]

            if path == "/auth/login":
                data, err = self._read_json_body()
                if err:
                    self._json({"error": err}, 400)
                    return
                password = data.get("password")
                if not isinstance(password, str) or not password:
                    self._json({"error": "password is required"}, 400)
                    return
                if not api.auth.throttle.allowed(self._client_key()):
                    self._json({"error": "too many attempts, try again later"}, 429)
                    return
                session = api.auth.login(password, self._client_key())
                if session is None:
                    self._json({"error": "invalid credentials"}, 401)   # never says WHICH part was wrong
                    return
                self._json({"session": session}, 200)
                return

            if not self._check_session():
                return

            if path == "/auth/logout":
                token = _session_from_headers(self.headers)
                if token:
                    api.auth.sessions.revoke(token)
                self._json({"ok": True})
            elif path == "/task":
                data, err = self._read_json_body()
                if err:
                    self._json({"error": err}, 400)
                    return
                text = data.get("text")
                if not isinstance(text, str) or not text.strip():
                    self._json({"error": "text is required"}, 400)
                    return
                if len(text) > _MAX_TASK_TEXT:
                    self._json({"error": f"text too long (max {_MAX_TASK_TEXT} chars)"}, 400)
                    return
                rec = api.submit_task(text.strip())
                self._json({"id": rec.id, "status": rec.status}, 201)
            elif path.startswith("/task/") and path.endswith("/cancel"):
                tid = path[len("/task/"):-len("/cancel")]
                ok = api.cancel_task(tid)
                self._json({"ok": ok} if ok else {"ok": False, "error": "task not active or unknown"}, 200 if ok else 404)
            else:
                self.send_error(404)

    return Handler


def _ws_sync_handler(api: AgentAPI, websocket):
    """Synchronous WebSocket handler. First message must be {"session": "<token>"}; anything else, or an
    invalid/expired session, closes the connection without ever subscribing it to events."""
    try:
        msg = websocket.recv(timeout=10)
        data = json.loads(msg)
        if not api.auth.sessions.validate(data.get("session")):
            websocket.send(json.dumps({"error": "unauthorized"}))
            return
    except Exception:
        return

    websocket.send(json.dumps({"type": "connected", "time": time.time()}))
    q = api.subscribe()
    try:
        while True:
            try:
                event = q.get(timeout=30)
                websocket.send(json.dumps(event, default=str))
            except Exception:
                try:
                    websocket.ping()
                except Exception:
                    break
    finally:
        api.unsubscribe(q)


def run_server(service: AgentService, voice=None, host="127.0.0.1", port=8420, auth: AuthStore | None = None):
    """Start the API server on top of a shared AgentService. Blocks the calling thread.

    Serves HTTPS with a self-signed, locally-generated certificate when the `cryptography` package is
    installed (see cua/server/tls.py); otherwise falls back to plain HTTP with an explicit warning -- the
    browser's microphone/speech APIs will not work over that fallback (they require a secure context).
    """
    from http.server import HTTPServer

    api = AgentAPI(service, voice=voice, auth=auth)

    if api.auth.generated_pin:
        print(f"Buddy phone PIN (shown once -- write it down): {api.auth.generated_pin}")
    else:
        print("Buddy phone PIN: already set (delete .cache/buddy_auth.json to reset)")

    ssl_ctx = None
    scheme, ws_scheme = "http", "ws"
    try:
        from cua.server.tls import ensure_cert, tls_available
        if tls_available():
            import ssl
            cert, key = ensure_cert()
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(cert), str(key))
            scheme, ws_scheme = "https", "wss"
    except Exception as e:
        print(f"WARNING: could not set up HTTPS ({e}); falling back to plain HTTP. The phone browser's "
             "microphone will not work over that (it requires a secure context) -- everything else still does.")

    Handler = create_handler(api)
    httpd = HTTPServer((host, port), Handler)
    httpd.allow_reuse_address = True
    if ssl_ctx:
        httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
    else:
        print("WARNING: serving plain HTTP -- traffic on this LAN is not encrypted. Install the `cryptography` "
             "package (already in requirements.txt) and restart to enable HTTPS.")

    print(f"Buddy API: {scheme}://{host}:{port}")
    if host == "0.0.0.0":
        from cua.server.tls import local_ip
        ip = local_ip()          # same resolution the cert's SAN was built from -- see tls.py docstring
        print(f"Phone: {scheme}://{ip}:{port}  (log in with the PIN above)")

    ws_thread = threading.Thread(target=_run_ws, args=(api, host, port + 1, ssl_ctx), daemon=True, name="ws-server")
    ws_thread.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def _run_ws(api: AgentAPI, host: str, port: int, ssl_ctx=None):
    import websockets.sync.server as ws_sync
    try:
        with ws_sync.serve(lambda ws: _ws_sync_handler(api, ws), host, port, ssl_context=ssl_ctx) as server:
            server.serve_forever()
    except Exception:
        pass
