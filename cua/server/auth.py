"""Password-based auth for the phone API: hashed PIN, short-lived sessions, login throttling.

Design, and why:
  - One shared PIN (not a username+password): this is a single-owner local device, not a multi-user service.
    There is nothing to enumerate, so "avoid leaking whether a login is correct" just means a uniform,
    generic failure response -- there's no username half of the check that could leak separately.
  - The PIN is generated once on first run and its HASH (scrypt, stdlib, memory-hard) is persisted to disk;
    the plaintext PIN is shown exactly once, at creation, the same way the old bearer token was shown on every
    run -- except now it survives restarts instead of changing every launch.
  - Login exchanges the PIN for a session token (cryptographically random, `secrets.token_urlsafe`), which is
    what every subsequent HTTP and WebSocket call presents instead of the password itself. Sessions expire
    (sliding TTL) and can be revoked (logout, or by restarting the server -- sessions are in memory only).
  - No cookies: the client sends the session explicitly as `Authorization: Bearer <session>` (or `X-Session`).
    A browser never attaches that header to a cross-site request on its own, so there is nothing for a
    CSRF-style forged request to ride on -- simpler than adding CSRF tokens on top of a cookie-based session.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _DKLEN = 2**14, 8, 1, 32


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN)
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN)
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


class SessionStore:
    """In-memory session tokens with a sliding TTL. Nothing here is written to disk -- a server restart
    invalidates every session, which is the simplest possible "revoke everything" story."""

    def __init__(self, ttl_s: float = 3600.0):
        self.ttl_s = ttl_s
        self._sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = time.time() + self.ttl_s
        return token

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            exp = self._sessions.get(token)
            if exp is None:
                return False
            if time.time() > exp:
                del self._sessions[token]
                return False
            self._sessions[token] = time.time() + self.ttl_s   # sliding expiry: still-active sessions don't die
            return True

    def revoke(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def revoke_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


class LoginThrottle:
    """Sliding-window lockout per source key (the client IP). Not per-account, since there is only one."""

    def __init__(self, max_attempts: int = 5, lockout_s: float = 300.0):
        self.max_attempts, self.lockout_s = max_attempts, lockout_s
        self._fails: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            recent = [t for t in self._fails[key] if now - t < self.lockout_s]
            self._fails[key] = recent
            return len(recent) < self.max_attempts

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._fails[key].append(time.time())

    def record_success(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


class AuthStore:
    """Owns the persisted PIN hash plus the session store and login throttle for one server instance."""

    def __init__(self, path: str | Path = ".cache/buddy_auth.json", session_ttl_s: float = 3600.0):
        self.path = Path(path)
        self.sessions = SessionStore(session_ttl_s)
        self.throttle = LoginThrottle()
        self._hash: str | None = None
        self.generated_pin: str | None = None       # set only on first run; the ONLY time the PIN is visible
        self._load_or_init()

    def _load_or_init(self) -> None:
        if self.path.is_file():
            try:
                self._hash = json.loads(self.path.read_text(encoding="utf-8"))["hash"]
                return
            except Exception:
                pass                                  # corrupt file: fall through and re-provision
        pin = f"{secrets.randbelow(1_000_000):06d}"
        self._hash = hash_password(pin)
        self.generated_pin = pin
        self._persist()

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"hash": self._hash}), encoding="utf-8")

    def verify(self, password: str) -> bool:
        return bool(password) and self._hash is not None and verify_password(password, self._hash)

    def set_password(self, password: str) -> None:
        self._hash = hash_password(password)
        self.sessions.revoke_all()                    # changing the PIN invalidates every existing session
        self._persist()

    # ---- login flow ------------------------------------------------------------------
    def login(self, password: str, client_key: str) -> str | None:
        """Returns a new session token, or None on bad credentials / while rate-limited."""
        if not self.throttle.allowed(client_key):
            return None
        if not self.verify(password):
            self.throttle.record_failure(client_key)
            return None
        self.throttle.record_success(client_key)
        return self.sessions.create()
