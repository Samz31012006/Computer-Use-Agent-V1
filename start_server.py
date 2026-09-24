"""Start Buddy headless -- phone access only, no desktop window.

Builds the same AgentService the laptop UI uses, so if you run the laptop app (start_buddy.bat) instead, it
already serves the phone from within that process; run this script only when you want phone access WITHOUT
opening the desktop window. Do not run both at once -- that would be two separate AgentService instances (two
Agents, two model runtimes) again, exactly the problem this service was built to avoid. If both try to bind the
same port, the second one will fail to start with "address already in use", which is the intended guard rail.

Login uses a PIN, not a URL token: on first run one is generated and printed ONCE (write it down; the server
never shows it again, since only its hash is kept). Bind to 0.0.0.0 for phone access -- still LAN-only, never
the public internet, never exposed via UPnP/port-forwarding.

Usage:
  python start_server.py                     # LAN access (0.0.0.0:8420)
  python start_server.py --local             # localhost only
  python start_server.py --set-password xyz  # set a custom password/PIN instead of the generated one
"""
import argparse

from cua.agent.service import AgentService
from cua.input.voice import build_voice
from cua.safety.policy import deny_all
from cua.server.api import run_server
from cua.server.auth import AuthStore


def main():
    p = argparse.ArgumentParser(description="Buddy API server (headless -- no desktop window)")
    p.add_argument("--local", action="store_true", help="Bind to localhost only")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--set-password", type=str, default=None, help="set/replace the login password, then exit")
    args = p.parse_args()

    auth = AuthStore()
    if args.set_password:
        auth.set_password(args.set_password)
        print("Password updated. All existing phone sessions were signed out.")
        return

    service = AgentService(confirm=deny_all)      # headless: no UI to ask, so risky steps fail closed
    voice = build_voice(on_event=lambda *a: None)
    voice.warm()

    host = "127.0.0.1" if args.local else "0.0.0.0"
    run_server(service, voice=voice, host=host, port=args.port, auth=auth)


if __name__ == "__main__":
    main()
