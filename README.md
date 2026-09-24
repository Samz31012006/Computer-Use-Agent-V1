# Buddy — a local-first Windows computer-use agent

Buddy takes typed or spoken commands and actually drives your desktop — opening apps, clicking things, typing
text, searching the web, finding files — running **entirely on-device**, on ordinary laptop hardware, with no
cloud calls of any kind.

```
"open chrome and search for the latest spacex launch"
"bluetooth settings"
"click the first video"
(spoken) "lock the pc"
```

## Why this exists

Most "computer-use agent" demos assume a GPU, a cloud LLM, or both. Buddy is built for the opposite case: a
plain laptop (this project targets a 4-core i5, ~7.6 GB RAM, no GPU) with everything running locally —
planning, speech-to-text, and screen understanding. That constraint shapes almost every design decision here,
and the project tries to be honest about the resulting trade-offs rather than hide them.

## How it works, in one picture

```
 command text ──► RouterPlanner (regex + catalog, <1ms, no RAM)
                        │ miss
                        ▼
                  LLMPlanner (local model via llama-server, only when the router can't)
                        │
                        ▼
                 TaskPlan (steps)
                        │
                        ▼
     StepRunner ── SkillEngine → UIAEngine → OcrActionEngine → VisualGroundingEngine
                        │              (each step is verified; failures retry/repair/fall through)
                        ▼
                  WindowsDriver ──► your desktop
```

A deterministic router handles everyday commands instantly and for free; the local LLM is consulted only when
the router doesn't recognize something. The same "cheap method first, escalate only on failure" idea repeats
for finding things on screen (UI Automation → OCR → lightweight CV heuristics → a small vision model, last
resort) and for the model itself (the planner picks whichever locally-available model actually fits in
*current* free RAM, rather than being pinned to one that might not load at all).

A full write-up of every layer — task lifecycle, the recovery ladder, the visual grounding cascade, the voice
pipeline, the safety model — is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest tests          :: 338 tests, all offline (fakes only, no model or desktop needed)

start_buddy.bat                                :: desktop UI
.venv\Scripts\python -m cua "open notepad"     :: one-shot CLI
.venv\Scripts\python -m cua                    :: CLI REPL
```

Buddy works with **nothing** in `models/` — it stays on the deterministic router path and voice is disabled.
See [`models/README.md`](models/README.md) for what to add and where to get it, if you want the local LLM
planner, voice input, or visual grounding.

## Voice

Microphone → energy-based voice-activity detection → a single accurate speech-to-text pass (`faster-whisper`,
falls back to Vosk) → the same text path as typing. Auto-stops shortly after you stop talking, rejects
transcripts the model itself isn't confident in (so background noise doesn't get executed as a command
instead of your actual words), and never runs partial speech. Needs no model files beyond what
`requirements.txt` installs — `faster-whisper` fetches its own model on first use.

## Access from your phone

```bat
.venv\Scripts\python start_server.py           :: LAN-only, self-signed HTTPS (required for mic access)
```

Prints a one-time PIN and a `https://<lan-ip>:8420` URL. Session-token auth (no passwords in URLs, no
cookies); the phone and the laptop app share the exact same agent, so only one task ever runs at a time no
matter which interface sent it.

## Safety

- **No shell / run-command tool exists at all.** An action Buddy doesn't have an explicit tool for simply
  can't be expressed, by the router or by the model — this isn't a filter on top of a general-purpose
  executor, there is no general-purpose executor.
- Every step — from the router *or* the model — passes through one safety gate before it runs. It asks for
  your OK before: closing/uninstalling apps, overwriting files, destructive hotkeys (Alt+F4, Delete, ...),
  clicking anything named Delete/Remove/Uninstall/..., or changing anything while a system-configuration
  window (Settings, Task Manager, Registry Editor) is focused.
- Global emergency stop: **Ctrl+Alt+Q**, from any interface, at any point mid-task.
- A step that already ran once (typed text, clicked a button) is never blindly re-run after a failed check —
  it's *repaired* by inspecting current state, not repeated.

## Project layout

```
cua/agent/        orchestrator, router, planner chain, recovery ladder, AgentService (the shared core)
cua/actions/       32 declared tools → registry; the four action engines (skill/UIA/OCR/visual)
cua/perception/    screen capture, OCR, the visual grounding cascade, the small vision model
cua/input/         microphone, voice-activity detection, speech-to-text, the voice state machine
cua/models/        the local LLM runtime (llama-server as a child process) + planner
cua/computer/      low-level Windows input (SendInput) and window management
cua/safety/        the confirmation policy + the global kill switch
cua/server/        the phone HTTPS/WebSocket server, PIN auth, self-signed TLS
cua/ui/            the desktop Tkinter UI
bench/             planning-accuracy and speech-to-text evaluation harnesses, run against measured corpora
tests/             338 tests — entirely offline, fakes throughout, no model or live desktop required
```

## Adding a tool

Declare it in `cua/actions/tools.py` (name, argument schema, safety level, verification predicate) and
implement it in an engine. The router and the model both see it automatically; nothing else is callable.

## Honest limitations

- Visual grounding without a named UI control (ordinal/spatial phrases like "the first result", "top-right
  button") is capped below auto-execute confidence on complex real pages — it will often ask rather than
  risk a wrong click.
- The local vision model's coordinate output isn't reliable enough to trust broadly; it's a last resort, not
  a general solution.
- Open-ended multi-step natural-language planning is meaningfully behind the deterministic router's coverage
  of everyday commands — that gap is exactly what the router keeps being extended to close.
- Windows only, by design (UI Automation, SendInput, WASAPI are all Windows APIs).

## License

MIT — see [`LICENSE`](LICENSE).
