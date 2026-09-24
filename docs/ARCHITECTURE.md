# Buddy architecture

How the planner, action engines, perception cascade, voice pipeline and safety model fit together, and why
each piece is shaped the way it is. Three ideas run through nearly every layer below, so it's worth naming
them up front:

- **Deterministic before probabilistic.** A regex+catalog router handles what it can recognize instantly and
  for free. The local LLM is the fallback for what the router misses — never the default path.
- **Cheap before expensive.** The same ladder repeats in perception (UI Automation → OCR → CV heuristics →
  vision model) and in speech-to-text: try the fast/free/reliable method first, escalate only on failure.
- **Confidence-gated action.** Nothing acts on a low-confidence result — not a screen coordinate, not a voice
  transcript. Below a measured threshold, the system asks again instead of guessing.

## 1. Interfaces & AgentService

The laptop window, the phone browser, and the CLI are three thin clients over **one** shared core. Earlier,
each interface built its own `Agent`, which meant its own model process and its own kill-switch registration
— running the laptop app and the phone server at once meant two agents fighting over the same mouse.
`AgentService` (`cua/agent/service.py`) exists specifically to make that impossible: it owns the one `Agent`,
the one `KillSwitch`, a single FIFO worker thread (only one task ever touches the keyboard/mouse at a time),
per-task state, and a pub/sub event stream every subscriber reads from.

```
 Laptop UI    Phone browser    CLI
     │              │           │
     └──────── submit text, subscribe to events ────────┘
                        │
                        ▼
              AgentService  (singleton: one Agent, one KillSwitch, one worker thread)
                        │ owns
                        ▼
   Agent (orchestrator) ── ChainPlanner (router → LLM) ── StepRunner (engines + recovery) ── llama-server
```

A lightweight `_session_context` dict — last app focused, last folder, last outcome — is threaded through
every call, so "open notepad" followed, separately, by "type hello" resolves correctly, without keeping a
growing conversation history.

## 2. Task lifecycle

`Agent.run()` turns text into actions on screen. Every path through it is bounded — steps, replans, retries,
and wall-clock time all have hard caps — so nothing can loop forever.

1. **Plan** — `ChainPlanner.plan(text, context)` → a `TaskPlan` or `None`.
2. **Approve** — router plans never ask; a model-made plan can require one up-front confirmation if configured.
3. **Execute each step** — a per-step safety check, then `StepRunner.run()`: engine selection, verification,
   retry/repair.
4. **Bounded replan** — on step failure, ask the planner to replan around the specific failure, at most once
   by default.
5. **Result** — a `TaskResult` with per-step detail, stage timings, and a structured reason (`ok`, `no_plan`,
   `declined`, `step_failed`, `aborted`).

A `Budget` (max_steps=12, max_replans=1, max_retries=1, timeout_s=90) is threaded through the whole run. The
model is called at most `1 + max_replans` times per task — never once per action.

## 3. Planning: router before model

`ChainPlanner` tries `RouterPlanner` first; the local LLM is consulted only when the router returns `None`.
On hardware this constrained, "fall back to the model" has to be a genuine fallback, not the common case.

**What the router understands:**

| Category | Examples |
|---|---|
| Explicit verbs | open / close / search / type / click / scroll / copy / find, each with its own clause parser |
| Bare targets | `"chrome"`, `"downloads"`, `"bluetooth settings"` — matched only against a known app, folder, Settings page or site, never guessed |
| System control | volume, mute, media transport, lock — plain global media keys |
| Multi-clause | "open chrome, go to youtube, search cats" — splits on connectors, or on an implicit new verb with no connector at all |

Bare-target matching is the router's broadest net, so it's the most carefully fenced: a candidate must equal
or contain a name from the *actually-installed* Start-menu app cache, a known folder, a known Settings page,
or a known site. Verified against real misheard voice transcripts and ordinary prose ("hello there", "tell
me a joke") with zero false launches.

When the router misses, `LLMPlanner` (`cua/models/local_llm.py`) prompts a local model, constrained to a
per-tool JSON schema (only the declared arguments for the one tool named), via `llama-server` running as a
child process over localhost HTTP — not an in-process binding, so a model crash or OOM can never take the
agent down, and its RAM is fully released on process exit.

## 4. RAM-adaptive model choice

A fixed choice of the larger, more capable planner model (Qwen3-1.7B, ~1.46 GB to load) simply couldn't load
on a machine that routinely has ~1.2 GB free with a browser open — 53% planning accuracy from a model that
never runs is 0% in practice.

| Model | Needs to load | Strict accuracy* | Chosen when |
|---|---|---|---|
| Qwen3-1.7B-Q4_K_M | ~1.46 GB | 53% | free RAM comfortably covers it |
| Qwen3-0.6B-Q4_K_M | ~0.66 GB | 28% | free RAM is tight |

<sub>* `bench/dynamic_corpus.py`, strict scoring — multi-step natural-language planning, not router-coverable
commands.</sub>

`Config.resolve_model()` checks *current* free RAM at load time and picks the most capable model that
actually fits, falling back to a smaller one rather than refusing to plan at all. The same
`model_ram_need_mb()` formula both chooses a model and gates loading it, so the two can never disagree. In
practice the assistant gets smarter automatically as RAM frees up, and never goes fully mute when it doesn't.

This adapts *which* local model runs; it doesn't raise the ceiling on what a ≤1.7B model can do. Complex
multi-step requests remain meaningfully weaker than the router's coverage of everyday commands — which is
why router coverage, not model size, is the metric this project keeps pushing on.

## 5. Executing a step

Every step goes through `StepRunner`, which tries each engine that declares it can handle the action, in
order, with a bounded recovery ladder:

1. **Run + verify** — the first capable engine executes the step, then a cheap predicate checks it happened.
2. **Retry (idempotent only)** — a step safe to repeat (opening an app) gets one retry; a click or keystroke
   never does.
3. **Repair (non-idempotent)** — an engine's `repair()` inspects current state and fixes only what's missing,
   instead of blindly re-running the action.
4. **Next engine** — fall through to the next engine that declares it can handle this action.
5. **Give up** — the Agent may request one bounded replan, or report failure with full detail.

**The four action engines, cheapest first:**

```
SkillEngine (hand-written, deterministic)
   → UIAEngine (Windows UI Automation tree, name-based control lookup)
      → OcrActionEngine (screen text, exact then fuzzy match)
         → VisualGroundingEngine (the full grounding cascade, see §6)
```

32 tools are registered in `ToolRegistry` (open_app, close_app, type_text, click_ui, navigate_url, find_file,
create_file, hotkey, scroll, ...). Each declares its own argument schema and idempotency, which is what lets
the recovery ladder reason generically about "is it safe to retry this" without a special case per tool.

## 6. Visual grounding cascade

Turning "click the first channel" or "the button in the top-right" into an actual screen coordinate is the
hardest perception problem here — and the one most tempting to solve by reaching straight for a vision model.
The cascade deliberately doesn't:

```
UIA (runs first, before this cascade — named controls)
  → OCR exact (screen text, substring match)
    → OCR fuzzy (difflib-scored, skipped for ordinal/spatial phrasing)
      → CV heuristics (ordinal/spatial reasoning over OCR word clusters)
        → VLM (SmolVLM-500M, last resort, capped confidence)
```

Every method returns the same shape — `GroundResult(found, x, y, confidence, description, bbox, reason,
method)` — and nothing downstream ever sees a raw coordinate that didn't pass through it:

| Confidence | Behaviour |
|---|---|
| ≥ 0.75 | execute directly |
| ≥ 0.50 | execute, lean on the existing verify/retry ladder to catch a miss |
| < 0.50 | refuse — ask rather than guess |

Live testing against a real page (not synthetic fixtures) showed the CV heuristic layer can misfire on
complex real-world layouts — it once matched a browser's own title-bar text instead of page content. Rather
than raise its reported confidence to look more capable, its confidence is deliberately capped *below* the
auto-execute threshold, so by default it refuses on anything it isn't proven reliable on. The VLM (SmolVLM-
500M, chosen over moondream2 for this hardware's RAM budget) gets the same treatment: it answers general
questions about a screenshot correctly, but its coordinate output measured unreliable, so its confidence
ceiling is capped at 0.35 — always below the auto-execute line.

## 7. Voice pipeline

Microphone → VAD → STT → the same text path as typing. The voice module never runs the agent itself; it just
produces a string and hands it to `AgentService` exactly like typed text would.

```
Microphone (WASAPI at native rate, resampled to 16kHz in software)
  → EnergyVAD (adaptive noise floor, debounced start ~200ms, hangover end ~700ms)
    → trim to exact speech span (no trailing pad)
      → faster-whisper base.en (single pass, local_files_only)
        → confidence gate (no_speech_prob + avg_logprob)
```

Why each stage is shaped this way:

- **WASAPI, not MME** — Windows' legacy MME host API produced near-dead audio (measured RMS ~1 of ~32000)
  when forced to capture at 16kHz on hardware whose native rate is 44.1/48kHz. WASAPI captures at its own
  native rate reliably; resampling happens in software after.
- **A real VAD, not a fixed silence timer** — an earlier design re-transcribed the entire growing recording
  every ~1.2s, then again at the end, then a third time with a bigger "refine" model: up to 5+ full Whisper
  passes for one command. `EnergyVAD` replaces all of it with one pass, on just the trimmed speech span.
- **Zero post-roll padding** — measured directly: appending ~200–500ms of trailing near-silence is Whisper's
  exact hallucination sweet spot. Cutting cleanly at the last voice-classified frame eliminated it.
- **base.en over tiny.en** — benchmarked on 8 synthetic test utterances: tiny.en scored 96.1% (with a real
  miss, "Open Chrome" heard as "Open Curl"); base.en scored 100% at ~1.4s latency.
- **Confidence-gated transcripts** — Whisper invents text from silence and noise. Every transcript is scored
  from the model's own `no_speech_prob` and `avg_logprob`, with known hallucination phrases hard-zeroed.
  Below threshold, the UI shows "Didn't catch that" instead of executing an invented command — tuned against
  real recordings so genuine speech (weakest case measured: 0.27) is never wrongly rejected (threshold: 0.15).
- **Background pre-warm** — the STT model loads on a background thread the instant recording starts, so load
  time overlaps with the user talking. `local_files_only=True` also removed an unnecessary Hugging Face Hub
  network check on every load (observed ~169s on a slow connection, cut to ~1.3s).

## 8. Safety model

The planner — router or model — cannot bypass safety. Every step, from whatever produced it, passes through
`needs_confirmation()` before it runs:

1. **Tool-declared risk** — a tool's own `confirm_if` rule (e.g. `close_app`, overwriting a file).
2. **Destructive hotkeys** — Alt+F4, Delete, Ctrl+Shift+Esc, ...
3. **Destructive-named targets** — clicking anything matching delete / remove / uninstall / format / sign out
   / shut down / restart.
4. **System-configuration windows** — any UI change while Settings, Task Manager, Registry Editor, etc. is
   the foreground window.

There is deliberately **no shell or run-command tool** — an unknown action fails closed rather than being
expressible at all. A global kill-switch hotkey (Ctrl+Alt+Q by default) is registered once by `AgentService`
and aborts whatever's running immediately, from any interface. The `confirm` callback is supplied per
interface: an interactive dialog on the laptop UI, `deny_all` (fail closed, no one to ask) for the headless
phone server.

## 9. Observability & evaluation

- **Structured event log** — `logs/tasks.jsonl`, one line per completed task: command text, which planner
  handled it, per-stage timing, RAM/CPU at the time, retries, replans, and the exact error if one occurred.
- **Benchmark harnesses** — `bench/dynamic_eval.py` scores planning accuracy against a corpus of multi-step
  tasks; `bench/stt_eval.py` compares speech-to-text candidates on synthetic (SAPI-TTS-generated) speech
  against known ground truth, so model choices come from a head-to-head number, not a guess.
- **`VoiceMetrics`** applies the same discipline to voice specifically: device, sample rate, VAD timestamps,
  raw transcript, confidence, and final text are all captured per utterance — nothing about what was "heard"
  is ever silently cleaned up before the user can see it.

## 10. Known limits, stated plainly

- CV-only grounding on complex real pages is capped below auto-execute confidence by design — safe, but
  conservative; it will often refuse rather than risk a wrong click on a busy page.
- The vision model's coordinate output isn't reliable enough to trust above a low confidence ceiling.
- Multi-step natural-language planning is meaningfully behind the router's coverage of everyday commands —
  28–53% strict accuracy depending on which local model fit in RAM, versus ~88% router coverage measured on
  an everyday-command sample.
- "Shut down" is deliberately not in the router — with voice mishears in the mix, a misheard shutdown is too
  costly a false positive to risk for a rarely-needed command.
- No brightness control — needs vendor-specific APIs not yet implemented.
- No cloud fallback, anywhere, by design. Every workaround above exists because the ceiling of a ≤1.7B local
  model on this exact hardware is real, and the project tries to be honest about that rather than hide it.
