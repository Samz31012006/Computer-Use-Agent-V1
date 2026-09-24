"""Energy-based voice activity detection: adaptive noise-floor calibration + debounced start + hangover
endpointing. Feed it one RMS value per audio chunk (`audioop.rms`); it tells you the instant speech starts
(debounced, so a single click/pop doesn't trigger it) and the instant it ends (a run of near-silent chunks
after confirmed speech -- the "hangover").

Why energy-based rather than webrtcvad/Silero: this project targets a single local Windows machine with no
GPU; a calibrated RMS threshold with debounce and hangover is enough to solve the actual problem (know when
the user starts and stops talking) without a native/ML dependency that could fail to install. If a specific
noisy environment defeats this, swap the implementation behind this same small feed()/reset() interface.

This is what actually fixes three separate real complaints in the voice pipeline:
  - "opens late" / clips the first word: VAD calibrates against ambient noise so real speech is detected in
    ~2 chunks (see start_confirm_chunks), and the caller keeps a small pre-roll so those confirming chunks
    (which WERE speech) aren't lost.
  - "doesn't stop when I stop talking": hangover is short (default ~700ms) instead of the old fixed 3s
    silence timeout, and is based on actual audio energy, not on the STT model happening to return text.
  - "reads wrong input": short noise blips (a cough, a click) that clear the debounce but never sustain are
    exposed via `speech_duration_chunks` so the caller can discard them instead of asking Whisper to
    transcribe near-silence, which is a well-known hallucination trigger for Whisper-family models.

`postroll_chunks` defaults to 0, not a few hundred ms of trailing padding, for the same reason: measured
directly (see tests/test_voice.py's trailing-artifact case and bench/), appending ~200-500ms of near-silence
after the last real speech frame is the single worst case for Whisper -- it's not enough silence for the
model's own internal VAD filter to confidently discard, so it hallucinates stray tokens (e.g. a trailing
" //") onto an otherwise perfect transcript. Cutting cleanly at the last voice-classified chunk, with no
trailing pad, reproducibly avoided this; a MUCH longer pad (2s+) also avoided it, but there is no reason to
add either the latency or the risk when 0 already gives a clean cut with no measured downside.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VADConfig:
    calibration_chunks: int = 4        # chunks of leading ambient noise used to seed the floor
    noise_multiplier: float = 3.5      # a chunk counts as speech above floor * multiplier
    min_threshold: float = 120.0       # absolute floor so a near-silent room doesn't trigger on a whisper
    start_confirm_chunks: int = 2      # consecutive voice-like chunks required to confirm speech START
    end_hangover_chunks: int = 7       # consecutive silence-like chunks required to confirm speech END
    preroll_chunks: int = 2            # chunks kept BEFORE the confirmed start (they were real speech)
    postroll_chunks: int = 0           # deliberately 0 -- see module docstring on trailing-silence hallucination


class EnergyVAD:
    """Stateful, one chunk at a time. `feed(rms)` returns True exactly once -- the instant end-of-speech is
    confirmed. After that, read `speech_start_idx` / `end_idx` (chunk indices, inclusive) to slice the
    caller's own audio buffer down to just the speech span (plus pre/post-roll)."""

    def __init__(self, config: VADConfig | None = None):
        self.cfg = config or VADConfig()
        self._chunk_idx = -1
        self._floor = self.cfg.min_threshold
        self._floor_n = 0
        self._voice_run = 0
        self._silence_run = 0
        self.speaking = False
        self.ever_spoke = False
        self.speech_start_idx: int | None = None
        self.last_voice_idx: int = -1

    def _threshold(self) -> float:
        return max(self._floor * self.cfg.noise_multiplier, self.cfg.min_threshold)

    def feed(self, rms: float) -> bool:
        self._chunk_idx += 1
        is_voice = rms > self._threshold()

        # Only ever let genuinely-quiet chunks adjust the floor -- if we let a loud chunk feed the average
        # just because it happened to land inside the first `calibration_chunks`, a user who starts talking
        # immediately would drag their own voice into the "noise" floor and raise the bar against themselves.
        if not self.speaking and not is_voice:
            if self._floor_n < self.cfg.calibration_chunks:
                self._floor_n += 1
                self._floor += (rms - self._floor) / self._floor_n
            else:
                self._floor += (rms - self._floor) * 0.1          # slow EMA adapt to drifting room noise

        if is_voice:
            self._voice_run += 1
            self._silence_run = 0
            self.last_voice_idx = self._chunk_idx
            if not self.speaking and self._voice_run >= self.cfg.start_confirm_chunks:
                self.speaking = True
                self.ever_spoke = True
                self.speech_start_idx = max(0, self._chunk_idx - self._voice_run + 1 - self.cfg.preroll_chunks)
            return False

        self._voice_run = 0
        self._silence_run += 1
        if self.speaking and self._silence_run >= self.cfg.end_hangover_chunks:
            return True
        return False

    @property
    def speech_duration_chunks(self) -> int:
        if self.speech_start_idx is None:
            return 0
        return max(0, self.last_voice_idx - self.speech_start_idx + 1)

    @property
    def end_idx(self) -> int:
        """Last chunk index to keep, including post-roll. Only meaningful once `ever_spoke`."""
        return self.last_voice_idx + self.cfg.postroll_chunks
