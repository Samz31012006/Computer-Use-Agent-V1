"""Voice input: microphone -> VAD -> STT -> command text.

State machine:
  IDLE -> LISTENING -> PROCESSING -> (yields command) -> IDLE

The voice module does NOT run the agent. It produces command strings that the agent consumes,
exactly like TextInput. The UI subscribes to events (partial transcripts, state changes) via callbacks.

Redesigned around a real VAD (cua/input/vad.py) instead of a fixed 3s silence timer and a "re-transcribe the
growing buffer every ~1.2s" streaming approximation. Concretely, per chunk (100ms):
  1. compute RMS, feed it to EnergyVAD -- no STT call at all while recording.
  2. VAD confirms speech START within ~200ms (2 chunks) of real speech, and END within ~700ms of the user
     falling silent (vs. the old fixed 3s wait).
  3. once END is confirmed, the raw audio is trimmed to just the speech span (+ small pre/post-roll) and
     handed to the STT backend for exactly ONE transcription pass -- not the old N-repartials-plus-finish-
     plus-refine sequence, which was the dominant source of "processing is slow".
A background thread calls `stt.warm()` the instant recording starts, so first-use model load overlaps with
the user actually talking instead of landing on top of the "Processing..." wait after they stop.

Every stage is instrumented (VoiceMetrics, emitted as VoiceEvent.METRICS alongside the final transcript) so
a misheard command can be diagnosed instead of guessed at: which device and sample rate were actually used,
how much audio was captured, its level, when VAD detected speech start/end, how long the STT pass took, and
-- critically -- the exact text produced, next to whatever was actually sent onward, so nothing is silently
"cleaned up" to look more correct than what was actually heard.
"""
from __future__ import annotations

import audioop
import enum
import json
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from cua.input.microphone import Microphone
from cua.input.stt import FasterWhisperSTT, NullSTT, SpeechToText, VoskSTT
from cua.input.vad import EnergyVAD, VADConfig

# Set BUDDY_VOICE_DEBUG=1 to save the exact audio handed to the STT model (WAV) plus its full metrics
# (JSONL) under logs/voice_debug/ -- the only way to tell "wrong mic capture" from "VAD cut it wrong" from
# "the model genuinely misheard it" is to listen to and inspect what was actually sent, not guess again.
# Off by default: it's real recorded speech, so it shouldn't be written to disk unasked.
_DEBUG_DIR = Path("logs/voice_debug") if os.environ.get("BUDDY_VOICE_DEBUG") else None


class VoiceState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"


class VoiceEvent(enum.Enum):
    STATE_CHANGED = "state_changed"
    PARTIAL_TRANSCRIPT = "partial"
    FINAL_TRANSCRIPT = "final"
    METRICS = "metrics"
    ERROR = "error"


_MAX_LISTEN_S = 30.0
_NO_INPUT_TIMEOUT = 8.0            # give up if the user never starts talking at all
_MIN_SPEECH_MS = 250               # shorter than this is a click/cough/bump, not a command -- discard it
# Below this, ask again rather than run a command the model likely invented (see cua/input/stt.py).
# Measured, not guessed: on the recorded test set, genuine speech scored 0.27-0.80 (the weakest being a real
# "Open YouTube."), while silence, noise and known hallucination phrases are hard-failed to exactly 0.0 by
# _confidence(). 0.15 therefore sits in the gap -- it blocks invented text without ever rejecting a real
# command, which matters more here: wrongly refusing what someone actually said is the worse failure.
MIN_TRANSCRIPT_CONFIDENCE = 0.15


@dataclass
class VoiceMetrics:
    """One utterance's full pipeline trace -- see module docstring. `.to_dict()` is what gets logged/shown."""
    device: str = ""
    sample_rate: int = 0
    channels: int = 1
    frames_captured: int = 0
    audio_duration_s: float = 0.0          # duration of the VAD-trimmed speech span actually transcribed
    audio_level_rms: float = 0.0
    vad_start_ts: float = 0.0              # 0 if no speech was ever detected
    vad_end_ts: float = 0.0
    stt_ms: float = 0.0                    # time in the single transcription pass
    raw_transcript: str = ""               # exactly what the STT model produced
    confidence: float = 0.0                # the model's own confidence in that transcript (see stt.py)
    rejected: bool = False                 # True => heard something, but too unconfident to act on
    final_text: str = ""                   # exactly what was sent to the planner -- never silently altered

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class VoiceInput:
    def __init__(self, stt: SpeechToText | None = None, on_event: Callable | None = None,
                 vad_config: VADConfig | None = None, max_listen_s: float = _MAX_LISTEN_S,
                 no_input_timeout: float = _NO_INPUT_TIMEOUT, min_speech_ms: float = _MIN_SPEECH_MS):
        self._stt = stt or NullSTT()
        self._mic = Microphone()
        self._on_event = on_event
        self._state = VoiceState.IDLE
        self._partial = ""
        self._stop = threading.Event()
        self._result_q: list[str] = []
        self._listen_thread: threading.Thread | None = None
        self._vad_config = vad_config
        self._max_listen_s = max_listen_s
        self._no_input_timeout = no_input_timeout
        self._min_speech_ms = min_speech_ms
        self._audio_buffer: list[bytes] = []
        self._metrics = VoiceMetrics()
        self.last_metrics: VoiceMetrics | None = None      # the most recently completed utterance's trace

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def partial_transcript(self) -> str:
        return self._partial

    @property
    def available(self) -> bool:
        mic_reason = self._mic.unavailable_reason()
        stt_reason = self._stt.unavailable_reason() if hasattr(self._stt, "unavailable_reason") else None
        return mic_reason is None and stt_reason is None

    def unavailable_reason(self) -> str | None:
        mic_reason = self._mic.unavailable_reason()
        if mic_reason:
            return f"microphone: {mic_reason}"
        stt_reason = self._stt.unavailable_reason() if hasattr(self._stt, "unavailable_reason") else None
        if stt_reason:
            return f"STT: {stt_reason}"
        return None

    def _emit(self, event: VoiceEvent, data=None):
        if self._on_event:
            try:
                self._on_event(event, data)
            except Exception:
                pass

    def _set_state(self, state: VoiceState):
        self._state = state
        self._emit(VoiceEvent.STATE_CHANGED, state)

    def start_listening(self) -> None:
        if self._state != VoiceState.IDLE:
            return
        self._stop.clear()
        self._partial = ""
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True, name="voice-listen")
        self._listen_thread.start()

    def stop_listening(self) -> str | None:
        if self._state == VoiceState.IDLE:
            return None
        self._stop.set()
        if self._listen_thread:
            self._listen_thread.join(timeout=15)
            self._listen_thread = None
        return self._result_q.pop(0) if self._result_q else None

    def _listen_loop(self):
        m = self._metrics = VoiceMetrics(sample_rate=self._mic.sample_rate)
        try:
            self._set_state(VoiceState.LISTENING)
            self._audio_buffer = []
            self._mic.start()                    # start capturing FIRST -- model warmup must never delay this
            m.device, m.channels = self._mic.device_name, self._mic.channels
            threading.Thread(target=self._stt.warm, daemon=True, name="stt-warm").start()

            vad = EnergyVAD(self._vad_config)
            start = time.time()
            null_since = start
            chunk_ms = self._mic.block_ms

            while not self._stop.is_set():
                chunk = self._mic.read(timeout=0.2)
                read_ts = time.time()
                if chunk is None:
                    if not vad.ever_spoke and read_ts - null_since > self._no_input_timeout:
                        break
                    if read_ts - start > self._max_listen_s:
                        break
                    continue
                null_since = read_ts

                self._audio_buffer.append(chunk)
                rms = audioop.rms(chunk, 2)
                if not vad.ever_spoke:
                    self._partial = ""
                ended = vad.feed(rms)
                if vad.speaking and not m.vad_start_ts:
                    m.vad_start_ts = read_ts
                    self._emit(VoiceEvent.PARTIAL_TRANSCRIPT, "Listening…")
                if ended:
                    m.vad_end_ts = read_ts
                    break

                if not vad.ever_spoke and read_ts - start > self._no_input_timeout:
                    break
                if read_ts - start > self._max_listen_s:
                    break

            self._mic.stop()

            if not vad.ever_spoke:
                self._record_metrics()
                self._partial = ""
                self._set_state(VoiceState.IDLE)
                return

            start_idx = vad.speech_start_idx or 0
            end_idx = min(len(self._audio_buffer) - 1, vad.end_idx)
            speech_chunks = self._audio_buffer[start_idx:end_idx + 1]
            duration_ms = len(speech_chunks) * chunk_ms
            if duration_ms < self._min_speech_ms:
                self._record_metrics()          # too short to be a real command -- likely a click/cough
                self._partial = ""
                self._set_state(VoiceState.IDLE)
                return

            self._set_state(VoiceState.PROCESSING)
            self._finalize(b"".join(speech_chunks))

        except Exception as e:
            self._mic.stop()
            self._emit(VoiceEvent.ERROR, str(e))
            self._set_state(VoiceState.IDLE)

    def _record_metrics(self):
        m = self._metrics
        m.frames_captured = len(self._audio_buffer)
        audio = b"".join(self._audio_buffer)
        if audio:
            try:
                m.audio_level_rms = audioop.rms(audio, 2)
            except Exception:
                pass
        self.last_metrics = m
        self._emit(VoiceEvent.METRICS, m)

    def _finalize(self, speech_audio: bytes):
        m = self._metrics
        m.audio_duration_s = len(speech_audio) / 2 / max(1, m.sample_rate)
        try:
            m.audio_level_rms = audioop.rms(speech_audio, 2)
        except Exception:
            pass

        t0 = time.perf_counter()
        try:
            result = self._stt.transcribe(speech_audio)
        except Exception:
            result = None
        m.stt_ms = (time.perf_counter() - t0) * 1000

        text = result.text if result and result.text else ""
        m.raw_transcript = text
        m.confidence = result.confidence if result else 0.0
        # A transcript the model itself isn't confident in is far more likely to be invented from room noise
        # than something the user said -- running it means executing a command nobody gave. Surface it as
        # "didn't catch that" instead. raw_transcript still records what was heard, so nothing is hidden.
        if text and m.confidence < MIN_TRANSCRIPT_CONFIDENCE:
            m.rejected = True
            text = ""
        m.final_text = text
        m.frames_captured = len(self._audio_buffer)
        self.last_metrics = m
        self._emit(VoiceEvent.METRICS, m)
        if _DEBUG_DIR:
            self._save_debug(speech_audio, m)

        self._partial = text
        if text:
            self._emit(VoiceEvent.FINAL_TRANSCRIPT, text)
            self._result_q.append(text)
        self._set_state(VoiceState.IDLE)

    def _save_debug(self, speech_audio: bytes, m: VoiceMetrics) -> None:
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            wav_path = _DEBUG_DIR / f"{stamp}.wav"
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(m.sample_rate)
                w.writeframes(speech_audio)
            with open(_DEBUG_DIR / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"wav": wav_path.name, **m.to_dict()}) + "\n")
        except Exception:
            pass          # debug capture must never break a real voice command

    def listen(self) -> str | None:
        self.start_listening()
        if self._listen_thread:
            self._listen_thread.join(timeout=self._max_listen_s + 10)
        return self._result_q.pop(0) if self._result_q else None

    def unload(self):
        self.stop_listening()
        self._stt.unload()

    def warm(self) -> None:
        """Best-effort background pre-load, meant to be called once at app startup so the model is already
        resident by the time the user's first voice command finishes recording -- see _listen_loop, which
        also calls this per-utterance as a safety net in case startup warmup hasn't finished yet."""
        threading.Thread(target=self._stt.warm, daemon=True, name="stt-prewarm").start()

    def commands(self) -> Iterator[str]:
        while True:
            text = self.listen()
            if text:
                yield text


class NullVoice:
    """Placeholder used until an STT backend is installed."""
    available = False
    state = VoiceState.IDLE
    partial_transcript = ""
    last_metrics = None

    def listen(self) -> str | None:
        raise NotImplementedError("voice input is not installed; use text input")

    def start_listening(self): pass
    def stop_listening(self): return None
    def unload(self): pass
    def warm(self): pass
    def unavailable_reason(self): return "voice input not configured"


def build_voice(models_dir: str = "models", on_event: Callable | None = None) -> VoiceInput | NullVoice:
    """faster-whisper-base.en is preferred (see cua/input/stt.py docstring for the measured comparison --
    100% accuracy on the benchmark set, ~1.4s single-pass latency); falls back to Vosk if the faster-whisper
    package isn't installed."""
    mic_reason = Microphone.unavailable_reason()
    if mic_reason:
        return NullVoice()

    whisper = FasterWhisperSTT("base.en")
    if whisper.unavailable_reason() is None:
        return VoiceInput(stt=whisper, on_event=on_event)

    path = VoskSTT.find_refine_model(models_dir) or VoskSTT.find_model(models_dir)
    stt = VoskSTT(model_path=str(path) if path else None)
    if stt.unavailable_reason():
        return NullVoice()
    return VoiceInput(stt=stt, on_event=on_event)
