"""Speech-to-text backends: faster-whisper (preferred) and Vosk (fallback if faster-whisper isn't installed
or a model can't load).

Single-pass by design: VAD (cua/input/vad.py) now decides exactly when the user started and stopped talking,
so the old "re-transcribe the growing buffer every ~1.2s for a live partial, then transcribe it again in
finish_utterance(), then transcribe it a THIRD time with a bigger refine model" pipeline is gone -- it was
the dominant cost in "processing is slow" (up to N+2 full Whisper passes over an ever-growing buffer for one
utterance) and added no accuracy over one clean pass on VAD-trimmed audio. Benchmarked head to head on 8
synthetic test utterances (bench/stt_eval.py, results in bench/results/stt_eval.json):

    faster-whisper-tiny.en   96.1% avg similarity (one real miss: "Open Chrome" heard as "Open Curl"), ~0.6s
    faster-whisper-base.en  100.0% avg similarity (perfect on all 8),                                  ~1.4s
    vosk two-pass (previous default)  99.8% avg similarity,                                            ~6.5s

base.en is both the most accurate AND fast enough for a single pass on a short command, so it's now the only
model used -- not a "fast" model for live feedback plus a "refine" pass, just one accurate pass, called once,
after the VAD says the user is done talking.

The STT protocol: `warm()` best-effort pre-loads the model (called in the background as soon as recording
starts, so load time overlaps with the user talking instead of adding to perceived latency after they stop);
`transcribe()` takes one complete, already-VAD-trimmed audio buffer and returns the full result in one call.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Protocol


class STTResult:
    __slots__ = ("text", "is_final", "confidence")

    def __init__(self, text: str, is_final: bool = True, confidence: float = 1.0):
        self.text = text
        self.is_final = is_final
        self.confidence = confidence

    def __repr__(self):
        return f"STTResult({self.text!r}, final={self.is_final})"


class SpeechToText(Protocol):
    def warm(self) -> None: ...
    def transcribe(self, pcm16: bytes) -> STTResult | None: ...
    def unload(self) -> None: ...

    @property
    def loaded(self) -> bool: ...


class VoskSTT:
    """Vosk-based STT. Lazy-loads the model on first use. Kept as the fallback for machines where
    faster-whisper can't be installed; single-pass, like FasterWhisperSTT -- see module docstring."""

    def __init__(self, model_path: str | None = None, sample_rate: int = 16000,
                 idle_unload_s: float = 300.0):
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._idle_unload_s = idle_unload_s
        self._model = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._reaper: threading.Thread | None = None

    @staticmethod
    def find_model(models_dir: str = "models", prefer_small: bool = False) -> Path | None:
        d = Path(models_dir)
        candidates = sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("vosk-model")) if d.is_dir() else []
        if not candidates:
            return None
        if prefer_small:
            small = [p for p in candidates if "small" in p.name]
            if small:
                return small[0]
        return candidates[0]

    @staticmethod
    def find_refine_model(models_dir: str = "models") -> Path | None:
        """A larger, non-'small' Vosk model, if one is present -- preferred over the small model since a
        single pass no longer has a real-time deadline to keep up with."""
        d = Path(models_dir)
        candidates = sorted(p for p in d.iterdir()
                            if p.is_dir() and p.name.startswith("vosk-model") and "small" not in p.name) if d.is_dir() else []
        return candidates[0] if candidates else None

    def unavailable_reason(self) -> str | None:
        try:
            import vosk  # noqa: F401
        except ImportError:
            return "vosk package not installed"
        if self._model is not None:
            return None
        path = self._model_path or self.find_model()
        if path is None or not Path(path).is_dir():
            return "no vosk model found in models/"
        return None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import vosk
        vosk.SetLogLevel(-1)
        path = self._model_path or str(self.find_model())
        if not path or not Path(path).is_dir():
            raise RuntimeError("no vosk model found")
        self._model = vosk.Model(path)
        self._start_reaper()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warm(self) -> None:
        try:
            self._ensure_loaded()
        except Exception:
            pass

    def transcribe(self, pcm16: bytes) -> STTResult | None:
        if not pcm16:
            return None
        import vosk
        self._ensure_loaded()
        self._last_used = time.time()
        rec = vosk.KaldiRecognizer(self._model, self._sample_rate)
        rec.SetWords(True)
        rec.AcceptWaveform(pcm16)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        return STTResult(text, is_final=True) if text else None

    def unload(self) -> None:
        with self._lock:
            self._model = None

    def _start_reaper(self):
        if self._reaper and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(target=self._reap, daemon=True, name="stt-idle-reaper")
        self._reaper.start()

    def _reap(self):
        while True:
            time.sleep(5)
            with self._lock:
                if self._model is None:
                    return
                if time.time() - self._last_used > self._idle_unload_s:
                    self._model = None
                    return


class FasterWhisperSTT:
    """faster-whisper (CTranslate2) backend, single pass -- see module docstring for why. `vad_filter=True`
    uses faster-whisper's own bundled Silero VAD (onnxruntime, already a transitive dependency -- no new
    install) as a second line of defense inside the model itself: our own energy VAD (cua/input/vad.py)
    trims the audio down to the speech span before this is ever called, but a second, independent VAD pass
    catches any residual near-silence at the edges that could otherwise make Whisper hallucinate text."""

    def __init__(self, model_size: str = "base.en", sample_rate: int = 16000, idle_unload_s: float = 300.0):
        self._model_size = model_size
        self._sample_rate = sample_rate
        self._idle_unload_s = idle_unload_s
        self._model = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._reaper: threading.Thread | None = None

    def unavailable_reason(self) -> str | None:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return "faster-whisper package not installed"
        return None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        try:
            # local_files_only: once the model is cached, never let a routine load stall on a "check for a
            # newer revision" network round-trip -- measured, on this machine, to turn a normally ~1-2s CPU
            # load into a multi-minute hang when the network is slow (a direct cause of "opens late").
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8",
                                       local_files_only=True)
        except Exception:
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")   # first-ever run
        self._start_reaper()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warm(self) -> None:
        """Best-effort pre-load, meant to be called on a background thread the instant recording starts --
        by the time the user finishes an utterance a second or more later, the model is often already
        resident, so its (potentially multi-second, first-load) cost overlaps with them talking instead of
        landing on top of the "Processing..." wait after they stop."""
        try:
            self._ensure_loaded()
        except Exception:
            pass

    def transcribe(self, pcm16: bytes) -> STTResult | None:
        if not pcm16:
            return None
        self._ensure_loaded()
        self._last_used = time.time()
        import numpy as np
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, beam_size=1, language="en", vad_filter=True)
        segs = list(segments)
        text = " ".join(s.text for s in segs).strip()
        if not text:
            return None
        return STTResult(text, is_final=True, confidence=_confidence(segs, text))

    def unload(self) -> None:
        with self._lock:
            self._model = None

    def _start_reaper(self):
        if self._reaper and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(target=self._reap, daemon=True, name="whisper-idle-reaper")
        self._reaper.start()

    def _reap(self):
        while True:
            time.sleep(5)
            with self._lock:
                if self._model is None:
                    return
                if time.time() - self._last_used > self._idle_unload_s:
                    self._model = None
                    return


# Whisper-family models invent text when handed near-silence or noise. These are the specific phrases this
# pipeline actually produced from a quiet room (seen in logs/tasks.jsonl: "We're not violent.", "and not
# back.") plus the well-known YouTube-caption artefacts Whisper was trained on. Matched only as a WHOLE
# utterance, so a genuine "thank you" said as part of a real command is never discarded.
_HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching", "thanks for watching!",
    "you", "bye", "bye.", "okay", "ok", "so", "oh", ".", "...", "[blank_audio]", "[silence]",
    "subscribe", "please subscribe", "like and subscribe", "we're not violent", "and not back",
}
_NO_SPEECH_MAX = 0.6        # Whisper's own default no_speech threshold
_LOGPROB_MIN = -1.0         # ...and its default average-logprob floor


def _confidence(segments, text: str) -> float:
    """0..1 confidence for a finished transcript, from the model's own per-segment signals.

    Without this the pipeline treated a confident "open chrome" and a hallucinated "We're not violent."
    (produced from a silent room) as equally valid and executed both -- see cua/input/voice.py, which now
    refuses to act below a floor rather than running whatever the model dreamed up."""
    if not segments:
        return 0.0
    if " ".join(text.lower().split()).strip(" .!?,") in _HALLUCINATIONS:
        return 0.0
    n = len(segments)
    no_speech = sum(getattr(s, "no_speech_prob", 0.0) or 0.0 for s in segments) / n
    logprob = sum(getattr(s, "avg_logprob", 0.0) or 0.0 for s in segments) / n
    if no_speech > _NO_SPEECH_MAX or logprob < _LOGPROB_MIN:
        return 0.0
    # map avg_logprob (0 = perfect, -1 = the floor) onto 0..1
    return max(0.0, min(1.0, 1.0 + logprob))


class NullSTT:
    """Placeholder when no STT backend is available."""
    loaded = False

    def warm(self): pass
    def transcribe(self, pcm16: bytes): return None
    def unload(self): pass
    def unavailable_reason(self): return "no STT backend configured"
