"""Voice input tests: VAD-driven state machine, single-pass STT protocol, error handling.
No microphone or model needed."""
import time

import pytest

from cua.input.microphone import _capture_device
from cua.input.stt import FasterWhisperSTT, NullSTT, STTResult, VoskSTT
from cua.input.vad import EnergyVAD, VADConfig
from cua.input.voice import NullVoice, VoiceEvent, VoiceInput, VoiceState


# ---- fake STT ----------------------------------------------------------------
class FakeSTT:
    loaded = True

    def __init__(self, text=None, delay=0, raises=False, confidence=1.0):
        self.text = text or ""
        self.delay = delay
        self.raises = raises
        self.confidence = confidence
        self.warmed = False
        self.unloaded = False
        self.calls = 0

    def warm(self):
        self.warmed = True

    def transcribe(self, pcm16: bytes) -> STTResult | None:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise RuntimeError("boom")
        return STTResult(self.text, is_final=True, confidence=self.confidence) if self.text else None

    def unload(self):
        self.unloaded = True
        self.loaded = False

    def unavailable_reason(self):
        return None


LOUD = b"\x00\x60" * 4000      # ~0x6000 amplitude repeating -> high RMS, classified as speech
QUIET = b"\x00\x00" * 4000     # silence -> low RMS


class FakeMic:
    """Replaces Microphone for testing: yields a scripted sequence of chunks (or a default loud-then-quiet
    utterance shape) then stops. `chunks=None` uses a realistic "speak, then go silent" sequence long enough
    for the VAD's default hangover to trigger."""

    def __init__(self, chunks=None, block_ms=100):
        if chunks is None:
            chunks = [LOUD] * 4 + [QUIET] * 10       # ~400ms speech, ~1000ms silence -> hangover (700ms) fires
        self._chunks = list(chunks)
        self._i = 0
        self._running = False
        self.sample_rate = 16000
        self.device_name = "fake-mic"
        self.channels = 1
        self.block_ms = block_ms

    @staticmethod
    def unavailable_reason():
        return None

    def start(self):
        self._running = True
        self._i = 0

    def stop(self):
        self._running = False

    def read(self, timeout=0.5):
        if not self._running or self._i >= len(self._chunks):
            return None
        c = self._chunks[self._i]
        self._i += 1
        return c

    @property
    def running(self):
        return self._running

    def drain(self):
        return []


# ---- state machine tests -----------------------------------------------------
def _make(stt=None, chunks=None, **kw):
    stt = stt or FakeSTT(text="open chrome")
    kw.setdefault("min_speech_ms", 100)
    v = VoiceInput(stt=stt, max_listen_s=2.0, no_input_timeout=0.5, **kw)
    v._mic = FakeMic(chunks=chunks)
    return v, stt


def test_initial_state_is_idle():
    v, _ = _make()
    assert v.state == VoiceState.IDLE
    assert v.partial_transcript == ""


def test_listen_with_final_result():
    v, stt = _make(stt=FakeSTT(text="open chrome"))
    result = v.listen()
    assert result == "open chrome"
    assert v.state == VoiceState.IDLE
    assert stt.calls == 1


def test_listen_with_no_speech_returns_none():
    v, stt = _make(chunks=[QUIET] * 6)
    result = v.listen()
    assert result is None
    assert v.state == VoiceState.IDLE
    assert stt.calls == 0                     # never even asked to transcribe silence


def test_short_noise_blip_is_discarded_without_transcribing():
    # two loud chunks (200ms) clear the VAD's start debounce, so a click/cough DOES register as "speech
    # started" -- but the resulting span is nowhere near min_speech_ms, so it must never reach the STT backend
    v, stt = _make(chunks=[LOUD] * 2 + [QUIET] * 10, min_speech_ms=1000)
    result = v.listen()
    assert result is None
    assert stt.calls == 0


def test_events_are_emitted():
    events = []
    v, _ = _make(stt=FakeSTT(text="hi"))
    v._on_event = lambda evt, data: events.append((evt, data))
    result = v.listen()
    assert result == "hi"
    types = [e[0] for e in events]
    assert VoiceEvent.STATE_CHANGED in types
    assert VoiceEvent.FINAL_TRANSCRIPT in types
    assert VoiceEvent.METRICS in types


def test_listening_partial_event_fires_once_speech_is_detected():
    events = []
    v, _ = _make(stt=FakeSTT(text="hi"))
    v._on_event = lambda evt, data: events.append((evt, data))
    v.listen()
    partials = [d for e, d in events if e == VoiceEvent.PARTIAL_TRANSCRIPT]
    assert partials == ["Listening…"]     # no live word-by-word text any more -- just a "heard you" cue


def test_stop_listening_stops_early():
    v, stt = _make(stt=FakeSTT(text="x", delay=0.05), chunks=[LOUD] * 200)
    v.start_listening()
    time.sleep(0.3)
    v.stop_listening()
    assert v.state == VoiceState.IDLE


def test_unload_releases_stt():
    v, stt = _make()
    v.unload()
    assert stt.unloaded


def test_error_in_stt_goes_to_idle():
    events = []
    v, stt = _make(stt=FakeSTT(raises=True))
    v._on_event = lambda evt, data: events.append((evt, data))
    result = v.listen()
    assert result is None
    assert v.state == VoiceState.IDLE
    # transcribe() exceptions are caught inside _finalize and treated as "nothing heard", not a hard ERROR
    assert not any(e[0] == VoiceEvent.ERROR for e in events)


def test_warm_is_called_in_background_as_soon_as_recording_starts():
    v, stt = _make(stt=FakeSTT(text="hi"))
    v.listen()
    assert stt.warmed


def test_metrics_are_recorded_for_a_completed_utterance():
    v, stt = _make(stt=FakeSTT(text="open chrome"))
    v.listen()
    m = v.last_metrics
    assert m is not None
    assert m.device == "fake-mic" and m.sample_rate == 16000 and m.channels == 1
    assert m.raw_transcript == "open chrome" and m.final_text == "open chrome"
    assert m.stt_ms >= 0
    assert m.audio_duration_s > 0


def test_debug_capture_writes_wav_and_metrics_when_enabled(tmp_path, monkeypatch):
    import cua.input.voice as voice_mod
    monkeypatch.setattr(voice_mod, "_DEBUG_DIR", tmp_path)
    v, _ = _make(stt=FakeSTT(text="open chrome"))
    v.listen()
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json
    row = json.loads(lines[0])
    assert row["wav"] == wavs[0].name and row["raw_transcript"] == "open chrome"


def test_debug_capture_disabled_by_default_writes_nothing(tmp_path):
    v, _ = _make(stt=FakeSTT(text="open chrome"))
    v.listen()
    assert not (tmp_path / "metrics.jsonl").exists()


def test_low_confidence_transcript_is_rejected_not_executed():
    v, _ = _make(stt=FakeSTT(text="We're not violent.", confidence=0.0))
    assert v.listen() is None                       # never reaches the planner
    m = v.last_metrics
    assert m.rejected and m.final_text == ""
    assert m.raw_transcript == "We're not violent."  # but what was heard is still recorded, not hidden


def test_confident_transcript_passes_through():
    v, _ = _make(stt=FakeSTT(text="open chrome", confidence=0.27))
    assert v.listen() == "open chrome"
    assert not v.last_metrics.rejected


def test_hallucination_phrases_score_zero_confidence():
    from cua.input.stt import _confidence

    class Seg:
        no_speech_prob, avg_logprob = 0.1, -0.2
    assert _confidence([Seg()], "Thanks for watching!") == 0.0
    assert _confidence([Seg()], "open chrome") > 0.0


def test_confidence_rejects_high_no_speech_probability():
    from cua.input.stt import _confidence

    class Noise:
        no_speech_prob, avg_logprob = 0.9, -0.2
    assert _confidence([Noise()], "some invented words") == 0.0


def test_metrics_emitted_even_when_nothing_heard():
    events = []
    v, _ = _make(chunks=[QUIET] * 6)
    v._on_event = lambda evt, data: events.append((evt, data))
    v.listen()
    assert any(e[0] == VoiceEvent.METRICS for e in events)


# ---- NullVoice / NullSTT ----------------------------------------------------
def test_null_voice_raises():
    nv = NullVoice()
    assert not nv.available
    with pytest.raises(NotImplementedError):
        nv.listen()


def test_null_stt_is_inert():
    ns = NullSTT()
    assert not ns.loaded
    ns.warm()
    assert ns.transcribe(b"\x00" * 100) is None
    ns.unload()


# ---- EnergyVAD ---------------------------------------------------------------
def _rms(chunk: bytes) -> float:
    import audioop
    return audioop.rms(chunk, 2)


def test_vad_detects_speech_start_after_debounce():
    vad = EnergyVAD(VADConfig(start_confirm_chunks=2))
    assert not vad.speaking
    vad.feed(_rms(LOUD))
    assert not vad.speaking            # one loud chunk isn't enough
    vad.feed(_rms(LOUD))
    assert vad.speaking and vad.ever_spoke


def test_vad_ignores_ambient_noise_below_threshold():
    vad = EnergyVAD()
    for _ in range(10):
        ended = vad.feed(_rms(QUIET))
        assert not ended
    assert not vad.ever_spoke


def test_vad_confirms_end_after_hangover():
    cfg = VADConfig(start_confirm_chunks=2, end_hangover_chunks=3)
    vad = EnergyVAD(cfg)
    vad.feed(_rms(LOUD))
    vad.feed(_rms(LOUD))
    assert vad.speaking
    assert not vad.feed(_rms(QUIET))
    assert not vad.feed(_rms(QUIET))
    assert vad.feed(_rms(QUIET))       # third silent chunk hits the hangover
    assert vad.speech_start_idx is not None
    assert vad.end_idx >= vad.last_voice_idx


def test_vad_preroll_includes_chunks_before_confirmed_start():
    cfg = VADConfig(start_confirm_chunks=2, preroll_chunks=2)
    vad = EnergyVAD(cfg)
    vad.feed(_rms(QUIET))
    vad.feed(_rms(QUIET))
    vad.feed(_rms(LOUD))       # idx 2
    vad.feed(_rms(LOUD))       # idx 3 -> confirms speaking
    assert vad.speech_start_idx == 0       # idx 2 - preroll(2) = 0


def test_vad_does_not_let_immediate_speech_contaminate_the_noise_floor():
    # user starts talking on chunk 0 -- the floor must not be dragged up by their own voice
    cfg = VADConfig(start_confirm_chunks=2, noise_multiplier=3.5, min_threshold=120.0)
    vad = EnergyVAD(cfg)
    for _ in range(6):
        vad.feed(_rms(LOUD))
    assert vad._floor == pytest.approx(120.0)      # untouched -- LOUD chunks never qualify as "quiet"


# ---- VoskSTT unit (no model needed) ------------------------------------------
def test_vosk_find_model_returns_none_when_missing(tmp_path):
    assert VoskSTT.find_model(str(tmp_path)) is None


def test_vosk_find_model_finds_directory(tmp_path):
    (tmp_path / "vosk-model-en").mkdir()
    assert VoskSTT.find_model(str(tmp_path)) is not None


def test_find_model_prefers_small_when_requested(tmp_path):
    (tmp_path / "vosk-model-en-us-0.22-lgraph").mkdir()
    (tmp_path / "vosk-model-small-en-us-0.15").mkdir()
    assert VoskSTT.find_model(str(tmp_path), prefer_small=True).name == "vosk-model-small-en-us-0.15"
    assert VoskSTT.find_model(str(tmp_path), prefer_small=False).name == "vosk-model-en-us-0.22-lgraph"


def test_find_refine_model_skips_small_and_returns_none_if_only_small_present(tmp_path):
    (tmp_path / "vosk-model-small-en-us-0.15").mkdir()
    assert VoskSTT.find_refine_model(str(tmp_path)) is None
    (tmp_path / "vosk-model-en-us-0.22-lgraph").mkdir()
    assert VoskSTT.find_refine_model(str(tmp_path)).name == "vosk-model-en-us-0.22-lgraph"


def test_vosk_unavailable_when_no_model(tmp_path):
    stt = VoskSTT(model_path=str(tmp_path / "nope"))
    assert stt.unavailable_reason() is not None


def test_vosk_unload_sets_model_to_none():
    stt = VoskSTT()
    stt._model = "fake"
    stt.unload()
    assert stt._model is None
    assert not stt.loaded


# ---- FasterWhisperSTT (no real model download -- the transcribe backend is mocked) --------------
class FakeSeg:
    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    def __init__(self, segments_text):
        self.segments_text, self.calls = segments_text, 0

    def transcribe(self, audio, beam_size=1, language="en", vad_filter=True):
        self.calls += 1
        return [FakeSeg(t) for t in self.segments_text], None


def test_faster_whisper_reports_available_when_package_installed():
    assert FasterWhisperSTT().unavailable_reason() is None    # faster-whisper IS installed in this environment


def test_faster_whisper_defaults_to_base_en():
    assert FasterWhisperSTT()._model_size == "base.en"


def test_faster_whisper_transcribe_runs_exactly_one_pass():
    stt = FasterWhisperSTT("base.en")
    fake_model = FakeWhisperModel([" Open", " Chrome."])
    stt._model = fake_model
    r = stt.transcribe(b"\x00\x00" * 4000)
    assert r.is_final and r.text == "Open  Chrome."     # segments joined verbatim, never reworded
    assert fake_model.calls == 1


def test_faster_whisper_transcribe_empty_audio_returns_none():
    stt = FasterWhisperSTT("base.en")
    stt._model = FakeWhisperModel([""])
    assert stt.transcribe(b"") is None


def test_faster_whisper_warm_loads_model():
    stt = FasterWhisperSTT("base.en")
    stt._ensure_loaded = lambda: setattr(stt, "_model", FakeWhisperModel([]))
    stt.warm()
    assert stt.loaded


def test_faster_whisper_warm_swallows_load_errors():
    stt = FasterWhisperSTT("base.en")
    def boom():
        raise RuntimeError("no model file")
    stt._ensure_loaded = boom
    stt.warm()           # must never raise -- it runs on a background thread with nothing to catch it
    assert not stt.loaded


def test_faster_whisper_unload_clears_model():
    stt = FasterWhisperSTT()
    stt._model = FakeWhisperModel([])
    stt.unload()
    assert stt._model is None and not stt.loaded


# ---- microphone device selection ------------------------------------------------------------------
def test_capture_device_prefers_wasapi_at_its_native_rate(monkeypatch):
    import sys

    class FakeDefault:
        device = [1, 4]

    class FakeSD:
        default = FakeDefault

        @staticmethod
        def query_hostapis():
            return [{"name": "MME", "default_input_device": 1},
                   {"name": "Windows WASAPI", "default_input_device": 7}]

        @staticmethod
        def query_devices(idx):
            return {1: {"name": "Realtek Mic (MME)", "default_samplerate": 44100.0},
                   7: {"name": "Realtek Mic (WASAPI)", "default_samplerate": 48000.0}}[idx]

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD)
    device, rate = _capture_device(16000)
    assert device == 7 and rate == 48000          # WASAPI's device and ITS native rate, not MME's


def test_capture_device_falls_back_to_default_when_no_wasapi(monkeypatch):
    import sys

    class FakeDefault:
        device = [3, 4]

    class FakeSD:
        default = FakeDefault

        @staticmethod
        def query_hostapis():
            return [{"name": "MME", "default_input_device": 3}]

        @staticmethod
        def query_devices(idx):
            return {"name": "Some Mic", "default_samplerate": 44100.0}

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD)
    device, rate = _capture_device(16000)
    assert device == 3 and rate == 44100
