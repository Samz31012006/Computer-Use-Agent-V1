"""Microphone capture using sounddevice. Yields 16-bit PCM chunks at 16 kHz mono.

Windows' legacy MME host API -- sounddevice/PortAudio's default input on this machine -- does not reliably
resample: forcing it to capture at 16 kHz on a device whose native rate is 44.1/48 kHz produced near-silent,
garbled audio in testing (measured RMS ~1 out of a possible ~32000, i.e. functionally dead air -- explains
voice recognition being "very bad" regardless of what was said). WASAPI captures correctly, but only at the
device's own native rate, so this module opens the stream at THAT rate and resamples down to 16 kHz in
software (stdlib `audioop.ratecv`) before handing chunks to callers. The public contract -- 16 kHz mono
int16 PCM chunks -- is unchanged, so nothing downstream (VoiceInput, VoskSTT) needs to know this happens.
"""
from __future__ import annotations

import audioop
import queue
from typing import Callable

_RATE = 16000
_CHANNELS = 1
_BLOCK_MS = 100           # small enough for VAD to confirm speech start/end within ~200ms, not ~500ms+


def _available() -> str | None:
    try:
        import sounddevice  # noqa: F401
        return None
    except (ImportError, OSError) as e:
        return str(e)


def _capture_device(preferred_rate: int) -> tuple[int | None, int]:
    """(device_index, native_rate) to actually open. Prefers WASAPI's default input device at its native
    rate (reliable); falls back to sounddevice's own default if WASAPI isn't present."""
    import sounddevice as sd
    try:
        for api in sd.query_hostapis():
            if "wasapi" in api["name"].lower() and api.get("default_input_device", -1) >= 0:
                dev = api["default_input_device"]
                return dev, int(sd.query_devices(dev)["default_samplerate"])
    except Exception:
        pass
    dev = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    try:
        return dev, int(sd.query_devices(dev)["default_samplerate"])
    except Exception:
        return dev, preferred_rate


class Microphone:
    def __init__(self, sample_rate: int = _RATE, block_ms: int = _BLOCK_MS):
        self.sample_rate = sample_rate            # public contract: chunks handed out are always this rate
        self.block_ms = block_ms                  # public: callers (VAD) need this to convert chunk counts to time
        self._block_ms = block_ms
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._stream = None
        self._running = False
        self._native_rate = sample_rate
        self._rs_state = None                      # audioop.ratecv() running conversion state
        self.device_name = ""                       # populated by start(); for instrumentation/diagnostics
        self.device_index: int | None = None
        self.channels = _CHANNELS

    @staticmethod
    def unavailable_reason() -> str | None:
        return _available()

    def start(self) -> None:
        if self._running:
            return
        import sounddevice as sd
        self._q = queue.Queue(maxsize=200)
        device, self._native_rate = _capture_device(self.sample_rate)
        self.device_index = device
        try:
            self.device_name = sd.query_devices(device)["name"]
        except Exception:
            self.device_name = f"device #{device}"
        self._rs_state = None
        block_size = max(1, int(self._native_rate * self._block_ms / 1000))
        resample = self._native_rate != self.sample_rate

        def callback(indata, frames, time_info, status):
            raw = bytes(indata)
            if resample:
                raw, self._rs_state = audioop.ratecv(raw, 2, _CHANNELS, self._native_rate, self.sample_rate,
                                                      self._rs_state)
            try:
                self._q.put_nowait(raw)
            except queue.Full:
                pass                                # drop rather than block the audio callback thread

        self._stream = sd.RawInputStream(
            samplerate=self._native_rate, blocksize=block_size,
            dtype="int16", channels=_CHANNELS, device=device, callback=callback)
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float = 0.5) -> bytes | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def running(self) -> bool:
        return self._running

    def drain(self) -> list[bytes]:
        chunks = []
        while not self._q.empty():
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return chunks
