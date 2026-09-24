"""Compare STT candidates on the synthetic speech samples in bench/speech_samples/ (generate them first with
scripts/gen_test_speech.py). Measures latency, RAM, and transcript accuracy against the known ground truth --
no human needs to re-speak the same sentences for every candidate.

  python -m bench.stt_eval
"""
from __future__ import annotations

import json
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path

import psutil

SENTENCES = {
    "open_chrome": "Open Chrome.",
    "open_youtube": "Open YouTube.",
    "youtube_search": "Open YouTube and search for SpaceX.",
    "research_spacex": "I want to research the latest SpaceX launch.",
    "click_first_channel": "Click the first channel.",
    "click_like": "Click the Like button.",
    "click_search_icon": "Click the search icon.",
    "conversational": "Hey, can you help me find that document I was working on yesterday?",
}
SAMPLES_DIR = Path("bench/speech_samples")


def _read_wav_pcm16(path: Path) -> bytes:
    """16kHz mono 16-bit PCM, resampling/downmixing if the file isn't already in that format."""
    import audioop
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        data = audioop.lin2lin(data, width, 2)
    if channels == 2:
        data = audioop.tomono(data, 2, 0.5, 0.5)
    if rate != 16000:
        data, _ = audioop.ratecv(data, 2, 1, rate, 16000, None)
    return data


def _similarity(a: str, b: str) -> float:
    norm = lambda s: "".join(c.lower() for c in s if c.isalnum() or c.isspace()).split()
    return SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()


def eval_vosk_single_pass() -> dict:
    """Vosk, single pass -- the fallback backend now uses the same single-transcribe()-call design as
    faster-whisper (see cua/input/stt.py); prefers the larger non-'small' model since there's no live
    real-time deadline to keep up with any more."""
    from cua.input.stt import VoskSTT
    path = VoskSTT.find_refine_model("models") or VoskSTT.find_model("models")
    stt = VoskSTT(model_path=str(path) if path else None)

    rows = []
    proc = psutil.Process()
    before = proc.memory_info().rss / 2**20
    for name, truth in SENTENCES.items():
        pcm = _read_wav_pcm16(SAMPLES_DIR / f"{name}.wav")
        t0 = time.perf_counter()
        result = stt.transcribe(pcm)
        ms = (time.perf_counter() - t0) * 1000
        text = result.text if result else ""
        rows.append({"case": name, "truth": truth, "raw": text, "final": text,
                    "similarity": round(_similarity(truth, text), 3), "latency_ms": round(ms)})
    after = proc.memory_info().rss / 2**20
    stt.unload()
    return {"name": "vosk_single_pass", "rows": rows, "rss_delta_mb": round(after - before)}


def eval_faster_whisper(model_size: str) -> dict | None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    proc = psutil.Process()
    before = proc.memory_info().rss / 2**20
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    load_ms = (time.perf_counter() - t0) * 1000

    rows = []
    for name, truth in SENTENCES.items():
        t0 = time.perf_counter()
        segments, info = model.transcribe(str(SAMPLES_DIR / f"{name}.wav"), beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"case": name, "truth": truth, "raw": text, "final": text,
                    "similarity": round(_similarity(truth, text), 3), "latency_ms": round(ms)})
    after = proc.memory_info().rss / 2**20
    return {"name": f"faster-whisper-{model_size}", "rows": rows, "rss_delta_mb": round(after - before),
           "load_ms": round(load_ms)}


def summarize(result: dict) -> dict:
    sims = [r["similarity"] for r in result["rows"]]
    lats = [r["latency_ms"] for r in result["rows"]]
    return {"name": result["name"], "avg_similarity": round(sum(sims) / len(sims), 3),
           "min_similarity": round(min(sims), 3), "avg_latency_ms": round(sum(lats) / len(lats)),
           "max_latency_ms": max(lats), "rss_delta_mb": result.get("rss_delta_mb"),
           "load_ms": result.get("load_ms")}


def main():
    if not SAMPLES_DIR.is_dir() or not list(SAMPLES_DIR.glob("*.wav")):
        print("no samples found; run: python scripts/gen_test_speech.py")
        return

    results = [eval_vosk_single_pass()]
    for size in ("tiny.en", "base.en"):
        r = eval_faster_whisper(size)
        if r:
            results.append(r)
        else:
            print(f"faster-whisper not installed; skipping {size}")

    for r in results:
        print(f"\n=== {r['name']} ===")
        for row in r["rows"]:
            print(f"  [{row['similarity']:.2f}] {row['latency_ms']:>5}ms  truth={row['truth']!r}")
            print(f"         heard={row['final']!r}")
        s = summarize(r)
        print(f"  -> avg_similarity={s['avg_similarity']} min={s['min_similarity']} "
             f"avg_latency={s['avg_latency_ms']}ms max={s['max_latency_ms']}ms "
             f"rss_delta={s['rss_delta_mb']}MB load={s.get('load_ms')}ms")

    out = Path("bench/results/stt_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "summaries": [summarize(r) for r in results]}, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
