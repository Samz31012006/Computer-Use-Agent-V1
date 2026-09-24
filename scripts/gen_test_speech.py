"""Generate synthetic test speech (Windows SAPI TTS) for objective, reproducible STT benchmarking --
no need for a human to speak the same sentences repeatedly for every candidate model."""
import os
import win32com.client

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

OUT_DIR = "bench/speech_samples"
os.makedirs(OUT_DIR, exist_ok=True)

sp = win32com.client.Dispatch("SAPI.SpVoice")
fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
fmt.Type = 34          # SAFT16kHz16BitMono -- matches what Vosk/faster-whisper expect
stream = win32com.client.Dispatch("SAPI.SpFileStream")

for name, text in SENTENCES.items():
    path = os.path.abspath(os.path.join(OUT_DIR, f"{name}.wav"))
    stream.Format = fmt
    stream.Open(path, 3, False)     # 3 = SSFMCreateForWrite
    sp.AudioOutputStream = stream
    sp.Speak(text)
    stream.Close()
    print(f"{name}: {text!r} -> {path}")
