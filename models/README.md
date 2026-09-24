# models/

Not checked into git (binaries, hundreds of MB to a few GB each). Buddy runs fine with **none** of these
present — it stays on the deterministic router and skips voice — and gets progressively more capable as you
add each piece. Everything here is loaded lazily and unloaded when idle, so nothing has to fit in RAM at once.

| File / folder | What it's for | Get it from |
|---|---|---|
| `llama-server.exe` (or `llama/llama-server.exe`) | Runs the local text-planner model | [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) — grab a Windows CPU build |
| A Qwen3 `.gguf` (e.g. `Qwen3-1.7B-Q4_K_M.gguf`, `Qwen3-0.6B-Q4_K_M.gguf`) | The text planner the router falls back to | Search Hugging Face for a GGUF quantization of Qwen3; keep more than one size around if your free RAM varies (see `Config.resolve_model()`) |
| `SmolVLM-500M-Instruct-*.gguf` + matching `mmproj-*.gguf` | Last-resort visual grounding (finding things on screen OCR/heuristics can't) | Search Hugging Face for a GGUF quantization of SmolVLM-500M-Instruct, with its `mmproj` file |
| `vosk-model-small-en-us-*/`, `vosk-model-en-us-*/` | Fallback speech-to-text if `faster-whisper` isn't installed | [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) |

**Which model actually gets used, and when:**
- Text planner: whichever `.gguf` here fits current free RAM best — see `Config.resolve_model()` in
  `cua/config.py`. Point `config.json` (`model_path`) at a specific one to prefer it when it fits.
- Voice: `faster-whisper` (installed via `requirements.txt`) downloads its own model on first use and needs
  nothing placed here; Vosk is only consulted if `faster-whisper` isn't installed.
- Visual grounding: only loaded the first time OCR and the CV heuristics both fail to find a target.

None of this is required to try the deterministic path, the UI, or the test suite.
