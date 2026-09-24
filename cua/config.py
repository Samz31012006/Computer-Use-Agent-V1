"""Runtime configuration. Defaults are tuned for the target machine (4-core i5, ~7.6 GB RAM, often <1.5 GB free).
Override with a config.json in the project root or environment variables (CUA_PLANNER=0 disables the LLM).

Model choice (evaluated, not assumed -- see bench/dynamic_eval.py and bench/results/dynamic_eval_*.json):
Qwen3-1.7B-Q4_K_M scores meaningfully better than Qwen3-0.6B-Q4_K_M on multi-step/natural-language planning
(53% vs 28% strict on bench/dynamic_corpus.py), but needs ~1.46 GB to load and this machine routinely has
only ~1.2 GB free with a browser open. Pinning it outright meant the planner refused to load on nearly every
real request, so every command the deterministic router didn't recognise failed outright -- 53% of a model
that never runs is 0%. resolve_model() therefore picks per load: the configured/largest model when there is
genuinely room for it, the smaller one when there isn't. Capability when possible, working always.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    # planner / model
    planner_enabled: bool = True          # False => deterministic router only ("planner-disabled mode")
    model_path: str | None = None
    llama_server_path: str | None = None
    models_dir: str = "models"
    chat_format: str = "auto"             # auto | chatml | qwen3 | none  (auto picks from the model file name)
    ctx_size: int = 1536
    threads: int = 4
    idle_unload_s: float = 60.0           # unload the model after this much idle time
    min_free_ram_mb: int = 400            # refuse to load if less than this would remain free
    max_new_tokens: int = 220
    prompt_variant: str = "dynamic"       # dynamic | static | zero  (chosen by bench/planner_eval.py)
    per_tool_schema: bool = True         # constrain each step to exactly one tool + its declared arguments
    confirm_llm_plans: bool = False       # True = ask before running any AI plan (risky steps always ask regardless)
    llm_retries: int = 1                  # extra attempts when the model returns invalid output
    # task budgets (hard limits: the agent can never loop)
    max_steps: int = 12
    max_replans: int = 1
    max_retries: int = 1
    task_timeout_s: float = 90.0
    log_dir: str = "logs"

    # visual grounding VLM (Milestone 9) -- a SEPARATE small model, lazy-loaded only when the OCR/CV grounding
    # layers can't resolve a target; see cua/perception/vlm.py and cua/perception/grounder.py. Evaluated:
    # SmolVLM-500M-Instruct (Apache 2.0, ~520MB total incl. mmproj) -- moondream2's only available quantized
    # GGUF options were 1.8-3.8GB, too heavy next to the already-loaded 1.7B text planner. SmolVLM loads in
    # ~1.5s and answers general questions about a screenshot correctly, but its coordinate/bounding-box output
    # is NOT reliably accurate (measured: out-of-bounds and inconsistent across repeated tries) -- so it's
    # wired in with a deliberately low confidence ceiling; the OCR+CV heuristic layers carry most of the real
    # grounding work, exactly as intended by the cascade (cheapest capable method wins).
    vlm_enabled: bool = True
    vlm_model_path: str | None = None
    vlm_mmproj_path: str | None = None
    vlm_ctx_size: int = 2048
    vlm_max_new_tokens: int = 60
    vlm_idle_unload_s: float = 60.0

    def model_ram_need_mb(self, model: Path, mmproj: Path | None = None) -> float:
        """RAM a model needs to load: weights (mmapped, so ~reclaimable) plus KV/compute buffers. One formula,
        used both to CHOOSE a model and to decide whether loading it is safe, so the two can never disagree."""
        size = model.stat().st_size + (mmproj.stat().st_size if mmproj else 0)
        return size / 2**20 * 1.1 + 150

    def model_fits(self, model: Path, free_mb: float) -> bool:
        return free_mb - self.model_ram_need_mb(model) >= self.min_free_ram_mb

    def model_candidates(self) -> list[Path]:
        """Text-planner models on disk, most capable (largest) first. Excludes the VLM's own files."""
        d = Path(self.models_dir)
        if not d.is_dir():
            return []
        vlm = {self.resolve_vlm_model(), self.resolve_vlm_mmproj()}
        return sorted((p for p in d.glob("*.gguf") if "mmproj" not in p.name.lower() and p not in vlm),
                      key=lambda p: p.stat().st_size, reverse=True)

    def resolve_model(self, free_mb: float | None = None) -> Path | None:
        """The most capable model that ACTUALLY FITS in free RAM right now.

        This machine has ~7.6 GB total and routinely only ~1.2 GB free with a browser open. A fixed choice of
        Qwen3-1.7B (needs ~1.46 GB) therefore failed to load on nearly every real request -- the planner was
        effectively dead, so every command the deterministic router didn't recognise failed outright. Picking
        per-load instead means the big model is still used whenever there's room for it, and the smaller one
        keeps the assistant working instead of failing when there isn't. A configured `model_path` is still
        preferred when it fits; when it doesn't, a smaller model that does beats refusing to plan at all.
        """
        if free_mb is None:
            import psutil
            free_mb = psutil.virtual_memory().available / 2**20

        preferred = Path(self.model_path) if self.model_path else None
        if preferred is not None and not preferred.is_file():
            preferred = None
        if preferred is not None and self.model_fits(preferred, free_mb):
            return preferred

        for cand in self.model_candidates():
            if self.model_fits(cand, free_mb):
                return cand
        # Nothing fits: return the smallest real candidate (or the configured one) so the caller's RAM check
        # reports a precise, honest "need X MB" against the cheapest option rather than a vague failure.
        candidates = self.model_candidates()
        return candidates[-1] if candidates else preferred

    def resolve_server(self) -> Path | None:
        if self.llama_server_path:
            p = Path(self.llama_server_path)
            return p if p.is_file() else None
        for cand in (Path(self.models_dir) / "llama-server.exe", Path(self.models_dir) / "llama" / "llama-server.exe"):
            if cand.is_file():
                return cand
        found = shutil.which("llama-server")
        return Path(found) if found else None

    def resolve_vlm_model(self) -> Path | None:
        if self.vlm_model_path:
            p = Path(self.vlm_model_path)
            return p if p.is_file() else None
        hits = sorted(Path(self.models_dir).glob("*[Ss]molVLM*[Ii]nstruct*.gguf"))
        hits = [p for p in hits if "mmproj" not in p.name.lower()]
        return hits[0] if hits else None

    def resolve_vlm_mmproj(self) -> Path | None:
        if self.vlm_mmproj_path:
            p = Path(self.vlm_mmproj_path)
            return p if p.is_file() else None
        hits = sorted(Path(self.models_dir).glob("*mmproj*[Ss]molVLM*.gguf"))
        return hits[0] if hits else None


def load_config(path: str | Path = "config.json") -> Config:
    cfg = Config()
    p = Path(path)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
    if os.environ.get("CUA_PLANNER") == "0":
        cfg.planner_enabled = False
    return cfg
