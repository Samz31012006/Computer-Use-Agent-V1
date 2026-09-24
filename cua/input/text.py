"""Text input layer: the always-available interface. Anything that yields command strings can feed the Agent."""
from __future__ import annotations

from typing import Iterator


class TextInput:
    def __init__(self, prompt: str = "cua> "):
        self.prompt = prompt

    def commands(self) -> Iterator[str]:
        while True:
            try:
                text = input(self.prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return
            if text.lower() in ("exit", "quit"):
                return
            if text:
                yield text
