"""Runtime configuration, all overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("GRAPHRAG_MODEL", "gpt-4o-mini")
    hops: int = int(os.getenv("GRAPHRAG_HOPS", "2"))
    chunk_size: int = int(os.getenv("GRAPHRAG_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "150"))

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key)


def load_config() -> Config:
    return Config()
