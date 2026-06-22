"""Minimal OpenAI-compatible LLM client.

Kept dependency-light (just `requests`). Any endpoint exposing
`/chat/completions` works — OpenAI, Azure OpenAI, OpenRouter, a local server.
When no API key is configured, `available` is False and callers fall back to
their heuristic paths so the pipeline still runs end to end.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from .config import Config


class LLM:
    def __init__(self, config: Config):
        self.cfg = config

    @property
    def available(self) -> bool:
        return self.cfg.has_llm

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        if not self.available:
            raise RuntimeError("No LLM configured (set OPENAI_API_KEY).")
        resp = requests.post(
            f"{self.cfg.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.cfg.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def complete_json(self, system: str, user: str) -> Any:
        return loads_lenient(self.complete(system, user))


def loads_lenient(raw: str) -> Any:
    """Parse JSON from a model response, tolerating code fences and trailing prose."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    raw = raw.strip()
    try:
        return json.JSONDecoder().raw_decode(raw)[0]
    except json.JSONDecodeError:
        start = raw.find("{")
        if start == -1:
            start = raw.find("[")
        if start >= 0:
            return json.JSONDecoder().raw_decode(raw[start:])[0]
        raise
