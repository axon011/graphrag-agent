"""LLM client with pluggable providers.

Providers (set ``GRAPHRAG_LLM``; default ``auto``):
  - ``api``    — any OpenAI-compatible HTTP endpoint (uses ``OPENAI_API_KEY``)
  - ``codex``  — the Codex CLI (``codex exec``), via your subscription
  - ``claude`` — the Claude Code CLI (``claude -p``)
  - ``gemini`` — the Gemini CLI (``gemini -p``)
  - ``off``    — no LLM; callers fall back to their heuristic path

``auto`` picks the first available: api key, then codex, claude, gemini, else off.
This keeps the agent runnable with a local CLI subscription and no API key.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

import requests

from .config import Config

_GUARD = (
    "You are a text-generation API, not an interactive agent. Do not run shell "
    "commands, do not read or write files, do not explain. Output ONLY the "
    "requested content as your final message.\n\n"
)


class LLM:
    def __init__(self, config: Config):
        self.cfg = config
        self.provider = self._resolve()

    def _resolve(self) -> str:
        p = (os.getenv("GRAPHRAG_LLM", "auto") or "auto").lower()
        if p != "auto":
            return p
        if self.cfg.api_key:
            return "api"
        for cli in ("codex", "claude", "gemini"):
            if shutil.which(cli):
                return cli
        return "off"

    @property
    def available(self) -> bool:
        return self.provider not in ("off", "")

    # ---- public ----
    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        if self.provider == "api":
            return self._http(system, user, temperature)
        if self.provider in ("codex", "claude", "gemini"):
            return self._cli(f"{system}\n\n{user}")
        raise RuntimeError("No LLM available (set GRAPHRAG_LLM or OPENAI_API_KEY).")

    def complete_json(self, system: str, user: str) -> Any:
        return loads_lenient(self.complete(system, user))

    # ---- backends ----
    def _http(self, system: str, user: str, temperature: float) -> str:
        resp = requests.post(
            f"{self.cfg.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.cfg.model, "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _cli(self, prompt: str) -> str:
        if self.provider == "codex":
            return self._codex(prompt)
        if self.provider == "claude":
            return self._claude(prompt)
        return self._gemini(prompt)

    def _codex(self, prompt: str) -> str:
        exe = shutil.which("codex") or "codex"
        fd, out = tempfile.mkstemp(suffix=".txt", prefix="grag_")
        os.close(fd)
        try:
            cmd = [exe, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
                   "--ephemeral", "--color", "never", "-o", out, "-"]
            r = subprocess.run(cmd, input=_GUARD + prompt, capture_output=True,
                               text=True, timeout=240, encoding="utf-8")
            txt = ""
            try:
                with open(out, encoding="utf-8") as fh:
                    txt = fh.read().strip()
            except OSError:
                pass
            if not txt:
                raise RuntimeError(f"codex returned nothing ({(r.stderr or '')[:200]})")
            return txt
        finally:
            try:
                os.remove(out)
            except OSError:
                pass

    def _claude(self, prompt: str) -> str:
        exe = shutil.which("claude") or "claude"
        r = subprocess.run(
            [exe, "-p", "--output-format", "json", "--strict-mcp-config",
             "--mcp-config", '{"mcpServers":{}}'],
            input=_GUARD + prompt, capture_output=True, text=True,
            timeout=240, encoding="utf-8",
        )
        if r.returncode != 0:
            raise RuntimeError(f"claude exited {r.returncode}")
        env = json.loads(r.stdout)
        text = env.get("result", "")
        if not text.strip():
            raise RuntimeError("claude returned empty result")
        return text

    def _gemini(self, prompt: str) -> str:
        exe = shutil.which("gemini") or "gemini"
        r = subprocess.run([exe, "-p", _GUARD + prompt], capture_output=True,
                           text=True, timeout=240, encoding="utf-8")
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(f"gemini failed ({(r.stderr or '')[:200]})")
        return r.stdout.strip()


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
        for ch in ("{", "["):
            i = raw.find(ch)
            if i >= 0:
                return json.JSONDecoder().raw_decode(raw[i:])[0]
        raise
