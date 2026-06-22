"""Document loading and chunking."""
from __future__ import annotations

from pathlib import Path

from .models import Chunk


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows on paragraph/word boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        end = min(len(text), start + size)
        # try not to cut mid-word
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start += step
    return out


def chunk_text(text: str, source: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    return [
        Chunk(id=f"{source}#{i}", text=w, source=source)
        for i, w in enumerate(_windows(text, size, overlap))
    ]


def load_chunks(path: str | Path, size: int = 900, overlap: int = 150) -> list[Chunk]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return chunk_text(text, source=p.name, size=size, overlap=overlap)
