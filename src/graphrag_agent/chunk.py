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
    if "#" in source:
        raise ValueError(
            f"source must not contain '#': {source!r}. Chunk ids are '{{source}}#{{index}}', "
            "so a '#' in the source makes the document unrecoverable from the id."
        )
    return [
        Chunk(id=f"{source}#{i}", text=w, source=source)
        for i, w in enumerate(_windows(text, size, overlap))
    ]


def source_of(chunk_id: str) -> str:
    """Recover the source document from a chunk id.

    The inverse of the id built in `chunk_text`. Consumers use this to map an
    entity's `mentions` (chunk ids) back to the documents that mention it —
    see graphrag-studio's document layer.
    """
    return chunk_id.rsplit("#", 1)[0]


def relative_source(path: str | Path, root: str | Path | None = None) -> str:
    """The `source` label for a file: repo-relative path, forward slashes.

    Falls back to the bare filename when `root` is None or the path lies outside
    it. Note that filenames are NOT unique in a real corpus — a tree with several
    `README.md` files collapses them into one source, and anything keying on
    `source` as a document identity is then silently wrong. Pass `root`.
    """
    p = Path(path)
    if root is not None:
        try:
            return p.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass  # outside root — fall through to the basename
    return p.name


def load_chunks(
    path: str | Path,
    size: int = 900,
    overlap: int = 150,
    root: str | Path | None = None,
) -> list[Chunk]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return chunk_text(text, source=relative_source(p, root), size=size, overlap=overlap)
