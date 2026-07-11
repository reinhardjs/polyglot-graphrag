"""Pluggable chunking strategies (v2.6.0 REQ-3).

Pure, dependency-free text → list[str] chunkers. No GPU / model needed, so
they are directly unit-testable (tests/test_chunking.py).

Semantics: each strategy keeps its NATURAL UNIT as one chunk
(1 sentence = 1 chunk, 1 paragraph = 1 chunk, 1 section = 1 chunk). An
oversized unit is sub-split to fit `chunk_size` (token-estimate budget). We do
NOT merge small units across boundaries — fine-grained chunks retrieve better,
and this matches the v2.5.0 one-chunk-per-sentence behaviour.

Token estimate: round(len(text) / 4) — fast proxy (~4 chars/token for EN),
adequate for chunk-size budgeting. Profiles specify chunk_size/overlap in
tokens.

`fixed` is the exception: a sliding character window of `chunk_size` tokens
with `overlap` tokens of carry-over (used when there is no natural structure).
"""
import re

# Sentence boundary splitter (keeps the delimiter attached to the sentence).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*")
# Markdown header (ATX style). Captures the full header line.
_HEADER_SPLIT = re.compile(r"(?m)^(#{1,6}\s+.*)$")
# Paragraph splitter (one or more blank lines).
_PARA_SPLIT = re.compile(r"\n\s*\n")


def _est_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def _split_long(text: str, chunk_size: int, overlap: int = 0,
                sep: str = " ") -> list:
    """Sub-split an oversized piece by `sep` into <= chunk_size-token windows.

    Used when a single natural unit (sentence/paragraph/section) exceeds the
    budget. `sep` is the granularity to cut on (space for words, newline for
    sentences).
    """
    size_chars = chunk_size * 4
    if len(text) <= size_chars:
        return [text.strip()] if text.strip() else []
    units = text.split(sep)
    chunks, cur, cur_len = [], [], 0
    for u in units:
        add = (u + sep)
        if cur and cur_len + len(add) > size_chars:
            chunks.append(sep.join(cur).strip())
            # carry overlap tail
            tail_units = cur[-max(1, (overlap * 4) // max(1, len(sep) + 8)):] if overlap else []
            cur = tail_units
            cur_len = sum(len(x) + len(sep) for x in cur)
        cur.append(u)
        cur_len += len(add)
    if cur:
        chunks.append(sep.join(cur).strip())
    return [c for c in chunks if c]


def chunk_sentence(text: str, chunk_size: int = 512, overlap: int = 64) -> list:
    """One chunk per sentence. Oversized sentence sub-split by words."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return [text.strip()] if text.strip() else []
    out = []
    for s in sents:
        if _est_tokens(s) > chunk_size:
            out.extend(_split_long(s, chunk_size, overlap, sep=" "))
        else:
            out.append(s)
    return out


def chunk_paragraph(text: str, chunk_size: int = 512, overlap: int = 64) -> list:
    """One chunk per paragraph. Oversized paragraph sub-split by sentence."""
    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    if not paras:
        return [text.strip()] if text.strip() else []
    out = []
    for p in paras:
        if _est_tokens(p) > chunk_size:
            # sub-split by sentence first, then words if still too big
            sub = chunk_sentence(p, chunk_size, overlap)
            out.extend(sub)
        else:
            out.append(p)
    return out


def chunk_section(text: str, chunk_size: int = 512, overlap: int = 64,
                  header_prefix: str = "##") -> list:
    """One chunk per markdown section (header + body). Large sections keep the
    header on each sub-chunk (sub-split by sentence)."""
    if not text.strip():
        return []
    matches = list(_HEADER_SPLIT.finditer(text))
    if not matches:
        return chunk_paragraph(text, chunk_size, overlap)
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if not block:
            continue
        header = m.group(0).strip()
        if _est_tokens(block) > chunk_size:
            body = block[len(header):].strip()
            for sub in chunk_sentence(body, chunk_size, overlap):
                sections.append(f"{header}\n\n{sub}".strip())
        else:
            sections.append(block)
    return sections


def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 64) -> list:
    """Sliding window over characters. size/overlap in tokens (×4 = chars)."""
    if not text.strip():
        return []
    size = max(1, chunk_size * 4)
    step = max(1, (chunk_size - overlap) * 4)
    chunks = []
    for i in range(0, len(text), step):
        window = text[i:i + size].strip()
        if window:
            chunks.append(window)
        if i + size >= len(text):
            break
    return chunks


# Dispatch table used by both daemons.
CHUNK_STRATEGIES = {
    "sentence": chunk_sentence,
    "paragraph": chunk_paragraph,
    "section": chunk_section,
    "fixed": chunk_fixed,
}


def chunk_text(text: str, strategy: str = "sentence",
               chunk_size: int = 512, overlap: int = 64,
               header_prefix: str = "##") -> list:
    """Dispatch to the named strategy. Unknown strategy → sentence (safe default)."""
    fn = CHUNK_STRATEGIES.get(strategy, chunk_sentence)
    if strategy == "section":
        return fn(text, chunk_size=chunk_size, overlap=overlap,
                  header_prefix=header_prefix)
    return fn(text, chunk_size=chunk_size, overlap=overlap)
