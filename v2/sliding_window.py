"""Sliding-window extraction for LONG documents (Gemini 3 Pro design).

Problem with naive character slicing:
  - Splits words/entity names in half → GLiNER misses them
  - Breaks coreference ("It depends on..." in window 2 has no referent)

Solution:
  1. spaCy sentence-boundary tokenization → chunk on COMPLETE sentences
     with 1-2 sentence overlap (never splits an entity)
  2. Per chunk: GLiNER detects entities, E2B extracts relations using a
     PREVIOUS WINDOW SUMMARY to resolve coreferences
  3. A <=3-sentence summary of each chunk feeds the NEXT chunk's prompt
  4. Merge entities (dedup by name) + edges (union across windows)

Reuses _call_gliner / _call_e2b / _parse_and_validate from
hybrid_extraction to stay DRY.
"""

import re
import requests

from config import (
    EXTRACTION_LLM_BASE_URL,
    EXTRACTION_LLM_MODEL,
)

GLINER_DAEMON_URL = "http://localhost:8000"

# Lazy-load spaCy (downloads en_core_web_sm on first run if missing)
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download",
                           "en_core_web_sm"], check=True)
            import spacy
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Step 1: Sentence-boundary chunker (Gemini design)
# ---------------------------------------------------------------------------

def sentence_chunk(text: str, max_words: int = 3000,
                  overlap_sentences: int = 2) -> list:
    """Split text into sentence-boundary chunks with overlap.

    Returns list of {"text": str, "sentences": list[str], "idx": int}
    where each chunk starts AND ends on complete sentences. Overlap is in
    whole sentences so no entity name is ever split across a boundary.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    pos = 0
    while pos < len(sentences):
        word_count = 0
        end = pos
        while end < len(sentences) and word_count < max_words:
            word_count += len(sentences[end].split())
            end += 1
        # Guard: at least one sentence per chunk
        if end == pos:
            end = pos + 1

        chunk_text = " ".join(sentences[pos:end])
        chunks.append({
            "text": chunk_text,
            "sentences": sentences[pos:end],
            "idx": len(chunks),
        })

        # Slide with sentence-level overlap.
        # Always advance: if the window already reached the end, move past it;
        # otherwise step back by overlap_sentences (but never stay put).
        if end >= len(sentences):
            pos = end  # final chunk, consume rest
        else:
            new_pos = end - overlap_sentences
            pos = new_pos if new_pos > pos else end  # guarantee forward progress

    return chunks

# ---------------------------------------------------------------------------
# Step 2: E2B sliding-window prompt (Gemini design)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SLIDING = """You are a precise Relation Extraction engine for a GraphRAG system. Your task is to analyze a chunk of text (the "Current Window") and extract relationships between predefined entities.

You will also be provided with a brief summary of the immediate preceding text (the "Previous Window Summary"). Use this summary ONLY to resolve coreferences (e.g., "it", "they", "this service") in the Current Window. Do NOT extract relations that exist solely in the Previous Window Summary.

You must only extract relations between the entities provided in the PRE-DETECTED ENTITIES list."""


def build_sliding_prompt(chunk_text: str, entities: list,
                         relation_types: list, prev_summary: str = "") -> str:
    """Build the user message for E2B sliding-window extraction."""
    entity_lines = "\n".join(
        f'- {{"name": "{e["name"]}", "type": "{e.get("type", "unknown")}"}}'
        for e in entities
    )

    parts = []
    parts.append("### PRE-DETECTED ENTITIES\n" + entity_lines)

    if prev_summary:
        parts.append(
            "### PREVIOUS WINDOW SUMMARY "
            "(For Coreference Resolution Only)\n" + prev_summary
        )

    parts.append("### ALLOWED RELATION TYPES\n" + ", ".join(relation_types))
    parts.append("### CURRENT WINDOW (Extract Relations From Here)\n" + chunk_text)
    parts.append(
        "### OUTPUT FORMAT\n"
        "Output as a strict JSON array. Each object must have:\n"
        '- "source_name": exact entity name from PRE-DETECTED ENTITIES\n'
        '- "target_name": exact entity name from PRE-DETECTED ENTITIES\n'
        '- "relation_type": one of the allowed types\n\n'
        "Example:\n"
        '[{"source_name": "PR-482", "target_name": "BUG-204", '
        '"relation_type": "FIXES"}]\n\n'
        "If no relations found, output: []"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 3: Summary generation (Gemini coreference mechanism)
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """Summarize the following text block focusing on the main subjects, actors, and their ongoing actions. This summary will be used to resolve pronouns in the next iteration. Keep it under 3 sentences.

Text:
{text}
"""


def generate_summary(text: str, max_tokens: int = 256) -> str:
    """Generate a <=3 sentence summary using E2B for coref resolution."""
    resp = requests.post(
        f"{EXTRACTION_LLM_BASE_URL}/chat/completions",
        json={
            "model": EXTRACTION_LLM_MODEL,
            "messages": [{
                "role": "user",
                "content": SUMMARY_PROMPT.format(text=text),
            }],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"] or ""
    # Strip a possible <|channel|>thought block like _parse_and_validate does
    last_tag = content.rfind("<|channel|>")
    if last_tag != -1:
        content = content[last_tag + len("<|channel|>"):].strip()
    return content.strip()


# ---------------------------------------------------------------------------
# Helpers reused from hybrid_extraction (lazy import to avoid circular deps)
# ---------------------------------------------------------------------------

def _call_gliner(text: str, entity_types: list, daemon_url: str = None) -> list:
    from hybrid_extraction import _call_gliner as _g
    return _g(text, entity_types, daemon_url)


def _call_e2b(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    from hybrid_extraction import _call_e2b as _e
    return _e(system_prompt, user_prompt, timeout)


def _parse_and_validate(content: str, entities: list, valid_types: list) -> list:
    from hybrid_extraction import _parse_and_validate as _p
    return _p(content, entities, valid_types)


# ---------------------------------------------------------------------------
# Step 4: Full sliding-window orchestrator
# ---------------------------------------------------------------------------

def sliding_window_extract(text: str, domain: dict) -> dict:
    """Extract entities+edges from long documents using sentence-boundary
    chunks with coreference resolution via previous-window summaries.

    Returns write_graph-compatible dict:
        {"nodes": [...], "edges": [...]}
    so ingest.ingest_text() can feed it directly to write_graph().
    """
    if domain is None:
        import domain_loader
        domain = domain_loader.get_domain("engineering")

    chunks = sentence_chunk(text, max_words=3000, overlap_sentences=2)

    all_entities = {}
    all_edges = {}
    prev_summary = ""

    for chunk in chunks:
        chunk_text = chunk["text"]

        # 1. GLiNER on this chunk (entities with spans)
        entities = _call_gliner(chunk_text, domain.get("entity_types", []))

        # 2. Build prompt with prev_summary for coref
        prompt = build_sliding_prompt(
            chunk_text, entities, domain.get("relation_types", []),
            prev_summary,
        )

        # 3. E2B extraction
        response = _call_e2b(SYSTEM_PROMPT_SLIDING, prompt)

        # 4. Parse + validate (strict entity check)
        relations = _parse_and_validate(
            response, entities, domain.get("relation_types", [])
        )

        # 5. Merge entities (dedup by name, first occurrence wins)
        for e in entities:
            key = e["name"].lower()
            if key not in all_entities:
                all_entities[key] = e

        # 6. Merge edges (union across windows by src-type-tgt).
        #    Track ORIGINAL-case names (for write_graph resolver lookup)
        #    but dedup by lowercase to avoid case-variant duplicates.
        for rel in relations:
            src = rel["source_name"]
            tgt = rel["target_name"]
            rtype = rel["relation_type"]
            key = (src.lower(), rtype.lower(), tgt.lower())
            if key not in all_edges:
                all_edges[key] = {"source": src, "type": rtype, "target": tgt}

        # 7. Generate summary for NEXT window (Gemini coref mechanism)
        prev_summary = generate_summary(chunk_text)

    return {
        "nodes": [
            {"id": e["name"], "type": e.get("type", "unknown")}
            for e in all_entities.values()
        ],
        "edges": [
            {"source": v["source"], "type": v["type"], "target": v["target"]}
            for v in all_edges.values()
        ],
    }
