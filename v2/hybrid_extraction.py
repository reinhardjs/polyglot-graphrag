"""GLiNER entity detection + Gemma-4-E2B relation classification.

Single-pass hybrid extraction: GLiNER detects entities (character spans +
types), E2B classifies relationships between them. Unlike index_routing,
this does NOT use batch pairs — E2B gets the full document context with
pre-detected entities as a hint (where it excels).

Designed per Gemini 3 Pro sliding-window architecture: strict entity
verification, compact JSON-only output, no prose/explanation leakage.
"""

import json
import re
import requests

from config import (
    EXTRACTION_LLM_BASE_URL,
    EXTRACTION_LLM_MODEL,
)

GLINER_DAEMON_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Prompts (Gemini 3 Pro Relation Extraction Engine design)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise Relation Extraction engine for a GraphRAG \
system. Your task is to analyze a chunk of text and extract relationships \
between predefined entities.

Rules:
1. Only use entity names from the PRE-DETECTED ENTITIES list — exact string match.
2. Only use the allowed relation types.
3. Choose the single best relation type per pair.
4. Return ONLY compact JSON — no prose, no markdown, no explanation.
5. If no relations exist, return an empty array: []"""

HYBRID_USER_TEMPLATE = """### PRE-DETECTED ENTITIES
{entity_lines}

### ALLOWED RELATION TYPES
{relation_types}

### DOCUMENT
{text}

Extract all relationships between the pre-detected entities.
Return compact JSON:
{{"relations":[{{"source":"Entity1","target":"Entity2","type":"RELATION","confidence":0.95}}]}}"""


# ---------------------------------------------------------------------------
# GLiNER
# ---------------------------------------------------------------------------

def _call_gliner(text: str, entity_types: list, daemon_url: str = None) -> list:
    """Call GLiNER /extract_entities endpoint."""
    url = (daemon_url or GLINER_DAEMON_URL or "http://localhost:8000")
    resp = requests.post(
        f"{url}/extract_entities",
        json={"text": text, "entity_types": entity_types},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("entities", [])


# ---------------------------------------------------------------------------
# E2B
# ---------------------------------------------------------------------------

def _call_e2b(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    """Call Gemma-4-E2B on :8082 (or configured base URL)."""
    resp = requests.post(
        f"{EXTRACTION_LLM_BASE_URL}/chat/completions",
        json={
            "model": EXTRACTION_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Parsing + strict entity validation (Gemini recommendation)
# ---------------------------------------------------------------------------

def _parse_and_validate(content: str, entities: list, valid_types: list) -> list:
    """Parse JSON, validate against entity list + vocabulary.

    Discards any relation referencing an entity NOT in the pre-detected set
    (prevents noise from hallucinated entities). Falls back to object
    salvage if Gemma emits leading prose before the JSON.
    """
    cleaned = content.strip()
    # Strip E2B's <|channel|>thought reasoning block. The real
    # JSON output appears AFTER the closing <|channel|> tag.
    # Find the LAST occurrence of that tag; keep everything after it.
    last_tag = cleaned.rfind("<|channel|>")
    if last_tag != -1:
        cleaned = cleaned[last_tag + len("<|channel|>"):].strip()
    # Strip markdown fences
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
    # Grab LAST JSON array or object (final answer, not examples in thinking)
    arrays = [m.start() for m in re.finditer(r"\[", cleaned)]
    objs = [m.start() for m in re.finditer(r"\{", cleaned)]
    candidates = arrays + objs
    if candidates:
        start = max(candidates)  # last JSON opening
        snippet = cleaned[start:]
        try:
            if start in arrays:
                data = json.loads(snippet)
            else:
                data = json.loads(snippet)
        except (json.JSONDecodeError, ValueError):
            # Salvage individual relation objects
            from index_router import _salvage_objects
            data = {"relations": _salvage_objects(cleaned)}
    else:
        data = {"relations": []}

    if isinstance(data, list):
        rel_list = data
    elif isinstance(data, dict):
        rel_list = data.get("relations", [])
    else:
        rel_list = []

    valid_names = {e["name"] for e in entities}
    valid_types_set = set(valid_types) if valid_types else set()

    relations = []
    seen = set()
    for rel in rel_list:
        src = rel.get("source") or rel.get("source_name")
        tgt = rel.get("target") or rel.get("target_name")
        rtype = rel.get("type") or rel.get("relation_type")
        # Entity verification (Gemini recommendation)
        if src not in valid_names or tgt not in valid_names:
            continue  # discard non-matching entities to prevent noise
        if valid_types_set and rtype not in valid_types_set:
            continue
        try:
            conf = float(rel.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        key = (src.lower(), rtype.lower(), tgt.lower())
        if key in seen:
            continue
        seen.add(key)
        relations.append({
            "source_name": src,
            "target_name": tgt,
            "relation_type": rtype,
            "confidence": conf,
        })
    return relations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_hybrid(doc_id: str, text: str, domain: dict) -> dict:
    """GLiNER detects entities → E2B classifies relations.

    `domain` is the profile dict from domain_loader.get_domain()
    (same shape as extract_graph_llm's `profile` param).
    If None (TOML archived), loads from domain_config.yaml.
    Returns the same dict shape as extract_graph_llm:
        {"nodes": [...], "edges": [...]}
    so ingest.ingest_text() can feed it directly to write_graph().
    """
    # Fallback: TOML profile may be archived → load YAML domain
    if domain is None:
        import domain_loader
        domain = domain_loader.get_domain("engineering")

    # 1. GLiNER entity detection
    entities = _call_gliner(text, domain.get("entity_types", []))

    # 2. Build hybrid prompt
    entity_lines = "\n".join(
        f"- {e['name']} (type: {e.get('type', 'unknown')})"
        for e in entities
    )
    user_msg = HYBRID_USER_TEMPLATE.format(
        entity_lines=entity_lines,
        relation_types=", ".join(domain.get("relation_types", [])),
        text=text,
    )

    # 3. Single E2B call (NOT batched pairs)
    response = _call_e2b(SYSTEM_PROMPT, user_msg)

    # 4. Parse + validate
    relations = _parse_and_validate(
        response, entities, domain.get("relation_types", [])
    )

    # 5. Return in extract_graph_llm-compatible shape
    return {
        "nodes": [
            {"id": e["name"], "type": e.get("type", "unknown")}
            for e in entities
        ],
        "edges": [
            {
                "source": r["source_name"],
                "target": r["target_name"],
                "type": r["relation_type"],
            }
            for r in relations
        ],
    }
