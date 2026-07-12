"""GLiNER entity detection + Gemma-4-E2B relation classification.

Single-pass hybrid extraction: GLiNER detects entities (character spans +
types), E2B classifies relationships between them. Unlike index_routing,
this does NOT use batch pairs — E2B gets the full document context with
pre-detected entities as a hint (where it excels).

Designed per Gemini 3 Pro sliding-window architecture: strict entity
verification, compact JSON-only output, no prose/explanation leakage.
"""

import json
import os
import re
import requests

from config import (
    EXTRACTION_LLM_BASE_URL,
    EXTRACTION_LLM_MODEL,
)

# Audit log for dropped entities (Phase 1 of dynamic-label plan).
_DROPPED_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "dropped_entities.jsonl"
)

# Module-level buffer of dropped entities per doc_id, flushed at doc end.
_dropped_buffer: dict = {}


def _domain_name(domain: dict) -> str:
    """Derive the domain's short name from its config dict.

    ``domain_loader.get_domain()`` returns a sub-dict WITHOUT a ``name`` key,
    so we recover it from ``neo4j_label`` (e.g. "Journal" -> "journal") or
    ``collection`` (e.g. "journal_chunks" -> "journal"). Falls back to
    "engineering".
    """
    if not isinstance(domain, dict):
        return "engineering"
    nl = domain.get("neo4j_label")
    if nl:
        return nl.lower()
    coll = domain.get("collection", "")
    if coll.endswith("_chunks"):
        coll = coll[:-len("_chunks")]
    return coll or "engineering"


def _record_dropped(doc_id, domain_name, src, tgt,
                    src_missing, tgt_missing):
    """Buffer a dropped-entity observation for the audit log.

    Called from _parse_and_validate whenever an edge references an entity
    GLiNER did not detect. Flushed by ``flush_dropped_log()`` at document end.
    Never raises.
    """
    try:
        key = doc_id or "__unknown__"
        _dropped_buffer.setdefault(key, {
            "doc_id": doc_id,
            "domain": domain_name,
            "timestamp": __import__("time").time(),
            "dropped": [],
        })
        if src_missing and src:
            _dropped_buffer[key]["dropped"].append(
                {"name": src, "side": "src", "inferred_type": ""})
        if tgt_missing and tgt:
            _dropped_buffer[key]["dropped"].append(
                {"name": tgt, "side": "tgt", "inferred_type": ""})
    except Exception:
        pass


def flush_dropped_log():
    """Append buffered dropped-entity records to the jsonl audit log."""
    if not _dropped_buffer:
        return
    try:
        os.makedirs(os.path.dirname(_DROPPED_LOG), exist_ok=True)
        with open(_DROPPED_LOG, "a") as f:
            for rec in _dropped_buffer.values():
                if rec["dropped"]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    finally:
        _dropped_buffer.clear()


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
        json={"text": text, "labels": entity_types},
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
            "max_tokens": 4096,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Parsing + strict entity validation (Gemini recommendation)
# ---------------------------------------------------------------------------

def _parse_and_validate(content: str, entities: list, valid_types: list,
                        doc_id: str = None, domain: dict = None,
                        domain_name: str = None) -> list:
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
    # Strip markdown fences (```json ... ```)
    fenced = None
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if m:
            fenced = m.group(1).strip()
    if fenced is not None:
        # Parse the fenced content directly (it is the final JSON)
        try:
            data = json.loads(fenced)
        except (json.JSONDecodeError, ValueError):
            from index_router import _salvage_objects
            data = {"relations": _salvage_objects(fenced)}
    else:
        # No fence: find the LAST '[' that starts a COMPLETE balanced
        # JSON array (the final answer). Thinking blocks may contain
        # example '{...}' snippets, but the real output is a '[...]'.
        last_arr = cleaned.rfind("[")
        if last_arr != -1:
            # Extract balanced brackets from last_arr
            depth = 0
            end = -1
            for i in range(last_arr, len(cleaned)):
                if cleaned[i] == "[":
                    depth += 1
                elif cleaned[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                snippet = cleaned[last_arr:end]
                try:
                    data = json.loads(snippet)
                except (json.JSONDecodeError, ValueError):
                    from index_router import _salvage_objects
                    data = {"relations": _salvage_objects(cleaned)}
            else:
                from index_router import _salvage_objects
                data = {"relations": _salvage_objects(cleaned)}
        else:
            # No array at all — maybe a bare {...} object
            last_obj = cleaned.rfind("{")
            if last_obj != -1:
                snippet = cleaned[last_obj:]
                try:
                    data = json.loads(snippet)
                except (json.JSONDecodeError, ValueError):
                    from index_router import _salvage_objects
                    data = {"relations": _salvage_objects(cleaned)}
            else:
                # LAST RESORT: extract relations from E2B's thinking
                # block. Gemma emits lines like:
                #   * "Authored by: frank" -> frank AUTHORED PR-485
                # or "* X -> Y [REL]" inside its reasoning. Harvest those.
                data = _salvage_from_thinking(cleaned, entities, valid_types)

    if isinstance(data, list):
        rel_list = data
    elif isinstance(data, dict):
        rel_list = data.get("relations", [])
    else:
        rel_list = []

    valid_names = {e["name"] for e in entities}
    valid_types_set = set(valid_types) if valid_types else set()

    # Normalize for fuzzy matching: E2B often appends trailing punctuation
    # (e.g. "Zhang et al." vs GLiNER's "Zhang et al.") or
    # wraps in quotes. Strip trailing/leading punctuation + whitespace
    # and build a normalized lookup so near-matches still validate.
    import re as _re
    def _norm(s: str) -> str:
        return _re.sub(r"[\s\"'`.,;:]+$", "", (s or "").strip()).strip()

    valid_norm = {_norm(n): n for n in valid_names}

    relations = []
    seen = set()
    for rel in rel_list:
        src = rel.get("source") or rel.get("source_name")
        tgt = rel.get("target") or rel.get("target_name")
        rtype = rel.get("type") or rel.get("relation_type")
        # Entity verification (Gemini recommendation) — normalize first
        src_n = _norm(src)
        tgt_n = _norm(tgt)
        src_real = valid_norm.get(src_n)
        tgt_real = valid_norm.get(tgt_n)
        if src_real is None or tgt_real is None:
            # Record dropped entity name for dynamic-label promotion.
            try:
                from label_provider import get_provider
                if domain_name is None:
                    domain_name = _domain_name(domain)
                prov = get_provider(domain_name or "engineering")
                if src_real is None:
                    prov.record_unknown(src)
                if tgt_real is None:
                    prov.record_unknown(tgt)
                _record_dropped(doc_id, domain_name, src, tgt,
                                src_real is None, tgt_real is None)
            except Exception:
                pass  # never let label bookkeeping break extraction
            continue  # discard non-matching entities to prevent noise
        if valid_types_set and rtype not in valid_types_set:
            continue
        try:
            conf = float(rel.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        key = (src_real.lower(), rtype.lower(), tgt_real.lower())
        if key in seen:
            continue
        seen.add(key)
        relations.append({
            "source_name": src_real,
            "target_name": tgt_real,
            "relation_type": rtype,
            "confidence": conf,
        })
    return relations


def _salvage_from_thinking(content: str, entities: list,
                             valid_types: list) -> dict:
    """LAST-RESORT parser: harvest relations from Gemma's thinking block.

    When E2B emits a huge <|channel|>thought block and exhausts
    max_tokens before the final JSON, the reasoning text still
    contains correct relations as bullet examples, e.g.:
        * "Authored by: frank" -> frank AUTHORED PR-485
        * "X -> Y [REL]"  or  "frank AUTHORED PR-485"
    We regex-harvest (src, tgt, type) triples whose src/tgt are in
    the pre-detected entity set, then validate against valid_types.
    """
    valid_names = {e["name"] for e in entities}
    valid_types_set = set(valid_types) if valid_types else set()
    harvested = []
    seen = set()

    # Pattern A: "* "<phrase>" -> <SRC> <TYPE> <TGT>"
    pat_a = re.compile(
        r'->\s*([^\s"][^\n]*?)\s+([A-Z][A-Z_]+)\s+([^\s"][^\n]*)'
    )
    # Pattern B: "<SRC> <TYPE> <TGT>" (space-separated, all caps type)
    pat_b = re.compile(
        r'\b([A-Za-z0-9\-]+)\s+([A-Z][A-Z_]{2,})\s+([A-Za-z0-9\-]+)\b'
    )

    for m in pat_a.finditer(content):
        src, rtype, tgt = m.group(1).strip(), m.group(2), m.group(3).strip()
        if src in valid_names and tgt in valid_names:
            if not valid_types_set or rtype in valid_types_set:
                key = (src.lower(), rtype.lower(), tgt.lower())
                if key not in seen:
                    seen.add(key)
                    harvested.append({"source_name": src,
                                    "target_name": tgt,
                                    "relation_type": rtype})

    for m in pat_b.finditer(content):
        src, rtype, tgt = m.group(1), m.group(2), m.group(3)
        if src in valid_names and tgt in valid_names:
            if not valid_types_set or rtype in valid_types_set:
                key = (src.lower(), rtype.lower(), tgt.lower())
                if key not in seen:
                    seen.add(key)
                    harvested.append({"source_name": src,
                                    "target_name": tgt,
                                    "relation_type": rtype})

    return {"relations": harvested}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_hybrid(doc_id: str, text: str, domain: dict,
                   domain_name: str = None) -> dict:
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

    # Dynamic labels: enrich GLiNER's vocabulary with promoted candidates.
    from label_provider import get_provider
    if domain_name is None:
        domain_name = _domain_name(domain)
    provider = get_provider(domain_name)
    dynamic = provider.get_active()
    static_labels = list(domain.get("entity_types", []))
    all_labels = static_labels + dynamic

    # 1. GLiNER entity detection (enriched label set)
    entities = _call_gliner(text, all_labels)

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

    # 4. Parse + validate (records dropped entities for dynamic labels)
    relations = _parse_and_validate(
        response, entities, domain.get("relation_types", []),
        doc_id=doc_id, domain=domain, domain_name=domain_name,
    )

    # 5. Advance dynamic-label state (promotion/eviction) + flush audit log
    provider.step_document(doc_id)
    flush_dropped_log()

    # 6. Return in extract_graph_llm-compatible shape
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
