"""GLiNER entity detection + E2B relation classification.

Single-pass hybrid extraction: GLiNER detects entities (character spans +
types), E2B classifies relationships between them. Unlike index_routing,
this does NOT use batch pairs — E2B gets the full document context with
pre-detected entities as a hint (where it excels).

Our extraction design (runs on the E2B backend): strict entity
verification, compact JSON-only output, no prose/explanation leakage.
"""

import json
import os
import re
import requests

import logging
logger = logging.getLogger("hybrid_extraction")

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
                    src_missing, tgt_missing, inferred_type=""):
    """Buffer a dropped-entity observation for the audit log.

    Called from _parse_and_validate whenever an edge references an entity
    GLiNER did not detect. Flushed by ``flush_dropped_log()`` at document end.
    ``inferred_type`` (Strategy 3) is stamped later by
    ``_stamp_inferred_types`` once the LLM fallback has classified the name.
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
                {"name": src, "side": "src",
                 "inferred_type": inferred_type, "method": ""})
        if tgt_missing and tgt:
            _dropped_buffer[key]["dropped"].append(
                {"name": tgt, "side": "tgt",
                 "inferred_type": inferred_type, "method": ""})
    except Exception:
        pass


def _stamp_inferred_types(doc_id, classified: dict) -> None:
    """Enrich buffered dropped entries with Strategy-3 inferred types.

    After the LLM fallback classifies dropped names, stamp the matched type
    and method onto the buffered audit records so the jsonl log carries the
    full trace (name -> inferred_type -> method). Idempotent.
    """
    try:
        key = doc_id or "__unknown__"
        rec = _dropped_buffer.get(key)
        if not rec:
            return
        by_name = {}
        for d in rec["dropped"]:
            by_name.setdefault(d["name"], []).append(d)
        for name, typ in classified.items():
            for d in by_name.get(name, []):
                d["inferred_type"] = typ
                d["method"] = "llm_fallback"
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


# ---------------------------------------------------------------------------
# Strategy 3: LLM Fallback NER (entity-type classification)
# ---------------------------------------------------------------------------

FALLBACK_SYSTEM_PROMPT = """You are an entity-type classifier for a GraphRAG \
system. You are given a list of entity names and a CLOSED set of allowed \
entity types. For each name, return the single best-matching allowed type. \
If no allowed type fits, return "UNKNOWN".

Rules:
1. Use ONLY types from the allowed list (or "UNKNOWN").
2. Match by semantic meaning, not string equality.
3. Return ONLY compact JSON — no prose, no markdown, no explanation.
4. Format: {"EntityName": "Type", ...}"""

FALLBACK_USER_TEMPLATE = """### ALLOWED TYPES
{types}

### ENTITY NAMES TO CLASSIFY
{names}

Return a JSON object mapping each name to its best-matching allowed type:
{{"Name1": "TypeA", "Name2": "TypeB"}}"""


def _classify_entities(names: list, domain_types: list,
                       timeout: int = 60) -> dict:
    """Classify dropped entity names into domain semantic types via E2B.

    Returns {name: inferred_type} for names that matched an allowed type.
    Names classified as UNKNOWN (or unparseable) are omitted. Never raises —
    returns {} on any failure so the caller falls back to Strategy 2 behavior.
    """
    if not names or not domain_types:
        return {}
    try:
        names_json = json.dumps(names, ensure_ascii=False)
        types_line = ", ".join(domain_types)
        user_msg = FALLBACK_USER_TEMPLATE.format(
            types=types_line, names=names_json)
        resp = _call_e2b(FALLBACK_SYSTEM_PROMPT, user_msg, timeout=timeout)
        # Parse: strip thinking block + fences, then json.loads.
        cleaned = resp.strip()
        last_tag = cleaned.rfind("<|channel|>")
        if last_tag != -1:
            cleaned = cleaned[last_tag + len("<|channel|>"):].strip()
        if "```" in cleaned:
            m = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
            if m:
                cleaned = m.group(1).strip()
        # Find the first balanced {...} object.
        start = cleaned.find("{")
        if start == -1:
            return {}
        depth, end = 0, -1
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return {}
        data = json.loads(cleaned[start:end])
        allowed = set(domain_types)
        result = {}
        for name, typ in data.items():
            if not isinstance(typ, str):
                continue
            typ = typ.strip()
            if typ in allowed:
                result[name] = typ
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("fallback classification failed: %s", e)
        return {}


def _run_extraction(text: str, domain: dict, entities: list,
                    domain_name: str = None, doc_id: str = None) -> list:
    """Run one GLiNER-enriched E2B relation-extraction pass.

    Shared by the primary path and the Strategy-3 second pass. Returns the
    validated relation list (edges only; entities already known).
    """
    entity_lines = "\n".join(
        f"- {e['name']} (type: {e.get('type', 'unknown')})"
        for e in entities
    )
    user_msg = HYBRID_USER_TEMPLATE.format(
        entity_lines=entity_lines,
        relation_types=", ".join(domain.get("relation_types", [])),
        text=text,
    )
    response = _call_e2b(SYSTEM_PROMPT, user_msg)
    return _parse_and_validate(
        response, entities, domain.get("relation_types", []),
        doc_id=doc_id, domain=domain, domain_name=domain_name,
    )


GLINER_DAEMON_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Prompts (Gemma-4-E2B Relation Extraction Engine design)
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
    """Call GLiNER /extract_entities endpoint.

    Splits `text` into word-windows of C.GLINER_CHUNK_WORDS and calls the
    endpoint per chunk, then merges entities (deduped by name+type). GLiNER
    cost grows superlinearly with input length and can hang on large docs, so
    bounding each call to a small window keeps extraction fast and prevents the
    daemon's _gliner_lock from being wedged by one huge document.
    """
    import config as C
    words = text.split()
    size = getattr(C, "GLINER_CHUNK_WORDS", 512)
    if len(words) <= size:
        chunks = [text]
    else:
        chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    merged = []
    seen = set()
    for ch in chunks:
        if not ch.strip():
            continue
        try:
            ents = _call_gliner_one(ch, entity_types, daemon_url)
        except Exception as e:
            # One chunk failing must not lose the others.
            print(f"[hybrid] GLiNER chunk failed: {e}", flush=True)
            continue
        for e in ents:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            key = (name.lower(), e.get("type"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(e)
    return merged


def _call_gliner_one(text: str, entity_types: list, daemon_url: str = None) -> list:
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
# Parsing + strict entity validation (our extraction design)
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
        # Entity verification (our extraction design) — normalize first
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

    # 2. Primary E2B relation extraction pass
    relations = _run_extraction(text, domain, entities,
                                domain_name=domain_name, doc_id=doc_id)

    # 3. Strategy 3 — LLM Fallback NER for dropped entities
    recovered = _strategy3_fallback(doc_id, text, domain, domain_name,
                                   entities, relations, provider)
    if recovered["new_entities"]:
        entities = entities + recovered["new_entities"]
        if recovered["second_pass"]:
            # Re-extract relations now that entities are known.
            extra = _run_extraction(text, domain, entities,
                                    domain_name=domain_name, doc_id=doc_id)
            # Merge de-duplicated by (src, type, tgt)
            seen_keys = {(r["source_name"], r["relation_type"], r["target_name"])
                         for r in relations}
            for r in extra:
                k = (r["source_name"], r["relation_type"], r["target_name"])
                if k not in seen_keys:
                    seen_keys.add(k)
                    relations.append(r)

    # 4. Advance dynamic-label state (promotion/eviction) + flush audit log
    provider.step_document(doc_id)
    flush_dropped_log()

    # 5. Return in extract_graph_llm-compatible shape
    return {
        "nodes": [
            {
                "id": e["name"],
                "type": e.get("type", "unknown"),
                **({"discovered_by": e["discovered_by"]}
                   if e.get("discovered_by") else {}),
            }
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


def _strategy3_fallback(doc_id, text, domain, domain_name, entities,
                       relations, provider) -> dict:
    """Recover entities E2B referenced but GLiNER missed, via E2B typing.

    Triggered only when (a) llm_fallback.enabled and (b) drops occurred.
    Calls E2B once to classify dropped names into domain semantic types,
    records them on the provider (so the TYPE is promoted by Strategy 2),
    and returns recovered entity nodes {id, type}.

    Returns:
        {"new_entities": [...], "second_pass": bool}
    """
    cfg = (domain or {}).get("llm_fallback", {}) or {}
    if not cfg:
        # llm_fallback lives at top-level of domain_config.yaml, not inside
        # each domain sub-dict. Read it via the loader's top-level accessor.
        try:
            import domain_loader
            cfg = domain_loader.get_top_level("llm_fallback") or {}
        except Exception:
            cfg = {}
    enabled = cfg.get("enabled", False)
    result = {"new_entities": [], "second_pass": False}
    if not enabled:
        return result

    # Collect dropped names from the audit buffer for this doc.
    dropped = _dropped_buffer.get(doc_id or "__unknown__", {}).get("dropped", [])
    names = sorted({d["name"] for d in dropped if d.get("name")})
    if not names:
        return result

    domain_types = list(domain.get("entity_types", []))
    classified = _classify_entities(names, domain_types)
    if not classified:
        return result

    # Stamp the inferred types + method onto the audit buffer so the jsonl
    # log carries the full trace: dropped name -> inferred_type -> method.
    _stamp_inferred_types(doc_id, classified)

    # Record on provider with inferred_type so Strategy 2 promotes the TYPE.
    for name, inferred in classified.items():
        try:
            provider.record_unknown(name, inferred_type=inferred)
        except Exception:
            pass

    # Build recovered entity nodes with correct semantic type + discovery
    # marker so write_graph can apply the :Discovered label (Option B).
    known = {e["name"].lower() for e in entities}
    new_entities = []
    for name, inferred in classified.items():
        if name.lower() not in known:
            new_entities.append({
                "name": name,
                "type": inferred,
                "discovered_by": "llm_fallback",
            })
            known.add(name.lower())
    result["new_entities"] = new_entities
    result["second_pass"] = bool(cfg.get("second_pass", True))
    logger.info("Strategy3 recovered %d entities for doc %s: %s",
                len(new_entities), doc_id,
                {e["name"]: e["type"] for e in new_entities})
    return result
