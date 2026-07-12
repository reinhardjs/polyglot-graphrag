"""index_router.py — V3.0 Hybrid Index-Routing extraction pipeline.

Stage 1: GLiNER detects entities + character spans (via GPU daemon
         POST /extract_entities on :8000).
Stage 2: build_route_request maps entities to indices, generates pairs
         (all_pairs or same_sentence), and packs the Index-Routing JSON.
Stage 3: query_routes sends the payload to the Qwen index-router LLM
         (:8082) which classifies relationships between entity pairs.
Stage 4: extract() orchestrates the full flow and returns
         {"entities": [...], "relations": [...]}.

No entity *types* or *relation types* are hardcoded here — they come from
domain_loader via domain_config.yaml.
"""
from __future__ import annotations

import re
import time
import requests
from typing import Dict, Any, List, Optional, Tuple

import domain_loader


# ── Stage 1: GLiNER entity extraction (via GPU daemon) ───────────────────────
def extract_entities(
    text: str,
    entity_types: List[str],
    daemon_url: str = "http://localhost:8000",
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    """Call GPU daemon /extract_entities → list of {name,type,start,end}."""
    r = requests.post(
        f"{daemon_url}/extract_entities",
        json={"text": text, "labels": entity_types},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("entities", [])


# ── Stage 2: build Index-Routing request ─────────────────────────────────────
def _sentences(text: str) -> List[str]:
    """Split text into sentences (used for same_sentence pair strategy)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def build_route_request(
    text: str,
    entities: List[Dict[str, Any]],
    relation_types: List[str],
    pair_strategy: str = "all_pairs",
) -> Dict[str, Any]:
    """Pack entities + pairs into the Index-Routing JSON schema.

    `entities` is a list of {name, type, start, end} (from GLiNER).
    Returns the payload sent to the Qwen index-router LLM.
    """
    # Assign stable indices
    indexed = []
    for i, e in enumerate(entities):
        indexed.append({
            "index": i,
            "name": e["name"],
            "type": e["type"],
        })

    n = len(indexed)
    pairs: List[Dict[str, int]] = []

    if pair_strategy == "same_sentence":
        import re as _re
        sents = _sentences(text)
        # For each sentence, find entities whose name appears as a whole word.
        # Use word-boundary matching (not naive substring) so "A" doesn't
        # match inside "separate", etc.
        for sent in sents:
            local_idx = []
            for e in indexed:
                pat = r"(?<![\w])" + _re.escape(e["name"].lower()) + r"(?![\w])"
                if _re.search(pat, sent.lower()):
                    local_idx.append(e["index"])
            for a in range(len(local_idx)):
                for b in range(a + 1, len(local_idx)):
                    pairs.append({"source": local_idx[a], "target": local_idx[b]})
    else:  # all_pairs
        for a in range(n):
            for b in range(a + 1, n):
                pairs.append({"source": a, "target": b})

    return {
        "text": text,
        "entities": indexed,
        "pairs": pairs,
        "relation_types": relation_types,
    }


# ── Stage 3: query the Qwen index-router LLM ─────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a precise relationship extraction engine for a knowledge graph. "
    "You receive a source text and a numbered list of entity pairs. "
    "For EACH pair, read the text and decide if a directed relationship of one "
    "of the ALLOWED types genuinely exists between the two entities. "
    "Determine the correct direction (source -> target) from the text meaning; "
    "you may swap the given source/target if the text implies the reverse. "
    "Only emit relationships you can justify from the text. Do not invent them. "
    "Use ONLY the allowed relation types. "
    "Respond with a single JSON object and nothing else: "
    '{"relations":[{"source":<idx>,"target":<idx>,"type":"<TYPE>","confidence":<0..1>}]}. '
    "Each source/target must be an entity index number from the list. "
    "Output JSON only, no explanation, no markdown fences."
)


def _format_pairs(payload: Dict[str, Any]) -> str:
    lines = []
    ent = {e["index"]: e for e in payload["entities"]}
    for p in payload["pairs"]:
        s = ent[p["source"]]
        t = ent[p["target"]]
        lines.append(
            f"  pair {p['source']}->{p['target']}: "
            f"({s['type']}) '{s['name']}' -> ({t['type']}) '{t['name']}'"
        )
    return "\n".join(lines)


def _salvage_objects(text: str) -> List[Dict[str, Any]]:
    """Extract complete {...} JSON objects from a possibly-truncated string.

    Used when the LLM output is cut off at max_tokens mid-array. Scans for
    balanced-brace objects and parses each independently, discarding any
    trailing incomplete fragment.
    """
    import json
    out = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i  # only track the outermost object's start
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    frag = text[start:i + 1]
                    try:
                        obj = json.loads(frag)
                        # Accept either a bare relation object or a wrapper
                        # {"relations":[...]} (recurse into its list).
                        if isinstance(obj, dict) and "relations" in obj:
                            for rel in obj.get("relations", []):
                                if isinstance(rel, dict) and "source" in rel \
                                        and "target" in rel:
                                    out.append(rel)
                        elif isinstance(obj, dict) and "source" in obj \
                                and "target" in obj:
                            out.append(obj)
                    except Exception:
                        pass
                    start = None
    return out


def _parse_relations(content: str, payload: Dict[str, Any],
                     min_confidence: float) -> List[Dict[str, Any]]:
    """Parse + validate LLM output into relation dicts."""
    import json
    cleaned = content.strip()
    if "```" in cleaned:
        import re as _re
        m = _re.search(r"```(?:json)?\s*(.*?)```", cleaned, _re.DOTALL)
        if m:
            cleaned = m.group(1).strip()

    parsed = None
    idx_obj = cleaned.find("{")
    idx_arr = cleaned.find("[")
    try:
        if idx_arr != -1 and (idx_obj == -1 or idx_arr < idx_obj):
            arr, _ = json.JSONDecoder().raw_decode(cleaned[idx_arr:])
            parsed = {"relations": arr}
        elif idx_obj != -1:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[idx_obj:])
            if "relations" in obj:
                parsed = obj
            elif "source" in obj:
                parsed = {"relations": [obj]}
            else:
                parsed = {"relations": []}
    except Exception:
        parsed = {"relations": _salvage_objects(cleaned)}
    if parsed is None:
        return []
    obj = parsed

    n_ent = len(payload["entities"])
    # Build name→index lookup so models that emit entity NAMES (not indices)
    # are supported too (model-agnostic). Matches exact name or case-insensitive.
    name_to_idx = {}
    for e in payload["entities"]:
        nm = e.get("name", "")
        if nm:
            name_to_idx[nm] = e["index"]
            name_to_idx[nm.lower()] = e["index"]
    relations = []
    seen = set()
    for rel in obj.get("relations", []):
        src = rel.get("source")
        tgt = rel.get("target")
        rtype = rel.get("type")
        try:
            conf = float(rel.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        if src is None or tgt is None or not rtype:
            continue
        # Resolve source/target: accept int index OR entity name string.
        if isinstance(src, str):
            src = name_to_idx.get(src, name_to_idx.get(src.lower(), None))
        if isinstance(tgt, str):
            tgt = name_to_idx.get(tgt, name_to_idx.get(tgt.lower(), None))
        if not isinstance(src, int) or not isinstance(tgt, int):
            continue
        if src < 0 or src >= n_ent or tgt < 0 or tgt >= n_ent:
            continue
        if src == tgt:
            continue
        if rtype not in payload["relation_types"]:
            continue  # vocabulary enforcement
        if conf < min_confidence:
            continue
        key = (src, tgt, rtype)
        if key in seen:
            continue
        seen.add(key)
        relations.append({
            "source": src, "target": tgt,
            "type": rtype, "confidence": conf,
        })
    return relations


def _classify_batch(
    payload: Dict[str, Any],
    pairs: List[Dict[str, int]],
    qwen_url: str,
    min_confidence: float,
    timeout: int,
    model: str = None,
) -> List[Dict[str, Any]]:
    """Classify a small batch of pairs (keeps the relation model focused)."""
    if model is None:
        # V3.0: relation-classifier model is config-driven (swap without code edits)
        try:
            import config as _C
            model = getattr(_C, "EXTRACTION_LLM_MODEL", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        except Exception:
            model = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    sub = {
        "text": payload["text"],
        "entities": payload["entities"],
        "pairs": pairs,
        "relation_types": payload["relation_types"],
    }
    user_msg = (
        f"Relation types allowed: {', '.join(payload['relation_types'])}\n\n"
        f"Text:\n{payload['text']}\n\n"
        f"Entity pairs to classify:\n{_format_pairs(sub)}\n\n"
        "For each listed pair, if the text supports a relationship, output it. "
        "Choose the single best relation type and correct direction. "
        "Respond with compact JSON only: "
        '{"relations":[{"source":0,"target":1,"type":"TYPE","confidence":0.9}]}'
    )
    r = requests.post(
        qwen_url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _parse_relations(content, payload, min_confidence)


def query_routes(
    payload: Dict[str, Any],
    qwen_url: str = "http://localhost:8082/v1/chat/completions",
    min_confidence: float = 0.6,
    timeout: int = 120,
    batch_size: int = 6,
) -> List[Dict[str, Any]]:
    """Send the route request to Qwen in small pair-batches, merge results.

    The 1.5B model degrades when asked to classify many pairs at once, so we
    chunk `pairs` into `batch_size` groups and merge the classified relations.
    """
    all_pairs = payload["pairs"]
    if not all_pairs:
        return []
    merged = []
    seen = set()
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i + batch_size]
        try:
            rels = _classify_batch(payload, batch, qwen_url, min_confidence, timeout)
        except Exception:
            continue
        for rel in rels:
            key = (rel["source"], rel["target"], rel["type"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(rel)
    return merged


# ── Stage 4: full orchestration ──────────────────────────────────────────────
def extract(
    text: str,
    domain_name: Optional[str] = None,
    daemon_url: str = "http://localhost:8000",
    qwen_url: str = "http://localhost:8082/v1/chat/completions",
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the full Index-Routing pipeline.

    Returns {"entities":[{name,type,start,end}],
              "relations":[{source,target,type,confidence}]}
    where relation source/target are entity *indices* (resolve via entities list).
    """
    domain = domain_loader.get_domain(domain_name)
    entity_types = domain["entity_types"]
    relation_types = domain["relation_types"]
    pair_strategy = domain.get("pair_strategy", "all_pairs")
    mc = min_confidence if min_confidence is not None else domain.get("min_confidence", 0.6)

    t0 = time.time()
    entities = extract_entities(text, entity_types, daemon_url=daemon_url)
    t_gliner = time.time() - t0

    t1 = time.time()
    payload = build_route_request(text, entities, relation_types, pair_strategy)
    relations = query_routes(payload, qwen_url=qwen_url, min_confidence=mc)
    t_qwen = time.time() - t1

    # Remap indices → names for caller convenience
    ent_by_idx = {i: e for i, e in enumerate(entities)}
    remapped = []
    for rel in relations:
        s = ent_by_idx.get(rel["source"], {})
        t = ent_by_idx.get(rel["target"], {})
        remapped.append({
            "source": rel["source"],
            "target": rel["target"],
            "source_name": s.get("name"),
            "target_name": t.get("name"),
            "type": rel["type"],
            "confidence": rel["confidence"],
        })

    return {
        "entities": entities,
        "relations": remapped,
        "meta": {
            "gliner_seconds": round(t_gliner, 2),
            "qwen_seconds": round(t_qwen, 2),
            "total_seconds": round(t_gliner + t_qwen, 2),
            "n_entities": len(entities),
            "n_pairs": len(payload["pairs"]),
            "n_relations": len(remapped),
            "pair_strategy": pair_strategy,
        },
    }
