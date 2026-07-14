"""
snomed_diagnose.py — SNOMED CT clinical diagnosis via term matching + IDF + temporal.

Optimized retrieval pipeline (v1.0):
  * IDF is precomputed once into an in-memory lookup table (_IDF_CACHE) so
    per-keyword Cypher CONTAINS queries are eliminated entirely. Query time
    is O(keywords) dict lookups instead of N Cypher round-trips.
  * Only 2 Cypher queries run per request: _symptom_to_disorder (symptom→
    disorder edge expansion) and _names_batch (id→name resolution, with
    non-English fallback in a single ordered query).
  * Neo4j driver is a module-level singleton (_DRIVER), not per-call.
  * Symptom→disorder expansion carries the SNOMED concept id so the companion
    corpus (clinical_prose) can corroborate by concept id (dual-evidence).

Entry points used by serve_gpu.py when domain == "snomed":
  1. retrieve_symptoms(query, top_k)
        Free-text symptom query (e.g. "fever and rash"). Splits on 'and'/commas,
        matches each keyword against the precomputed IDF table, scores by IDF.
  2. retrieve_temporal(presentation, top_k)
        Temporal dict {"day1":[...], ...} — each day's symptoms are matched
        independently, scores accumulated with later-day weighting, then
        multiplied by (distinct_days_covered)^1.5 for multi-day disorders.
"""
import math
import re
import threading
from neo4j import GraphDatabase

import config as C

# SNOMED relationship typeIds for true disorder→symptom edges (sparse but authoritative).
REL_EDGE_TYPES = ["246090004", "47429007", "363702006"]
EXPAND_BOOST = 2.0  # additive score per symptom→disorder expansion edge (absorbs old EDGE_BOOST)

# Later days weigh more: a disorder spanning many days gets amplified.
DAY_WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]

# ── Module-level singletons ──────────────────────────────────────────────────
_DRIVER = None
_DRIVER_LOCK = threading.Lock()

# Precomputed IDF lookup: word -> (idf: float, cids: list[str]).
# Built lazily on first use (see _ensure_idf_cache). ~20MB for the full SNOMED
# English description set; negligible relative to the 12GB GPU box.
_IDF_CACHE = None
_IDF_CACHE_LOCK = threading.Lock()


def _driver():
    """Module-level Neo4j driver singleton (no per-call construction)."""
    global _DRIVER
    if _DRIVER is None:
        with _DRIVER_LOCK:
            if _DRIVER is None:
                _DRIVER = GraphDatabase.driver(
                    C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    return _DRIVER


def _parse_keywords(text: str) -> list:
    """Split free-text symptoms into individual keywords (lowercase, stripped)."""
    parts = re.split(r"\band\b|,|;", text.lower())
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]


def _total_active(driver) -> int:
    """Count of active SnomedConcept nodes — N in IDF denominator."""
    with driver.session() as s:
        r = s.run(
            "MATCH (c:SnomedConcept {active:1}) RETURN count(c) AS cnt"
        ).single()
    return r["cnt"] if r else 1


def _ensure_idf_cache():
    """Build the in-memory IDF lookup table once, lazily, thread-safe."""
    global _IDF_CACHE
    if _IDF_CACHE is not None:
        return
    with _IDF_CACHE_LOCK:
        if _IDF_CACHE is not None:
            return
        driver = _driver()
        N = _total_active(driver)
        with driver.session() as s:
            rows = s.run(
                "MATCH (c:SnomedConcept {active:1})-[:DESCRIBED_AS]->"
                "(d:SnomedDescription {languageCode:'en'}) "
                "RETURN c.id AS cid, d.term AS term"
            ).data()
        word_to_cids: dict = {}
        for r in rows:
            cid = r["cid"]
            term = (r["term"] or "").lower()
            for w in re.findall(r"[a-z0-9]+", term):
                word_to_cids.setdefault(w, set()).add(cid)
        cache = {}
        for w, cids in word_to_cids.items():
            cache[w] = (math.log(N / len(cids)), list(cids))
        _IDF_CACHE = cache


def _idf_lookup(keyword: str):
    """Return (idf, cids) from the precomputed table. (None, None) if absent."""
    _ensure_idf_cache()
    return _IDF_CACHE.get(keyword, (0.0, None))


def _name_for(driver, concept_id: str) -> str:
    """Best English term for a concept id (preferred term, then any English, then any)."""
    with driver.session() as s:
        rec = s.run(
            "MATCH (c:SnomedConcept {id:$id})-[:DESCRIBED_AS]->"
            "(d:SnomedDescription) "
            "WHERE d.languageCode='en' "
            "RETURN d.term AS term ORDER BY d.typeId DESC LIMIT 1",
            id=concept_id,
        ).data()
        if rec:
            return rec[0]["term"]
        rec = s.run(
            "MATCH (c:SnomedConcept {id:$id})-[:DESCRIBED_AS]->"
            "(d:SnomedDescription) "
            "RETURN d.term AS term LIMIT 1",
            id=concept_id,
        ).data()
        return rec[0]["term"] if rec else concept_id


def _names_batch(driver, concept_ids: set) -> dict:
    """Return {concept_id: best_term} for a set of concept IDs.

    English preferred, but non-English rows are included as fallback in the
    SAME ordered query (ORDER BY CASE WHEN languageCode='en' THEN 0 ELSE 1 END),
    so no per-concept _name_for fallback queries are needed.
    """
    if not concept_ids:
        return {}
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:SnomedConcept)-[:DESCRIBED_AS]->"
            "(d:SnomedDescription) "
            "WHERE c.id IN $ids "
            "RETURN c.id AS id, d.term AS term, d.languageCode AS lang "
            "ORDER BY CASE WHEN d.languageCode='en' THEN 0 ELSE 1 END, "
            "d.typeId DESC",
            ids=list(concept_ids),
        ).data()
    out = {}
    for r in rows:
        if r["id"] not in out:
            out[r["id"]] = r["term"]
    return out


def _symptom_to_disorder(driver, symptom_ids: set) -> dict:
    """Expand matched SYMPTOM concepts to the DISORDERS they point to.

    Follows associated-finding edges (both directions of
    REL_246090004|47429007|363702006, plus the IS-A REL_116680003 as a weaker
    signal) from the given symptom concept ids to disorder targets, and
    returns {disorder_id: edge_count}. This is what lets a symptom query
    ("rash", "fever") surface the actual DISEASE ("Measles (disorder)") whose
    concept_id the clinical-prose article carries — enabling concept-level
    dual-evidence corroboration.
    """
    if not symptom_ids:
        return {}
    rel_names = [f"REL_{t}" for t in REL_EDGE_TYPES] + ["REL_116680003"]
    with driver.session() as s:
        # symptom -(edge)-> disorder
        fwd = s.run(
            "MATCH (s:SnomedConcept)-[r]->(d:SnomedConcept) "
            "WHERE s.id IN $ids AND type(r) IN $rel_types "
            "RETURN d.id AS disorder, count(DISTINCT s.id) AS cnt",
            ids=list(symptom_ids), rel_types=rel_names).data()
        # disorder -(edge)-> symptom  (reverse direction, same semantics)
        rev = s.run(
            "MATCH (d:SnomedConcept)-[r]->(s:SnomedConcept) "
            "WHERE s.id IN $ids AND type(r) IN $rel_types "
            "RETURN d.id AS disorder, count(DISTINCT s.id) AS cnt",
            ids=list(symptom_ids), rel_types=rel_names).data()
    out = {}
    for r in fwd + rev:
        out[r["disorder"]] = out.get(r["disorder"], 0) + r["cnt"]
    return out


def _build_context(name: str, cid: str, matched_terms: list,
                   score: float, edge_matches: int,
                   days_covered: int = 0) -> dict:
    """Assemble a human-readable diagnosis candidate context string."""
    term_strs = [f"{kw} (idf={idf:.1f})" for kw, idf in matched_terms]
    tc = len(matched_terms)
    evidence_parts = [f"term-match({tc} keywords)"]
    if edge_matches:
        evidence_parts.append(f"edges({edge_matches} matches)")
    if days_covered:
        evidence_parts.append(f"days_covered({days_covered})")
    text = (
        f"[DIAGNOSIS CANDIDATE] {name} (SNOMED {cid})\n"
        f"  Matched terms: {(', '.join(term_strs) if term_strs else '(edge match only)')}\n"
        f"  Total score: {score:.2f}\n"
        f"  Evidence: {', '.join(evidence_parts)}"
    )
    return {
        "text": text,
        "doc_id": "",
        "doc_type": "snomed",
        "chunk_idx": -1,
        "concept_id": cid,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_symptoms(query: str, top_k: int = 5) -> list:
    """Free-text symptom keywords → ranked diagnosis candidates via term-match + IDF.

    Keyword matching uses the precomputed IDF lookup table (O(1) per keyword,
    no per-keyword Cypher). Symptom→disorder edge expansion adds the actual
    disease concepts (with concept id) so the companion corpus can corroborate.
    """
    driver = _driver()
    try:
        keywords = _parse_keywords(query)

        # kw_data = [(keyword, idf, {concept_id, ...}), ...] from the IDF table.
        kw_data = []
        for kw in keywords:
            idf, cids = _idf_lookup(kw)
            if not cids or idf <= 0:
                continue
            kw_data.append((kw, idf, set(cids)))

        all_matched_cids = set()
        # disorder_id -> {"score": float, "matched_terms": [(kw, idf), ...], "edge_matches": int}
        disorder_info = {}

        for kw, idf, cids in kw_data:
            all_matched_cids.update(cids)
            for cid in cids:
                info = disorder_info.setdefault(cid, {
                    "score": 0.0,
                    "matched_terms": [],
                    "edge_matches": 0,
                })
                info["score"] += idf
                info["matched_terms"].append((kw, idf))

        # Symptom→disorder expansion: the matched concepts are mostly SYMPTOMS
        # (e.g. "rash", "fever"). Follow associated-finding edges from those
        # symptom concepts to the DISORDERS they point to, and add each disorder
        # as a candidate carrying its SNOMED concept id. Enables concept-level
        # dual-evidence corroboration with the companion corpus.
        if all_matched_cids:
            exp = _symptom_to_disorder(driver, all_matched_cids)
            for dis, cnt in exp.items():
                if dis in disorder_info:
                    # already a candidate; reinforce, don't double-count edges
                    disorder_info[dis]["edge_matches"] += cnt
                    disorder_info[dis]["score"] += EXPAND_BOOST * cnt
                else:
                    all_matched_cids.add(dis)
                    disorder_info[dis] = {
                        "score": EXPAND_BOOST * cnt,
                        "matched_terms": [],
                        "edge_matches": cnt,
                    }

        # Batch-fetch names (English-first, non-English fallback in one query).
        names = _names_batch(driver, all_matched_cids)

        # Rank by descending score.
        ranked = sorted(disorder_info.items(),
                        key=lambda kv: -kv[1]["score"])[:top_k]

        out = []
        for cid, info in ranked:
            name = names.get(cid, _name_for(driver, cid))
            out.append(_build_context(
                name, cid,
                info["matched_terms"],
                info["score"],
                info["edge_matches"],
                days_covered=0,
            ))
        return out

    finally:
        # Driver is a singleton; do NOT close it (shared across requests).
        pass


def retrieve_temporal(presentation: dict, top_k: int = 5) -> list:
    """Temporal / graduated presentation → ranked diagnosis candidates.

    `presentation` maps day label → list of free-text symptoms, e.g.
        {"day1": ["fever","fatigue"],
         "day2": ["rash","lymphadenopathy"],
         "day3": ["weight loss","oral candidiasis"]}

    Each day's symptoms are matched independently. Per-disorder scores are
    accumulated across days using DAY_WEIGHTS (later days weigh more). After
    accumulation, each disorder's score is multiplied by (distinct_days)^1.5
    to amplify multi-day (systemic) disorders.
    """
    driver = _driver()
    try:
        days = sorted(presentation.keys(),
                      key=lambda k: int(re.sub(r"\D", "", str(k)) or 0))

        # disorder_id -> {"score": float, "matched_terms": [(kw, idf, day_num), ...],
        #                 "edge_matches": int, "days_covered": set(int)}
        disorder_info = {}
        all_matched_cids = set()

        for idx, day_label in enumerate(days):
            day_weight = DAY_WEIGHTS[min(idx, len(DAY_WEIGHTS) - 1)]
            day_num = idx + 1
            symptoms = presentation[day_label]
            keywords = [s.strip().lower() for s in symptoms
                        if s and isinstance(s, str) and len(s.strip()) > 2]

            for kw in keywords:
                idf, cids = _idf_lookup(kw)
                if not cids or idf <= 0:
                    continue

                all_matched_cids.update(cids)
                add_score = idf * day_weight
                for cid in cids:
                    info = disorder_info.setdefault(cid, {
                        "score": 0.0,
                        "matched_terms": [],
                        "edge_matches": 0,
                        "days_covered": set(),
                    })
                    info["score"] += add_score
                    info["matched_terms"].append((kw, idf, day_num))
                    info["days_covered"].add(day_num)

        # Multi-day coverage boost: check each concept's descriptions for
        # keywords from ALL days. A concept whose name mentions "fever" (day 1)
        # AND "weight loss" (day 3) deserves the coverage boost even if it
        # was only directly matched on one day.
        if days and all_matched_cids:
            all_kw_by_day = {}
            for idx, day_label in enumerate(days):
                day_num = idx + 1
                all_kw_by_day[day_num] = [
                    s.strip().lower() for s in presentation[day_label]
                    if s and isinstance(s, str) and len(s.strip()) > 2
                ]

            with driver.session() as s:
                rows = s.run(
                    "MATCH (c:SnomedConcept)-[:DESCRIBED_AS]->"
                    "(d:SnomedDescription {languageCode:'en'}) "
                    "WHERE c.id IN $ids "
                    "RETURN c.id AS cid, d.term AS term",
                    ids=list(all_matched_cids),
                ).data()

            concept_terms = {}
            for r in rows:
                cid = r["cid"]
                if cid not in concept_terms:
                    concept_terms[cid] = set()
                concept_terms[cid].add(r["term"].lower())

            for cid, terms_set in concept_terms.items():
                all_text = " ".join(terms_set)
                matched_days = set()
                for dnum, kws in all_kw_by_day.items():
                    for kw in kws:
                        if kw in all_text:
                            matched_days.add(dnum)
                            break
                if cid in disorder_info:
                    disorder_info[cid]["days_covered"] = matched_days

        # Apply multi-day coverage boost: multiply score by (distinct_days)^1.5
        for info in disorder_info.values():
            nd = len(info["days_covered"])
            if nd > 1:
                info["score"] *= nd ** 1.5

        # Batch-fetch names (single ordered query with non-English fallback).
        names = _names_batch(driver, all_matched_cids)

        # Rank by descending score.
        ranked = sorted(disorder_info.items(),
                        key=lambda kv: -kv[1]["score"])[:top_k]

        out = []
        for cid, info in ranked:
            name = names.get(cid, _name_for(driver, cid))
            nd = len(info["days_covered"])
            out.append(_build_context(
                name, cid,
                [(kw, idf) for kw, idf, _ in info["matched_terms"]],
                info["score"],
                info["edge_matches"],
                days_covered=nd,
            ))
        return out

    finally:
        pass


# ── Registry contract ───────────────────────────────────────────────────────
# The domains/ registry calls `retrieve(query, top_k)` (primary) and, for
# graduated presentations, `retrieve_temporal(presentation, top_k)` which this
# module already defines above.
def retrieve(query: str, top_k: int = 5) -> list:
    """Registry entry point: free-text symptom query."""
    return retrieve_symptoms(query, top_k=top_k)
