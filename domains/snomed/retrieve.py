"""
snomed_diagnose.py — SNOMED CT clinical diagnosis via term matching + IDF + temporal.

Replaces snomed_retrieve.py. Primary signal: symptom keyword CONTAINS matching
against SnomedDescription.term with IDF weighting (rare symptoms weigh more).
Secondary signal: true associated-finding edges (REL_246090004, REL_47429007,
REL_363702006) as sparse additive boosts. No IS-A hierarchy traversal.

Entry points used by serve_gpu.py when domain == "snomed":
  1. retrieve_symptoms(query, top_k)
        Free-text symptom query (e.g. "fever and rash"). Splits on 'and'/commas,
        matches each keyword against English description terms, scores by IDF.
  2. retrieve_temporal(presentation, top_k)
        Temporal dict {"day1":[...], "day2":[...], ...} — each day's symptoms
        are matched independently, scores are accumulated with later-day weighting,
        then multiplied by (distinct_days_covered)^1.5 for multi-day disorders.
"""
import math
import re
from neo4j import GraphDatabase

import config as C

# SNOMED relationship typeIds for true disorder→symptom edges (sparse but authoritative).
REL_EDGE_TYPES = ["246090004", "47429007", "363702006"]
EDGE_BOOST = 2.0  # additive score per matched edge

# Later days weigh more: a disorder spanning many days gets amplified.
DAY_WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]


def _driver():
    return GraphDatabase.driver(C.NEO4J_URI,
                                auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))


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


def _idf_for_keyword(driver, keyword: str, N: int) -> float:
    """Compute IDF = log(N / n_concepts_containing_keyword). Returns 0 if no matches."""
    with driver.session() as s:
        r = s.run(
            "MATCH (c:SnomedConcept {active:1})-[:DESCRIBED_AS]->"
            "(d:SnomedDescription {languageCode:'en'}) "
            "WHERE toLower(d.term) CONTAINS $kw "
            "RETURN count(DISTINCT c) AS cnt",
            kw=keyword.lower(),
        ).single()
    n = r["cnt"] if r else 0
    if n == 0:
        return 0.0
    return math.log(N / n)


def _concepts_for_keyword(driver, keyword: str) -> set:
    """Return set of active concept IDs whose English descriptions CONTAIN keyword."""
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:SnomedConcept {active:1})-[:DESCRIBED_AS]->"
            "(d:SnomedDescription {languageCode:'en'}) "
            "WHERE toLower(d.term) CONTAINS $kw "
            "RETURN DISTINCT c.id AS id",
            kw=keyword.lower(),
        ).data()
    return {r["id"] for r in rows}


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
    """Return {concept_id: best_English_term} for a set of concept IDs."""
    if not concept_ids:
        return {}
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:SnomedConcept)-[:DESCRIBED_AS]->"
            "(d:SnomedDescription {languageCode:'en'}) "
            "WHERE c.id IN $ids "
            "RETURN c.id AS id, d.term AS term "
            "ORDER BY d.typeId DESC",
            ids=list(concept_ids),
        ).data()
    out = {}
    for r in rows:
        if r["id"] not in out:
            out[r["id"]] = r["term"]
    return out


def _edge_boost(driver, symptom_ids: set) -> dict:
    """Query true disorder→symptom edges in reverse.

    For each disorder with an edge of type REL_246090004|47429007|363702006
    pointing TO any of the given symptom_ids, return {disorder_id: count}.
    """
    if not symptom_ids:
        return {}
    rel_names = [f"REL_{t}" for t in REL_EDGE_TYPES]
    with driver.session() as s:
        rows = s.run(
            "MATCH (d:SnomedConcept)-[r]->(s:SnomedConcept) "
            "WHERE type(r) IN $rel_types AND s.id IN $ids "
            "RETURN d.id AS disorder, count(DISTINCT s.id) AS edge_cnt",
            rel_types=rel_names,
            ids=list(symptom_ids),
        ).data()
    return {r["disorder"]: r["edge_cnt"] for r in rows}


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
    }


# ── Public API ────────────────────────────────────────────────────────────────


def retrieve_symptoms(query: str, top_k: int = 5) -> list:
    """Free-text symptom keywords → ranked diagnosis candidates via term-match + IDF.

    Primary signal: each keyword is matched via CONTAINS against
    SnomedDescription.term, scored by IDF(log(N/n)) so rare symptoms
    (e.g. 'oral candidiasis') weigh more than common ones (e.g. 'fever').

    Secondary signal: true associated-finding edges (REL_246090004, REL_47429007,
    REL_363702006) add +EDGE_BOOST per matched edge. These are sparse but authoritative.
    """
    driver = _driver()
    try:
        N = _total_active(driver)
        keywords = _parse_keywords(query)

        # For each keyword, compute IDF and find matching concept IDs.
        # kw_data = [(keyword, idf, {concept_id, ...}), ...]
        kw_data = []
        for kw in keywords:
            idf = _idf_for_keyword(driver, kw, N)
            if idf <= 0:
                continue
            cids = _concepts_for_keyword(driver, kw)
            if not cids:
                continue
            kw_data.append((kw, idf, cids))

        # Build term-match scores per concept.
        # Each concept whose name contains a keyword gets that keyword's IDF weight.
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

        # Secondary signal: edge boost for disorders connected to matched symptoms.
        if all_matched_cids:
            edge_hits = _edge_boost(driver, all_matched_cids)
            for dis, cnt in edge_hits.items():
                if dis in disorder_info:
                    disorder_info[dis]["edge_matches"] += cnt
                    disorder_info[dis]["score"] += EDGE_BOOST * cnt
                else:
                    all_matched_cids.add(dis)
                    disorder_info[dis] = {
                        "score": EDGE_BOOST * cnt,
                        "matched_terms": [],
                        "edge_matches": cnt,
                    }

        # Batch-fetch names for matching speed.
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
        driver.close()


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
        N = _total_active(driver)
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
                idf = _idf_for_keyword(driver, kw, N)
                if idf <= 0:
                    continue
                cids = _concepts_for_keyword(driver, kw)
                if not cids:
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

        # Secondary edge signal across all days' matched concepts.
        if all_matched_cids:
            edge_hits = _edge_boost(driver, all_matched_cids)
            for dis, cnt in edge_hits.items():
                if dis in disorder_info:
                    disorder_info[dis]["edge_matches"] += cnt
                    disorder_info[dis]["score"] += EDGE_BOOST * cnt
                else:
                    all_matched_cids.add(dis)
                    disorder_info[dis] = {
                        "score": EDGE_BOOST * cnt,
                        "matched_terms": [],
                        "edge_matches": cnt,
                        "days_covered": set(),
                    }

        # Multi-day coverage boost: check each concept's descriptions for
        # keywords from ALL days. A concept whose name mentions "fever" (day 1)
        # AND "weight loss" (day 3) deserves the coverage boost even if it
        # was only directly matched on one day.
        if days and all_matched_cids:
            # Gather all keywords by day
            all_kw_by_day = {}
            for idx, day_label in enumerate(days):
                day_num = idx + 1
                all_kw_by_day[day_num] = [
                    s.strip().lower() for s in presentation[day_label]
                    if s and isinstance(s, str) and len(s.strip()) > 2
                ]

            # Fetch all English descriptions for all matched concepts
            with driver.session() as s:
                rows = s.run(
                    "MATCH (c:SnomedConcept)-[:DESCRIBED_AS]->"
                    "(d:SnomedDescription {languageCode:'en'}) "
                    "WHERE c.id IN $ids "
                    "RETURN c.id AS cid, d.term AS term",
                    ids=list(all_matched_cids),
                ).data()

            # Build per-concept description text (lowercase, unique terms)
            concept_terms = {}
            for r in rows:
                cid = r["cid"]
                if cid not in concept_terms:
                    concept_terms[cid] = set()
                concept_terms[cid].add(r["term"].lower())

            # Recompute days_covered: which distinct days' keywords appear
            # in any of this concept's descriptions?
            for cid, terms_set in concept_terms.items():
                all_text = " ".join(terms_set)
                matched_days = set()
                for dnum, kws in all_kw_by_day.items():
                    for kw in kws:
                        if kw in all_text:
                            matched_days.add(dnum)
                            break  # one match per day is enough
                if cid in disorder_info:
                    disorder_info[cid]["days_covered"] = matched_days
                else:
                    pass  # edge-only disorder, no descriptions to check

        # Apply multi-day coverage boost: multiply score by (distinct_days)^1.5
        for info in disorder_info.values():
            nd = len(info["days_covered"])
            if nd > 1:
                info["score"] *= nd ** 1.5

        # Batch-fetch names.
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
        driver.close()


# ── Registry contract ───────────────────────────────────────────────────────
# The domains/ registry calls `retrieve(query, top_k)` (primary) and, for
# graduated presentations, `retrieve_temporal(presentation, top_k)` which this
# module already defines above. This thin wrapper adapts the internal
# `retrieve_symptoms` to the registry's `retrieve` name without touching the
# working internals.
def retrieve(query: str, top_k: int = 5) -> list:
    """Registry entry point: free-text symptom query."""
    return retrieve_symptoms(query, top_k=top_k)
