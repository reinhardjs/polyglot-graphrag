"""
snomed_retrieve.py — SNOMED CT clinical retrieval for the /ask pipeline.

SNOMED is a terminology graph, not a prose corpus, so it is retrieved by
CYPHER TRAVERSAL, not by Qdrant vector search. This module provides two
entry points used by serve_gpu.py when `domain == "snomed"`:

  1. retrieve_symptoms(query, top_k)
        Free-text symptom query (e.g. "chest pain and shortness of breath").
        Matches the text against :SnomedDescription terms, resolves to
        :SnomedConcept, then reverses REL_116680003 (associated finding) to
        surface candidate disorders. Returns ranked diagnosis contexts.

  2. retrieve_temporal(presentation, top_k)
        Graduated / temporal presentation as a dict of day -> symptom list,
        e.g. {"day1": ["fever","fatigue"], "day2": ["rash","lymphadenopathy"],
              "day3": ["weight loss","oral candidiasis"]}.
        Each day's symptoms are matched to concepts, the symptom set is
        accumulated over time, and disorders are scored by how many of their
        associated findings overlap the accumulated symptom set — with LATER
        days weighted more heavily (a progressive illness reveals its signature
        over time). This models the clinic scenario where day-1 is vague and
        the diagnosis only becomes clear as the picture evolves.

Return shape mirrors ask.neo4j_subgraph so the existing rerank + E4B
synthesis path is unchanged: a list of
    {"text": <profile string>, "doc_id": "", "doc_type": "", "chunk_idx": -1}

All graph access is read-only against Neo4j. SNOMED ids are opaque; we always
carry the FSN/term alongside so the synthesized answer is human-readable.
"""
import re
from neo4j import GraphDatabase

import config as C

# SNOMED relationship typeId for "Associated finding" — the diagnosis edge.
REL_ASSOCIATED_FINDING = "116680003"

# Later days weigh more: a disorder that explains a day-3 symptom scores higher
# than one that only explains a day-1 symptom (progressive revelation).
DAY_WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]  # index by (day-1), capped


def _driver():
    return GraphDatabase.driver(C.NEO4J_URI,
                                auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))


def _term_for(driver, concept_id: str) -> str:
    """Best English FSN/term for a concept id (or German fallback)."""
    with driver.session() as s:
        rec = s.run(
            "MATCH (c:SnomedConcept {id:$id})-[:DESCRIBED_AS]->"
            "(d:SnomedDescription) "
            "WHERE d.languageCode='en' "
            "RETURN d.term AS term ORDER BY d.typeId DESC LIMIT 1",
            id=concept_id).data()
        if rec:
            return rec[0]["term"]
        rec = s.run(
            "MATCH (c:SnomedConcept {id:$id})-[:DESCRIBED_AS]->"
            "(d:SnomedDescription) "
            "RETURN d.term AS term LIMIT 1",
            id=concept_id).data()
        return rec[0]["term"] if rec else concept_id


def match_concepts(driver, phrase: str, limit: int = 20) -> list:
    """Match a free-text symptom phrase to SNOMED concepts via term CONTAINS.

    The query may list several symptoms joined by 'and' / commas
    (e.g. "chest pain and shortness of breath"). We split into per-symptom
    chunks and match each independently (AND within a chunk), then union — so
    "chest pain" finds chest-pain findings and "shortness of breath" finds
    dyspnea, without requiring one term to contain every word of the query.
    Returns list of {"id":..., "term":...} ranked by phrase-exactness.
    """
    # split into symptom chunks
    chunks = re.split(r"\band\b|,|;", phrase.lower())
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        chunks = [phrase]

    seen = {}
    for chunk in chunks:
        words = [w for w in re.findall(r"[a-z][a-z0-9]+", chunk) if len(w) > 2]
        if not words:
            continue
        # phrase boost variants within this chunk
        phrase_variants = {" ".join(words)}
        for n in (3, 2):
            for i in range(len(words) - n + 1):
                phrase_variants.add(" ".join(words[i:i + n]))
        if len(words) >= 2:
            where = " AND ".join([f"toLower(d.term) CONTAINS $w{i}" for i in range(len(words))])
        else:
            where = "toLower(d.term) CONTAINS $w0"
        with driver.session() as s:
            rows = s.run(
                "MATCH (c:SnomedConcept)-[:DESCRIBED_AS]->(d:SnomedDescription {languageCode:'en'}) "
                f"WHERE {where} "
                "RETURN DISTINCT c.id AS id, d.term AS term LIMIT 200",
                **{f"w{i}": w for i, w in enumerate(words)}).data()
        for r in rows:
            cid = r["id"]
            if cid in seen:
                continue
            term_l = (r["term"] or "").lower()
            overlap = sum(1 for w in words if w in term_l)
            # Prefer the canonical symptom concept: term equals the chunk, or
            # starts with it (e.g. "Fever" over "Sandfly fever", "Chest pain"
            # over "Atypical chest pain").
            exact = 5 if (term_l.strip() == chunk or term_l.strip().startswith(chunk + " ")) else 0
            phrase_boost = max((3 if pv in term_l else 0) for pv in phrase_variants)
            seen[cid] = (exact, phrase_boost, overlap, -len(r["term"]), r["term"])
    # rank: exact desc, phrase_boost desc, overlap desc, shorter term first
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1][0], -kv[1][1], -kv[1][2], kv[1][3]))
    return [{"id": cid, "term": meta[4]} for cid, meta in ranked[:limit]]


def _ancestors(driver, concept_id: str, max_hops: int = 2) -> set:
    """Collect is-a / associated-finding ancestors (up REL_116680003)."""
    out = set()
    with driver.session() as s:
        rows = s.run(
            f"MATCH (c:SnomedConcept {{id:$id}})-[:REL_{REL_ASSOCIATED_FINDING}*1..{max_hops}]->(a:SnomedConcept) "
            "RETURN DISTINCT a.id AS aid",
            id=concept_id).data()
    for r in rows:
        out.add(r["aid"])
    return out


def _disorders_scored(driver, concept_ids: set, weight: float = 1.0) -> dict:
    """Reverse REL_116680003 for a SET of concept ids; return
    {disorder_id: (weighted_score, set_of_matched_concept_ids)}."""
    if not concept_ids:
        return {}
    with driver.session() as s:
        rows = s.run(
            f"MATCH (d:SnomedConcept)-[:REL_{REL_ASSOCIATED_FINDING}]->(s:SnomedConcept) "
            "WHERE s.id IN $ids "
            "RETURN d.id AS dis, collect(s.id) AS syms",
            ids=list(concept_ids)).data()
    out = {}
    for r in rows:
        dis = r["dis"]
        matched = set(r["syms"]) & concept_ids
        out[dis] = (out.get(dis, (0.0, set()))[0] + weight * len(matched),
                    out.get(dis, (0.0, set()))[1] | matched)
    return out


def _build_context(driver, disorder_id: str, matched_symptoms: list,
                   score: float) -> dict:
    """Assemble a human-readable diagnosis context string."""
    name = _term_for(driver, disorder_id)
    sym_terms = [f"{t}" for t in matched_symptoms]
    text = (f"[DIAGNOSIS CANDIDATE] {name} (SNOMED {disorder_id})\n"
            f"  Supporting findings: {', '.join(sym_terms)}\n"
            f"  Match score: {score:.2f}")
    return {"text": text, "doc_id": "", "doc_type": "snomed", "chunk_idx": -1}


# ── Public API ────────────────────────────────────────────────────────────────
def _symptom_concept_set(driver, chunk: str, exact_weight=1.0, ancestor_weight=0.5):
    """Match a symptom chunk (which may list several symptoms separated by
    spaces / 'and' / commas) to SNOMED concepts, expand each to ancestors, and
    return {concept_id: weight} where exact matches weigh more than ancestors.
    """
    # split into individual symptom phrases (only on 'and' / commas — NOT
    # spaces, so "chest pain" stays one phrase matched via AND(chest, pain)).
    parts = re.split(r"\band\b|,|;", chunk.lower())
    parts = [p.strip() for p in parts if p.strip() and len(p) > 2]
    out = {}
    for part in parts:
        syms = match_concepts(driver, part, limit=15)
        for s in syms:
            cid = s["id"]
            out[cid] = out.get(cid, 0.0) + exact_weight
            for anc in _ancestors(driver, cid, max_hops=1):
                out[anc] = out.get(anc, 0.0) + ancestor_weight
    return out  # {concept_id: weight}


def retrieve_symptoms(query: str, top_k: int = 5) -> list:
    """Free-text symptom -> ranked diagnosis candidates.

    Splits the query into symptom chunks, matches each to SNOMED concepts
    (expanding to ancestors for recall), reverses REL_116680003 to disorders,
    and ranks disorders by how many distinct symptoms they explain (a disorder
    explaining chest pain AND shortness of breath outranks one explaining only
    one).
    """
    driver = _driver()
    try:
        chunks = re.split(r"\band\b|,|;", query.lower())
        chunks = [c.strip() for c in chunks if c.strip()]
        if not chunks:
            chunks = [query]
        sym_term = {}
        disorder_scores = {}   # disorder_id -> score
        disorder_syms = {}     # disorder_id -> set(concept_id explained)
        for chunk in chunks:
            cset = _symptom_concept_set(driver, chunk)
            if not cset:
                continue
            ids = set(cset)
            for cid in ids:
                t = _term_for(driver, cid)
                if t:
                    sym_term[cid] = t
            scored = _disorders_scored(driver, ids, weight=1.0)
            for dis, (sc, matched) in scored.items():
                disorder_scores[dis] = disorder_scores.get(dis, 0.0) + sc
                disorder_syms.setdefault(dis, set()).update(matched)
        ranked = sorted(disorder_scores.items(), key=lambda kv: -kv[1])[: top_k * 3]
        out = []
        for dis, score in ranked:
            explained = [sym_term.get(i, i) for i in disorder_syms.get(dis, set())]
            out.append(_build_context(driver, dis, explained, score))
        return out[:top_k]
    finally:
        driver.close()


def retrieve_temporal(presentation: dict, top_k: int = 5) -> list:
    """Temporal/graduated presentation -> ranked diagnosis candidates.

    `presentation` maps day label -> list of free-text symptoms, e.g.
        {"day1": ["fever","fatigue"],
         "day2": ["rash","lymphadenopathy"],
         "day3": ["weight loss","oral candidiasis"]}
    Each day's symptoms are matched to concepts (expanded to ancestors), the
    concept set is accumulated over time, and disorders are scored by weighted
    overlap of their associated findings with the accumulated set — LATER days
    weigh more (a progressive illness reveals its signature over time).
    """
    driver = _driver()
    try:
        days = sorted(presentation.keys(),
                      key=lambda k: int(re.sub(r"\D", "", str(k)) or 0))
        day_scores = {}        # disorder_id -> weighted score
        day_match_detail = {}  # disorder_id -> list of (term, day)
        sym_term = {}
        for idx, day in enumerate(days):
            weight = DAY_WEIGHTS[min(idx, len(DAY_WEIGHTS) - 1)]
            cset = _symptom_concept_set(driver, " and ".join(presentation[day]))
            day_ids = set(cset)
            for cid in day_ids:
                t = _term_for(driver, cid)
                if t:
                    sym_term[cid] = t
            scored = _disorders_scored(driver, day_ids, weight=weight)
            for dis, (sc, matched) in scored.items():
                day_scores[dis] = day_scores.get(dis, 0.0) + sc
                for sid in matched:
                    day_match_detail.setdefault(dis, []).append(
                        (sym_term.get(sid, sid), idx + 1))
        # Rank by raw weighted score, but BOOST disorders that explain symptoms
        # across MULTIPLE days (a systemic illness, e.g. HIV, reveals itself
        # progressively) over those confined to a single day (local finding).
        def _distinct_days(dis):
            return len({d for _, d in day_match_detail.get(dis, [])})
        ranked = sorted(
            day_scores.items(),
            key=lambda kv: (-(_distinct_days(kv[0]) ** 1.5) * 2 - kv[1]))[: top_k * 3]
        out = []
        for dis, score in ranked:
            matched = [f"{t} (day{d})" for t, d in
                       sorted(set(day_match_detail.get(dis, [])),
                              key=lambda x: x[1])]
            out.append(_build_context(driver, dis, matched, score))
        return out[:top_k]
    finally:
        driver.close()
