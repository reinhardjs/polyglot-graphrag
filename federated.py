"""Federated retrieval assembly — config-driven multi-retriever composition.

Design: a thin FACTORY over the existing per-domain STRATEGY retrievers
(domains/<name>/retrieve.py exposes retrieve(query, top_k), optional
retrieve_temporal). The Mediator (/ask) delegates composition here instead of
hardcoding `if domain == "snomed": ...`.

This is what makes the system domain-independent: adding a companion corpus or
a new domain is a YAML edit (domain_config.yaml `companions:` / `retrieval:`),
with NO new `if` branch in serve_gpu.py.

Contract
--------
* FederateRequest: the minimal inputs the factory needs from /ask.
* assemble(domain, profile, query, top_k, presentation=None) -> FederateResult
    - runs each configured retriever (parallel threads)
    - tags every record with `_signal = <domain name>` (so the fusion
      dual-signal step can cross-reference signals by name, generically)
    - returns q_res (list[dict]), g_res (list[dict]), hits, path
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import config as C
from domain_loader import get_domain  # validates domain + returns config dict
from domains import get_retriever


# ── inputs / outputs ────────────────────────────────────────────────────────
@dataclass
class FederateRequest:
    domain: str
    query: str
    top_k: int = 5
    presentation: Optional[dict] = None  # temporal {day1:[...], ...}
    synthesize: bool = False       # forwarded for synthesis-kind decision
    mode: Optional[str] = None    # "differential" -> dual_signal pref


@dataclass
class FederateResult:
    q_res: List[dict] = field(default_factory=list)   # signal-bearing records
    g_res: List[dict] = field(default_factory=list)     # graph/structured hits
    qdrant_hits: int = 0
    graph_hits: int = 0
    path: str = "none"
    signals: List[str] = field(default_factory=list)    # distinct _signal tags seen


# ── helpers ───────────────────────────────────────────────────────────────────
def _as_record(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    return {"text": x, "doc_id": "", "doc_type": "", "chunk_idx": -1}


def _tag_signal(rec: dict, name: str) -> dict:
    rec["_signal"] = name
    return rec


def _resolve_retrieval(profile: dict, name: str, req: FederateRequest,
                        out: List[dict], hits: dict, signals: list) -> None:
    """Run ONE domain's retriever and merge its records into out[] with _signal."""
    retrieve = get_retriever(name, temporal=bool(req.presentation))
    if req.presentation is not None:
        hits_raw = retrieve(req.presentation, top_k=req.top_k)
    else:
        hits_raw = retrieve(req.query, top_k=req.top_k)
    tagged = [_tag_signal(_as_record(h), name) for h in hits_raw]
    out.extend(tagged)
    hits[name] = len(tagged)
    if name not in signals:
        signals.append(name)


# ── the factory ───────────────────────────────────────────────────────
def assemble(req: FederateRequest, vec: Optional[list] = None,
             rag=None) -> FederateResult:
    """Assemble the retriever set for `req.domain` purely from its config.

    Sources of evidence (config-driven, generic — no hardcoded domain branch):
      1. Primary retriever — if `retrieval != dense_prose`, the domain's own
         domains/<name>/retrieve.py (e.g. snomed term-match). Tagged `_signal`.
      2. `companions:` — list of companion corpus domains (e.g. clinical_prose).
         Each runs as its own signal. May be empty.
      3. Generic Qdrant + Neo4j subgraph — when the domain declares a
         `collection` (vector search) and/or `neo4j_label` (graph walk). This
         is the path engineering/legal/medical/etc. use. Runs in parallel
         threads. Records tagged `_signal=<domain>` (or `_signal=graph`).

    `vec` + `rag` are needed only for path (3); pass them from /ask.
    """
    profile = get_domain(req.domain)
    retrieval = profile.get("retrieval", "dense_prose")
    collection = profile.get("collection")
    neo4j_label = profile.get("neo4j_label")

    # Primary domain is a retriever source unless it's a pure prose corpus
    # (then it only acts as a companion, never a primary graph signal).
    primary_is_retriever = retrieval != "dense_prose"
    companions = list(profile.get("companions", []) or [])

    # ── (1)+(2) domain + companion retrievers, in parallel ──
    out: List[dict] = []
    hits: Dict[str, int] = {}
    signals: List[str] = []

    sources = []
    if primary_is_retriever:
        sources.append(req.domain)
    sources.extend(companions)

    threads = []
    for name in sources:
        t = threading.Thread(
            target=lambda n=name: _resolve_retrieval(
                profile, n, req, out, hits, signals))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ── (3) generic Qdrant + (optional) Neo4j parallel retrieval ──
    # A companion with `neo4j_label: null` is prose-only → skip the graph
    # leg entirely (avoids a wasted Neo4j query + `source_doc` warnings).
    if collection and rag is not None and vec is not None:
        g_out: List[dict] = []
        q_out: List[dict] = []
        from ask import resolve_collections
        collections = resolve_collections(collection)
        t1 = threading.Thread(
            target=lambda: q_out.extend(
                rag.qdrant_search_multi(vec, req.query, collections)))
        threads = [t1]
        if neo4j_label:   # only hit the graph when a label is declared
            t2 = threading.Thread(
                target=lambda: g_out.extend(
                    rag.neo4j_subgraph(
                        req.query,
                        label=neo4j_label,
                        entry_strategy=(profile.get("entry_strategy", "keyword")
                                       if profile else "keyword"))))
            threads.append(t2)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for r in q_out:
            out.append(_tag_signal(_as_record(r), req.domain))
        for r in g_out:
            out.append(_tag_signal(_as_record(r), req.domain + ":graph"))
        hits[req.domain + ":qdrant"] = len(q_out)
        hits[req.domain + ":graph"] = len(g_out)
        if req.domain not in signals:
            signals.append(req.domain)

    qdrant_hits = sum(hits.values())
    path = "hybrid" if len(signals) > 1 else (signals[0] if signals else "none")

    return FederateResult(
        q_res=out, g_res=[], qdrant_hits=qdrant_hits,
        graph_hits=hits.get(req.domain, 0), path=path, signals=signals,
    )
