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
    vec: Optional[list] = None     # precomputed query embedding (reused by the
                                    # default retriever to avoid a 2nd embed)


@dataclass
class FederateResult:
    q_res: List[dict] = field(default_factory=list)   # signal-bearing records
    g_res: List[dict] = field(default_factory=list)     # graph/structured hits
    qdrant_hits: int = 0
    graph_hits: int = 0
    path: str = "none"
    signals: List[str] = field(default_factory=list)    # distinct _signal tags seen
    failed: List[str] = field(default_factory=list)     # domains that errored (P3.12)


# ── helpers ───────────────────────────────────────────────────────────────────
def _as_record(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    return {"text": x, "doc_id": "", "doc_type": "", "chunk_idx": -1}


def _tag_signal(rec: dict, name: str) -> dict:
    rec["_signal"] = name
    return rec


def _resolve_retrieval(profile: dict, name: str, req: FederateRequest,
                        out: List[dict], hits: dict, signals: list,
                        failed: list = None) -> None:
    """Run ONE domain's retriever and merge its records into out[] with _signal.

    The retriever may pre-tag each record with a ``_sig`` field (e.g. the
    default retriever tags vector hits ``<domain>`` and graph hits
    ``<domain>:graph``). If absent, the signal falls back to ``name`` (used by
    companion retrievers, which have no internal split).

    Graceful degradation (P3.12): a per-domain retriever failure is recorded in
    ``failed`` and does NOT crash the whole /ask call — the other domains still
    return results.

    The effective top_k is taken from the domain's profile (configurable via
    domain_config.yaml `top_k`), falling back to the request's top_k.
    """
    try:
        retrieve = get_retriever(name, temporal=bool(req.presentation), vec=req.vec)
        domain_profile = get_domain(name)
        domain_top_k = domain_profile.get("top_k", req.top_k)
        if req.presentation is not None:
            hits_raw = retrieve(req.presentation, top_k=domain_top_k)
        else:
            hits_raw = retrieve(req.query, top_k=domain_top_k)
        for h in hits_raw:
            rec = _as_record(h)
            # Ensure 'text' field exists (some retrievers may omit it)
            if "text" not in rec or rec["text"] is None:
                rec["text"] = ""
            sig = rec.get("_sig", name)
            tagged = _tag_signal(rec, sig)
            out.append(tagged)
            if name not in signals:
                signals.append(name)
        hits[name] = len(hits_raw)
    except Exception as e:
        print(f"[federated] domain '{name}' retrieve failed: {e}", flush=True)
        if failed is not None:
            failed.append(name)
        hits[name] = 0



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

    # Primary domain + companions, all resolved uniformly through the
    # domains/ factory (get_retriever). The primary uses the default retriever
    # (Qdrant collection + Neo4j graph); companions use their own retrieve.py
    # (or the default if they have no custom file). Every retriever tags its
    # records with _signal; assemble just runs them in parallel and merges.
    primary_is_retriever = True  # primary always routed via get_retriever now
    companions = list(profile.get("companions", []) or [])

    # ── domain + companion retrievers, in parallel ──
    out: List[dict] = []
    hits: Dict[str, int] = {}
    signals: List[str] = []
    failed: List[str] = []

    sources = []
    if primary_is_retriever:
        sources.append(req.domain)
    sources.extend(companions)

    threads = []
    for name in sources:
        t = threading.Thread(
            target=lambda n=name: _resolve_retrieval(
                profile, n, req, out, hits, signals, failed))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    qdrant_hits = sum(hits.values())
    path = "hybrid" if len(signals) > 1 else (signals[0] if signals else "none")

    return FederateResult(
        q_res=out, g_res=[], qdrant_hits=qdrant_hits,
        graph_hits=hits.get(req.domain, 0), path=path, signals=signals,
        failed=failed,
    )
