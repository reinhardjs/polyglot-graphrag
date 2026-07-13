"""Phase 3 — Corrective RAG (CRAG) & Adaptive Routing.

A lightweight, dependency-free state machine (no LangGraph required) that wraps
the existing retrieve → condense → synthesize flow with two guards:

    ROUTE → RETRIEVE → EVALUATE → [CORRECT] → SYNTHESIZE
                          │
                          └─[INCORRECT/AMBIGUOUS] → REWRITE → RETRIEVE(2) → SYNTHESIZE

Nodes
-----
1. Router     — decide the retrieval path for a query:
                  "graph"  : strict factual lookup (entity/ID) → Neo4j only
                  "hybrid" : complex analysis → Qdrant + Neo4j
                Uses a fast heuristic; optionally confirmed by an E4B classifier
                (config.CRAG_USE_LLM_ROUTER).
2. Evaluator  — after retrieval, grade the context vs the query:
                  CORRECT | AMBIGUOUS | INCORRECT
                Heuristic (lexical overlap + non-empty) with optional E4B judge.
3. Fallback   — on INCORRECT/AMBIGUOUS, rewrite the query (E4B or heuristic) and
                run a second, wider retrieval, then synthesize.

Domain-agnostic: nothing here hardcodes a domain. Routing/labels come from the
active profile; the LLM prompts are generic. All LLM use is optional and guarded
so the pipeline degrades gracefully to pure heuristics if the E4B daemon is down.

Public API
----------
    run_crag(query, domain=None, synthesize=True, deps=None) -> dict
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

import config as C


# ── Router ───────────────────────────────────────────────────────────────────
# Entity/ID lookups look like: "who reported BUG-204", "what is ADR-014",
# "PR-482 status". Heuristic: presence of an ID-like token or a short factual
# "who/what/when/where"-style lookup → graph path.
_ID_RE = re.compile(r"\b[A-Za-z]{2,}-\d+\b")          # BUG-204, ADR-014, PR-482
_FACTUAL_LEAD = re.compile(r"^\s*(who|what|when|where|which)\b", re.IGNORECASE)


def route(query: str) -> str:
    """Return 'graph' for strict factual lookups, else 'hybrid'."""
    if _ID_RE.search(query):
        return "graph"
    # short factual question with no analytical verbs → graph
    analytical = re.search(r"\b(why|how|compare|analyz|impact|cause|explain|"
                           r"trade-?off|design|recommend)\w*\b", query, re.I)
    if _FACTUAL_LEAD.match(query) and not analytical and len(query.split()) <= 8:
        return "graph"
    return "hybrid"


def _llm_route(query: str) -> Optional[str]:
    """Optional E4B confirmation of the route. Returns 'graph'/'hybrid'/None."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=C.SYNTHESIS_LLM_BASE_URL,
                        api_key=C.SYNTHESIS_LLM_API_KEY)
        prompt = (
            "Classify the query for a knowledge base router. Answer with ONE "
            "word only: GRAPH if it is a strict factual lookup of a specific "
            "entity/ID/fact; HYBRID if it needs analysis, reasoning, or "
            "synthesis across documents.\nQuery: " + query + "\nAnswer:"
        )
        r = client.chat.completions.create(
            model=C.SYNTHESIS_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8, temperature=0.0,
        )
        txt = (r.choices[0].message.content or "").strip().lower()
        if "graph" in txt:
            return "graph"
        if "hybrid" in txt:
            return "hybrid"
    except Exception:
        return None
    return None


# ── Evaluator ────────────────────────────────────────────────────────────────
def evaluate_context(query: str, contexts: List[str]) -> str:
    """Grade retrieved context: CORRECT | AMBIGUOUS | INCORRECT (heuristic).

    - INCORRECT: no contexts at all.
    - CORRECT: strong lexical overlap between query terms and context.
    - AMBIGUOUS: some context but weak overlap (triggers a corrective pass).
    """
    if not contexts:
        return "INCORRECT"
    q_terms = set(re.findall(r"\w+", query.lower())) - _STOP
    if not q_terms:
        return "CORRECT" if contexts else "INCORRECT"
    joined = " ".join(contexts).lower()
    ctx_terms = set(re.findall(r"\w+", joined))
    overlap = len(q_terms & ctx_terms) / max(1, len(q_terms))
    if overlap >= 0.5:
        return "CORRECT"
    if overlap >= 0.2:
        return "AMBIGUOUS"
    return "INCORRECT"


_STOP = {"the", "a", "an", "of", "to", "is", "are", "was", "were", "and", "or",
         "in", "on", "for", "what", "who", "when", "where", "which", "how",
         "why", "does", "do", "did", "has", "have", "with", "that", "this"}


# ── Fallback / query rewrite ─────────────────────────────────────────────────
def rewrite_query(query: str) -> str:
    """Rewrite a query for a wider secondary search (heuristic + optional E4B)."""
    llm = _llm_rewrite(query)
    if llm:
        return llm
    # Heuristic fallback: strip stopwords + question scaffolding to keywords.
    terms = [t for t in re.findall(r"\w+", query.lower()) if t not in _STOP]
    return " ".join(terms) if terms else query


def _llm_rewrite(query: str) -> Optional[str]:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=C.SYNTHESIS_LLM_BASE_URL,
                        api_key=C.SYNTHESIS_LLM_API_KEY)
        prompt = (
            "Rewrite the user's question into a concise search query that will "
            "retrieve better results. Keep key entities. Output ONLY the "
            "rewritten query, no preamble.\nQuestion: " + query + "\nRewrite:"
        )
        r = client.chat.completions.create(
            model=C.SYNTHESIS_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=48, temperature=0.0,
        )
        txt = (r.choices[0].message.content or "").strip()
        return txt or None
    except Exception:
        return None


# ── Dependency injection surface (keeps the FSM testable) ─────────────────────
class CragDeps:
    """Injectable functions so the FSM can be unit-tested without daemons.

    Defaults wire to the real ask.py pipeline; tests pass fakes.
    """
    def __init__(self,
                 embed: Optional[Callable] = None,
                 retrieve: Optional[Callable] = None,
                 condense: Optional[Callable] = None,
                 synth: Optional[Callable] = None,
                 use_llm_router: Optional[bool] = None):
        import ask as rag
        self.embed = embed or rag.embed_query
        self.retrieve = retrieve or rag.parallel_retrieve
        self.condense = condense or rag.condense
        self.synth = synth or rag.synthesize
        self.use_llm_router = (C.CRAG_USE_LLM_ROUTER
                               if use_llm_router is None else use_llm_router)


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_crag(query: str,
             domain: Optional[str] = None,
             synthesize: bool = True,
             deps: Optional[CragDeps] = None) -> Dict:
    """Run the CRAG state machine and return a structured trace + answer.

    Returns:
        {
          "query", "route", "grade", "corrected" (bool),
          "rewritten_query" (str|None), "n_contexts", "contexts",
          "answer" (str|None), "path": ["route","retrieve","evaluate",...]
        }
    """
    deps = deps or CragDeps()
    trace: List[str] = []

    # 0. Domain profile (v0.x YAML) — resolves collection/label/entry strategy
    profile = None
    collections = None
    entry_strategy = "keyword"
    try:
        if domain:
            import domain_loader
            profile = domain_loader.get_domain(domain)
            coll = profile.get("collection")
            collections = [coll] if coll else None
            entry_strategy = profile.get("entry_strategy", "keyword")
    except Exception:
        profile = None

    # 1. ROUTE
    r = route(query)
    if deps.use_llm_router:
        confirmed = _llm_route(query)
        if confirmed:
            r = confirmed
    trace.append("route:" + r)

    # 2. RETRIEVE (first pass)
    vec = deps.embed(query)
    q_res, g_res = deps.retrieve(vec, query, collections, entry_strategy)
    if r == "graph":
        # strict factual → prefer graph hits; ignore vector pool for grading
        q_res_used = []
        g_res_used = g_res
    else:
        q_res_used, g_res_used = q_res, g_res
    contexts = deps.condense(query, q_res_used, g_res_used)
    trace.append("retrieve:%d" % len(contexts))

    # 3. EVALUATE
    grade = evaluate_context(query, contexts)
    trace.append("evaluate:" + grade)

    corrected = False
    rewritten = None
    # 4. FALLBACK (corrective pass) — only if the first pass was weak
    if grade in ("INCORRECT", "AMBIGUOUS"):
        rewritten = rewrite_query(query)
        corrected = True
        trace.append("rewrite")
        # widen: always hybrid on the corrective pass, both collections if known
        vec2 = deps.embed(rewritten)
        q2, g2 = deps.retrieve(vec2, rewritten, collections, "hybrid")
        # fuse first + second pass contexts, re-condense
        contexts2 = deps.condense(rewritten, q_res + q2, g_res + g2)
        if contexts2:
            contexts = contexts2
        grade = evaluate_context(rewritten, contexts)
        trace.append("re-evaluate:" + grade)

    # 5. SYNTHESIZE
    answer = None
    if synthesize and contexts:
        answer = deps.synth(query, contexts, profile)
        trace.append("synthesize")

    return {
        "query": query,
        "route": r,
        "grade": grade,
        "corrected": corrected,
        "rewritten_query": rewritten,
        "n_contexts": len(contexts),
        "contexts": contexts,
        "answer": answer,
        "path": trace,
    }


if __name__ == "__main__":
    import sys, json
    q = sys.argv[1] if len(sys.argv) > 1 else "who reported BUG-204?"
    dom = sys.argv[2] if len(sys.argv) > 2 else None
    out = run_crag(q, domain=dom, synthesize=False)
    out.pop("contexts", None)
    print(json.dumps(out, indent=2))
