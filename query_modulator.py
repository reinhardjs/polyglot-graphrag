"""Phase 1 — Domain-Agnostic Query Modulation (Symbolic Pre-Filtering).

Problem
-------
Before a query is embedded by Jina v3, domain slang / abbreviations can be
misunderstood by the embedding model (e.g. "NDA breached" should be grounded
as "Non-Disclosure Agreement Violation"). This module rewrites the raw query
by resolving known aliases to their canonical concepts from the Neo4j graph.

Domain-agnostic by design
-------------------------
Nothing about a specific domain is hardcoded here. The alias vocabulary is read
live from Neo4j for the *active domain* (selected via `ACTIVE_DOMAIN` in
config.py or `domain=` on the request). Switching domain = switching vocabulary,
no code change. This mirrors the v2.6.0 profile system.

Real schema (adapted from the assumed `(:Alias)-[:RESOLVES_TO]->(:Concept)`)
--------------------------------------------------------------------------------
Our graph stores entities as `(:Entity)` nodes with:
    id, name, type, source_doc, source_docs, profile, aliases[], name_vector
Domain scoping is done via the domain profile (collection + label) — there is
no `:Concept`/`:Alias` label in this deployment. We therefore derive the alias
map from `:Entity` nodes that carry an `aliases` property (and from name↔type
pairs) for the active domain.

Public API
----------
    build_alias_map(domain=None, driver=None) -> dict[str, str]
    modulate_query(query, domain=None) -> str
    QueryModulator.moderate(query, domain=None) -> str   # daemon-friendly wrapper
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import config as C


# ── Neo4j access ────────────────────────────────────────────────────────────
def _get_driver():
    """Return a Neo4j driver (mirrors ingest.get_neo4j on-demand pattern)."""
    import ingest as I  # local import keeps startup light
    return I.get_neo4j()


def _domain_label_clause(domain: Optional[str]) -> str:
    """Return a Neo4j label scope for alias lookup.

    NOTE: in the current deployment, `:Entity` nodes are NOT tagged with a
    per-domain secondary label (ingest creates plain `:Entity`). Domain routing
    happens at the Qdrant *collection* level during retrieval, not via Neo4j
    labels. We therefore scan all `:Entity` nodes for aliases — the alias map is
    small (<1k nodes) and shared across domains, which is correct: "NDA" means
    NDA in every domain. If a future ingest step applies `:Entity:<Label>`,
    this clause will scope automatically (no code change needed).
    """
    return ""


def build_alias_map(domain: Optional[str] = None,
                    driver=None) -> Dict[str, str]:
    """Build alias → canonical-name map for the active domain from Neo4j.

    Sources of aliases (domain-agnostic):
      1. `n.aliases` property on any :Entity node (explicit synonym list).
      2. entity `name` → its `type` (e.g. "NDA" of type "LegalDoc" → "LegalDoc").
    Returns {lower_alias: canonical_name}. Longest alias wins on collision.
    """
    own_driver = driver is None
    drv = driver or _get_driver()
    try:
        label = _domain_label_clause(domain)
        cypher = (
            f"MATCH (n:Entity{label}) "
            "WHERE n.aliases IS NOT NULL OR n.type IS NOT NULL "
            "RETURN n.name AS name, n.type AS type, n.aliases AS aliases"
        )
        with drv.session() as s:
            rows = s.run(cypher).data()
    finally:
        if own_driver:
            drv.close()

    amap: Dict[str, str] = {}
    for row in rows:
        name = (row.get("name") or "").strip()
        etype = (row.get("type") or "").strip()
        aliases = row.get("aliases") or []
        if not name:
            continue
        # explicit aliases → canonical name
        for a in aliases:
            a = str(a).strip()
            if a:
                amap[a.lower()] = name
        # include name itself in the map (self-reference; type expansion deferred)
        if etype:
            amap.setdefault(name.lower(), name)
    return amap


# ── Modulation ──────────────────────────────────────────────────────────────
_WORD_SPLIT = re.compile(r"\b[\w./+-]+\b", re.UNICODE)


def _tokenize_longest_first(amap: Dict[str, str]) -> List[str]:
    """Alias keys sorted by length desc so 'NDA' wins over 'N' on overlap."""
    return sorted(amap.keys(), key=len, reverse=True)


def _apply_alias_map(query: str, amap: Dict[str, str]) -> str:
    """Expand known aliases in `query` using a prebuilt alias→canonical map.

    The original query text is preserved; expansions are appended in
    parentheses so the embedding model sees both the colloquial and canonical
    form (best of both worlds for retrieval). Unknown tokens are untouched.
    """
    if not query or not query.strip() or not amap:
        return query

    expansions: List[str] = []
    seen: set = set()
    for alias in _tokenize_longest_first(amap):
        if alias in seen:
            continue
        pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        if pat.search(query):
            canonical = amap[alias]
            if canonical.lower() != alias and canonical.lower() not in seen:
                expansions.append(canonical)
                seen.add(canonical.lower())
            seen.add(alias)

    if not expansions:
        return query
    uniq = list(dict.fromkeys(expansions))
    return f"{query} ({'; '.join(uniq)})"


def modulate_query(query: str,
                   domain: Optional[str] = None,
                   driver=None) -> str:
    """Rewrite `query` by expanding known domain aliases in-place.

    Example (medical domain): "lisinopril for htn" →
        "lisinopril for htn (Hypertension)"
    Builds the alias map fresh from Neo4j each call (cheap, <1k nodes). For a
    cached variant use `QueryModulator.moderate`.
    """
    if not query or not query.strip():
        return query
    amap = build_alias_map(domain, driver)
    return _apply_alias_map(query, amap)


class QueryModulator:
    """Daemon-friendly wrapper (stateless; reuses the per-domain alias cache)."""

    _cache: Dict[str, Dict[str, str]] = {}

    @classmethod
    def moderate(cls, query: str, domain: Optional[str] = None,
                 driver=None) -> str:
        key = domain or "__all__"
        amap = cls._cache.get(key)
        if amap is None:
            amap = build_alias_map(domain, driver)
            cls._cache[key] = amap
        return _apply_alias_map(query, amap)

    @classmethod
    def refresh(cls, domain: Optional[str] = None) -> None:
        """Force alias-map rebuild (call after ingest to pick up new synonyms)."""
        cls._cache.pop(domain or "__all__", None)
