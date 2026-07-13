#!/usr/bin/env python3
"""WS2 benchmark — no-frills data extraction pipeline for GraphRAG comparison.

This tests the raw extraction LLM (E2B) with a fixed, small set of graph
facts. It does NOT use GLiNER, sliding-window, Jina, Qdrant, Neo4j, or
any other pipeline component — it's a pure model-isolation benchmark.

Usage:
    python bench_ws2.py
"""
import json, os, sys
import config as C

GROUND_TRUTH = {
    ("bob", "REPORTS", "BUG-204"),
    ("bob", "REPORTS", "BUG-207"),
    ("carol", "REVIEWED", "BUG-204"),
    ("BR-01", "IMPLEMENTS", "PR-482"),
    ("alice", "REVIEWED", "PR-482"),
    ("checkout-publisher", "DEPENDS_ON", "checkout database"),
    ("ADR-014", "REFERENCES", "auth-service"),
}


def get_profile():
    import domain_loader
    return domain_loader.get_domain("engineering")


def raw_extract(text, profile):
    """Call E2B directly with robust JSON extraction (mirrors ingest path but
    never falls back to GLiNER on trailing-prose parse errors — those are
    model/prompt noise, not flag effects)."""
    prompt_tmpl = (profile.get("extraction", {}).get("prompt")
                   if profile else None) or C.EXTRACTION_PROMPT