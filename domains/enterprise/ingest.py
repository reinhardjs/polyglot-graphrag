"""domains/enterprise/ingest.py — Enterprise KM ingestion (standard text pipeline).

Two paths:
  * ingest(source=...) — generic ingest of an explicit (doc_id, text) list
    via the default text pipeline (used by bulk ingesters / callers that
    already have document text).
  * ingest(seed=True) or ingest() with no source — seeds the system's OWN
    documentation (the repo's docs/ tree) into `enterprise` as `self-docs`,
    so a fresh user gets a queryable KB on first startup. This is what the
    daemon's SEED_ON_STARTUP hook invokes.
"""
from __future__ import annotations

from domains._default import default_ingest
from domains.enterprise import seed as _seed


def ingest(source=None, seed=None, **kw) -> int:
    """Enterprise ingest.

    If ``source`` (a list of (doc_id, text)) is provided, run the generic
    text pipeline. Otherwise (e.g. called by SEED_ON_STARTUP), seed the
    repo's own docs/ tree into the enterprise domain.
    """
    if source:
        return default_ingest("enterprise", source=source, **kw)
    # No explicit source -> seed the system's self-documentation.
    status = _seed.seed(domain="enterprise")
    print(f"[enterprise.ingest] {status}", flush=True)
    return 1 if status.startswith("enterprise: self-docs seeded") else 0
