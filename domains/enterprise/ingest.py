"""domains/enterprise/ingest.py — Enterprise KM ingestion (standard text pipeline).

The primary corpus uses the generic ingest path. Seeding demo data (ADR-021,
BUG-204, PR-482, runbook-checkout) is done via seed.py for the --demo flag.
"""
from __future__ import annotations

from domains._default import default_ingest


def ingest(source=None, **kw) -> int:
    """Generic enterprise ingest. source = list of (doc_id, text)."""
    return default_ingest("enterprise", source=source, **kw)
