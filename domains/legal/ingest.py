"""domains/legal/ingest.py — Legal & Compliance ingestion (standard text)."""
from __future__ import annotations

from domains._default import default_ingest


def ingest(source=None, **kw) -> int:
    """Generic legal ingest. source = list of (doc_id, text)."""
    return default_ingest("legal", source=source, **kw)
