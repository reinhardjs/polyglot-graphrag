"""domains/ — per-domain ingest + retrieve plugins for polyglot-graphrag.

Design goal: every knowledge domain (snomed, clinical_prose, engineering,
medical, ...) owns its OWN ingest and retrieve logic, kept under
``domains/<name>/``. The rest of the system never hardcodes a domain's
internals — it asks the registry:

    from domains import get_retriever, list_domains
    retrieve = get_retriever("snomed")
    contexts = retrieve(query, top_k=5)

Adding a new domain = drop a folder with ``retrieve.py`` (+ optional
``ingest.py``) and register it in ``domain_config.yaml``. No changes to
serve_gpu.py / ask.py / ingest.py.

Contract
--------
* ``retrieve.py`` MUST expose ``retrieve(query, top_k=5) -> list[dict]``
  where each dict has keys ``text``, ``doc_id``, ``doc_type``, ``chunk_idx``
  (same shape ask.py normalises from graph hits). It MAY also expose
  ``retrieve_temporal(presentation, top_k=5)`` for graduated presentations.
* ``ingest.py`` (optional) exposes ``ingest(source, **kw)`` that loads data
  into that domain's Qdrant collection / Neo4j subgraph.
* The domain's Qdrant collection name and Neo4j label come from
  ``domain_config.yaml`` (``collection`` / ``neo4j_label``), NOT from code.
"""

from __future__ import annotations

import importlib
import os
import threading
from typing import Callable, Dict, List, Optional

# Lazy import to avoid a hard dependency cycle at module import time.
from domain_loader import get_domain, list_domains as _cfg_list_domains

_DOMAINS_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache of imported retrieve modules: name -> module
_RETRIEVERS: Dict[str, object] = {}
_LOCK = threading.Lock()

# Optional explicit overrides (rarely needed): name -> callable
_OVERRIDES: Dict[str, Callable] = {}


def register_retriever(name: str, fn: Callable) -> None:
    """Manually register a retrieve callable (advanced use)."""
    with _LOCK:
        _OVERRIDES[name] = fn


def _load_retrieve_module(name: str):
    """Import domains/<name>/retrieve.py on demand and validate the contract."""
    mod_name = f"domains.{name}.retrieve"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"domain '{name}' has no retrieve.py under domains/{name}/ "
            f"({e})"
        ) from e
    if not hasattr(mod, "retrieve"):
        raise RuntimeError(
            f"domains/{name}/retrieve.py must expose retrieve(query, top_k=5)"
        )
    return mod


def get_retriever(name: str, temporal: bool = False) -> Callable:
    """Return the retrieve callable for a domain.

    Falls back to ``domain_config.yaml`` for the domain's existence, but the
    actual code lives in ``domains/<name>/retrieve.py``.

    ``temporal=True`` prefers ``retrieve_temporal(presentation, top_k)`` when
    the module defines it (used for graduated / day-by-day presentations).
    """
    # Explicit override wins.
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    with _LOCK:
        if name in _RETRIEVERS:
            mod = _RETRIEVERS[name]
        else:
            # validate the domain exists in config (raises KeyError if not)
            get_domain(name)
            mod = _load_retrieve_module(name)
            _RETRIEVERS[name] = mod
    if temporal and hasattr(mod, "retrieve_temporal"):
        return lambda query, top_k=5: mod.retrieve_temporal(query, top_k=top_k)
    return lambda query, top_k=5: mod.retrieve(query, top_k=top_k)


def list_domains() -> List[str]:
    """Domains with a retrieve.py present on disk (the runnable set)."""
    out = []
    for entry in sorted(os.listdir(_DOMAINS_DIR)):
        full = os.path.join(_DOMAINS_DIR, entry)
        if not os.path.isdir(full) or entry.startswith("_"):
            continue
        if os.path.exists(os.path.join(full, "retrieve.py")):
            out.append(entry)
    return out


def supported_domains() -> List[str]:
    """Domains declared in domain_config.yaml (the configured set)."""
    return _cfg_list_domains()
