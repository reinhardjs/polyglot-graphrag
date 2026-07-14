"""domains/ — per-domain ingest + retrieve plugins for polyglot-graphrag.

Design goal: every knowledge domain (snomed, clinical_prose, engineering, ...)
and every COMPANION corpus has a home under ``domains/<name>/``. A domain's
custom logic is OPTIONAL — if ``domains/<name>/retrieve.py`` (or ``ingest.py``)
is absent, the factory falls back to the shared default in ``domains/_default.py``
(the PRIMARY path: Qdrant ``collection`` + Neo4j ``neo4j_label``, generic
ingest via ingest.py::ingest_text). So the primary corpus of a domain needs no
custom code; only companions (different source/chunking) or special domains
(e.g. snomed's term-match) need plugins. The rest of the system never
hardcodes a domain's internals — it asks the registry:

    from domains import get_retriever, get_ingestor, list_domains
    retrieve = get_retriever("engineering")     # default unless overridden
    ingest  = get_ingestor("engineering_docs")  # custom ingest.py

Adding a new domain = add a block to ``domain_config.yaml`` (+ optional
``domains/<name>/`` plugin). No changes to serve_gpu.py / ask.py / ingest.py.

Contract
--------
* ``retrieve.py`` SHOULD expose ``retrieve(query, top_k=5) -> list[dict]``
  where each dict has keys ``text``, ``doc_id``, ``doc_type``, ``chunk_idx``.
  Optional ``retrieve_temporal(presentation, top_k=5)`` for graduated
  presentations. If absent, the default retriever is used.
* ``ingest.py`` (optional) exposes ``ingest(source=None, **kw)`` that loads
  data into that domain's Qdrant collection / Neo4j subgraph. If absent, the
  default ingestor is used.
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
from domain_loader import _resolve_alias

_DOMAINS_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache of imported retrieve modules: name -> module
_RETRIEVERS: Dict[str, object] = {}
_LOCK = threading.Lock()

# Cache of default-retriever fallbacks: name -> default_retrieve fn
_DEFAULT_RETRIEVERS: Dict[str, Callable] = {}
# Cache of ingest callables: name -> ingest fn (custom or default)
_INGESTORS: Dict[str, Callable] = {}

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


def get_retriever(name: str, temporal: bool = False, vec: Optional[list] = None) -> Callable:
    """Return the retrieve callable for a domain.

    Resolution order:
      1. Explicit override (register_retriever).
      2. ``domains/<name>/retrieve.py`` if present — BUT `name` is first
         resolved through any domain_config.yaml ALIAS (so `healthcare` →
         `snomed` uses the snomed custom retriever, not the slow default
         neo4j_subgraph fallback).
      3. **Default generic retriever** (domains._default.default_retrieve) —
         searches the domain's Qdrant ``collection`` + Neo4j ``neo4j_label``
         from domain_config.yaml. This is the PRIMARY path for domains like
         ``engineering`` and the fallback for any domain without a custom
         retrieve.py, so a domain NEVER needs a folder to be queryable.

    ``vec`` (precomputed query embedding) is forwarded to the default retriever
    so it reuses the embedding /ask already computed (no 2nd embed). Custom
    companion retrievers ignore it and embed internally (as before).
    """
    # Resolve alias → concrete target so e.g. `healthcare` (alias→snomed) uses
    # snomed's custom retriever instead of falling back to default_retrieve.
    try:
        resolved = _resolve_alias(name)
    except Exception:
        resolved = name
    if resolved != name:
        return get_retriever(resolved, temporal=temporal, vec=vec)
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    with _LOCK:
        if name in _RETRIEVERS:
            mod = _RETRIEVERS[name]
        elif name in _DEFAULT_RETRIEVERS:
            fn = _DEFAULT_RETRIEVERS[name]
            return lambda query, top_k=5: fn(name, query, top_k, vec=vec)
        else:
            # validate the domain exists in config (raises KeyError if not)
            get_domain(name)
            try:
                mod = _load_retrieve_module(name)
                _RETRIEVERS[name] = mod
            except RuntimeError:
                # No custom retrieve.py → use the default generic retriever.
                from domains._default import default_retrieve
                _DEFAULT_RETRIEVERS[name] = default_retrieve
                return lambda query, top_k=5: default_retrieve(name, query, top_k, vec=vec)
    if temporal and hasattr(mod, "retrieve_temporal"):
        return lambda query, top_k=5: mod.retrieve_temporal(query, top_k=top_k)
    # Forward vec only if the retriever accepts it (clinical_prose does;
    # snomed does not — it embeds internally via the resident daemon).
    import inspect
    sig = inspect.signature(mod.retrieve)
    if "vec" in sig.parameters:
        return lambda query, top_k=5: mod.retrieve(query, top_k=top_k, vec=vec)
    return lambda query, top_k=5: mod.retrieve(query, top_k=top_k)


def get_ingestor(name: str) -> Callable:
    """Return the ingest callable for a domain.

    Resolution order:
      1. ``domains/<name>/ingest.py::ingest`` if present (companions, custom
         sources/chunking).
      2. **Default generic ingestor** (domains._default.default_ingest) — the
         primary path: embed + graph-extract via ingest.py::ingest_text (the
         same machinery ``sync_docs.py`` / ``POST /ingest`` uses). So the
         PRIMARY corpus needs no custom ingest.py; only companions do.

    Returns a callable ``ingest(source=None, **kw) -> int``.
    """
    with _LOCK:
        if name in _INGESTORS:
            return _INGESTORS[name]
        try:
            mod_name = f"domains.{name}.ingest"
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, "ingest"):
                raise RuntimeError(
                    f"domains/{name}/ingest.py must expose ingest(source, **kw)")
            fn = mod.ingest
        except ModuleNotFoundError:
            from domains._default import default_ingest
            fn = default_ingest
        _INGESTORS[name] = fn
        return fn


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
