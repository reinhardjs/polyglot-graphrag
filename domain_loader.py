"""domain_loader.py — V3.0 YAML-driven domain configuration.

Loads domain_config.yaml once at import, validates schema, and serves
domain dicts by name. No hardcoded logic — everything derives from YAML.

Usage:
    from domain_loader import get_domain, list_domains, reload_domains
    d = get_domain("enterprise")
    d["entity_types"]   # -> ["ADR", "Runbook", "Postmortem", "RFC", "Service", "Team", ...]
    d["relation_types"] # -> ["DOCUMENTS", "OWNS", "DEPENDS_ON", ...]
"""
from __future__ import annotations

import os
import threading
from typing import Dict, Any, List

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "domain_config.yaml")

_REQUIRED_KEYS = [
    "collection",
    "neo4j_label",
    "entity_types",
    "relation_types",
    "pair_strategy",
    "min_confidence",
    "chunking",
    "metadata_schema",
]

_LOCK = threading.Lock()
_DOMAINS: Dict[str, Any] = {}
_DEFAULT_DOMAIN = "enterprise"  # must match domain_config.yaml `default_domain`.
# NOTE: the YAML `default_domain` key OVERRIDES this constant at load time
# (domain_loader reloads from YAML). This is only the fallback if the YAML
# omits `default_domain`. `enterprise` is the current default (set via
# `default_domain: enterprise` in domain_config.yaml). `engineering` was an
# earlier experimental domain that no longer exists in the YAML.
# Top-level config blocks (dynamic_labels, llm_fallback) exposed for runtime
# access without re-parsing the YAML.
_TOP_LEVEL: Dict[str, Any] = {}


def _validate_domain(name: str, cfg: Dict[str, Any]) -> None:
    """Raise ValueError if a domain block is missing required keys."""
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"domain '{name}' missing required keys: {missing}"
        )
    if cfg["pair_strategy"] not in ("all_pairs", "same_sentence"):
        raise ValueError(
            f"domain '{name}' pair_strategy must be 'all_pairs' or "
            f"'same_sentence', got '{cfg['pair_strategy']}'"
        )
    # Pure prose / vector-only domains (retrieval: dense_prose) have no graph
    # extraction, so empty entity/relation lists are valid.
    is_prose_only = cfg.get("retrieval") == "dense_prose"
    if not is_prose_only:
        if not isinstance(cfg["entity_types"], list) or not cfg["entity_types"]:
            raise ValueError(f"domain '{name}' entity_types must be a non-empty list")
        if not isinstance(cfg["relation_types"], list) or not cfg["relation_types"]:
            raise ValueError(f"domain '{name}' relation_types must be a non-empty list")
    # chunking sub-keys
    chunking = cfg.get("chunking", {})
    for ck in ("strategy", "chunk_size", "overlap"):
        if ck not in chunking:
            raise ValueError(f"domain '{name}' chunking missing '{ck}'")


def _load_file() -> Dict[str, Any]:
    """Read YAML from disk and validate all domains."""
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"domain config not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "domains" not in raw:
        raise ValueError("domain_config.yaml must have a top-level 'domains' key")
    domains = raw["domains"]
    if not isinstance(domains, dict) or not domains:
        raise ValueError("domain_config.yaml 'domains' must be a non-empty mapping")
    for name, cfg in domains.items():
        # An *alias* block ({alias: <target>}) is NOT a real domain — it
        # just points a public name at a concrete domain. Skip schema validation.
        if isinstance(cfg, dict) and "alias" in cfg:
            if not isinstance(cfg["alias"], str) or not cfg["alias"]:
                raise ValueError(f"domain '{name}' alias must be a non-empty string")
            continue
        _validate_domain(name, cfg)
    return {
        "domains": domains,
        "default_domain": raw.get("default_domain", _DEFAULT_DOMAIN),
        "dynamic_labels": raw.get("dynamic_labels", {}),
        "llm_fallback": raw.get("llm_fallback", {}),
    }


def reload_domains() -> None:
    """Reload domain_config.yaml from disk (call after editing the file)."""
    global _DOMAINS, _DEFAULT_DOMAIN, _TOP_LEVEL
    parsed = _load_file()
    with _LOCK:
        _DOMAINS = parsed["domains"]
        _DEFAULT_DOMAIN = parsed["default_domain"]
        _TOP_LEVEL = {
            "dynamic_labels": parsed.get("dynamic_labels", {}),
            "llm_fallback": parsed.get("llm_fallback", {}),
        }


def get_top_level(key: str) -> Dict[str, Any]:
    """Return a top-level config block (dynamic_labels / llm_fallback)."""
    _ensure_loaded()
    with _LOCK:
        return _TOP_LEVEL.get(key, {})


def _resolve_alias(name: str) -> str:
    """Resolve a domain *alias* to its concrete target name.

    An alias is a `domains:<name>:` block with an `alias:` key (e.g.
    `healthcare: {alias: snomed}`). This lets us expose a stable public
    name (`healthcare`) whose underlying strategy is another domain, WITHOUT
    renaming the concrete domain (which is referenced in many places).
    Returns `name` unchanged if it is not an alias.
    """
    with _LOCK:
        cfg = _DOMAINS.get(name)
    if isinstance(cfg, dict) and isinstance(cfg.get("alias"), str):
        return cfg["alias"]
    return name


def _ensure_loaded() -> None:
    with _LOCK:
        if not _DOMAINS:
            pass
    # Lazy-load outside lock to avoid re-entrancy; lock protects the dict
    if not _DOMAINS:
        reload_domains()


def get_domain(name: str | None = None) -> Dict[str, Any]:
    """Return the domain config dict for `name` (or default if None).

    `name` may be an *alias* (e.g. `healthcare` → `snomed`); it is
    resolved transparently via `_resolve_alias`. The returned dict is a
    deep copy of the concrete domain so callers can mutate it safely.
    """
    _ensure_loaded()
    resolved = _resolve_alias(name or _DEFAULT_DOMAIN)
    with _LOCK:
        key = resolved
        if key not in _DOMAINS:
            # Fall back to default if requested domain unknown
            if resolved != _DEFAULT_DOMAIN and _DEFAULT_DOMAIN in _DOMAINS:
                key = _DEFAULT_DOMAIN
            else:
                raise KeyError(
                    f"unknown domain '{name}' (available: {list_domains()})")
        import copy
        return copy.deepcopy(_DOMAINS[key])


def list_domains() -> List[str]:
    """Return list of available domain names."""
    _ensure_loaded()
    with _LOCK:
        return list(_DOMAINS.keys())


def get_default_domain() -> str:
    """Return the configured default domain name (may be an alias, e.g. `default`).

    Callers that need the actual strategy should pass the returned name to
    `get_domain()`, which resolves the alias transparently.
    """
    _ensure_loaded()
    with _LOCK:
        return _DEFAULT_DOMAIN


# Module import triggers initial load
reload_domains()
