"""domain_loader.py — V3.0 YAML-driven domain configuration.

Loads domain_config.yaml once at import, validates schema, and serves
domain dicts by name. No hardcoded logic — everything derives from YAML.

Usage:
    from domain_loader import get_domain, list_domains, reload_domains
    d = get_domain("engineering")
    d["entity_types"]   # -> ["Microservice", "Database", ...]
    d["relation_types"] # -> ["FIXES", "AUTHORED", ...]
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
_DEFAULT_DOMAIN = "engineering"
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


def _ensure_loaded() -> None:
    with _LOCK:
        if not _DOMAINS:
            pass
    # Lazy-load outside lock to avoid re-entrancy; lock protects the dict
    if not _DOMAINS:
        reload_domains()


def get_domain(name: str | None = None) -> Dict[str, Any]:
    """Return the domain config dict for `name` (or default if None)."""
    _ensure_loaded()
    with _LOCK:
        key = name or _DEFAULT_DOMAIN
        if key not in _DOMAINS:
            # Fall back to default if requested domain unknown
            if key is not None and key != _DEFAULT_DOMAIN:
                if _DEFAULT_DOMAIN in _DOMAINS:
                    return _DOMAINS[_DEFAULT_DOMAIN]
            raise KeyError(f"unknown domain '{key}' (available: {list_domains()})")
        return _DOMAINS[key]


def list_domains() -> List[str]:
    """Return list of available domain names."""
    _ensure_loaded()
    with _LOCK:
        return list(_DOMAINS.keys())


def get_default_domain() -> str:
    """Return the configured default domain name."""
    _ensure_loaded()
    with _LOCK:
        return _DEFAULT_DOMAIN


# Module import triggers initial load
reload_domains()
