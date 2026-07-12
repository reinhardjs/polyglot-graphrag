"""Dynamic Label Injection for GLiNER (v3.1.0).

Problem: the extraction pipeline drops relation edges when E2B references an
entity whose surface form GLiNER did not detect. GLiNER is given a fixed set of
entity TYPE labels from ``domain_config.yaml``; anything outside that set is
invisible.

Solution: accumulate the *names* of dropped entities, promote them to the active
GLiNER label set after ``promotion_threshold`` documents, and feed the enriched
label list back into future GLiNER calls. This is a zero-latency (in-memory)
enrichment of GLiNER's vocabulary -- it does NOT add a second LLM pass.

Design note on semantics
------------------------
GLiNER's ``labels`` argument is a list of *types* (e.g. "Model", "Dataset").
When an entity is dropped, we only know its *name* (e.g. "BM25"), not its type.
We therefore record the dropped name and, on promotion, inject that exact
surface form as an additional GLiNER label. ``gliner_multi-v2.1`` matches
specific phrases as entity types, so adding "BM25" makes future occurrences of
that phrase detectable. This directly raises recall for entities the static
schema missed, at the cost of a slightly larger label list (bounded by
``max_per_domain`` + LRU eviction + TTL).

State is persisted to ``~/.hermes/labels/{domain}_dynamic.json`` for crash
recovery and survives process restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import requests

logger = logging.getLogger("label_provider")


# Where dynamic-label state lives (L3 persistence in the caching architecture).
_STATE_DIR = os.path.expanduser("~/.hermes/labels")


# Relation-phrase fragments E2B emits as bogus source/target values.
# These are NOT entities and must never become GLiNER labels.
_RELATION_VERBS = (
    "is ", "are ", "was ", "were ", "uses", "used", "use ", "offers",
    "offered", "results", "result ", "outperforms", "outperformed",
    "measured", "measures", "tested", "tests", "test ", "calculated",
    "calculates", "collapsed", "approach", "compared", "compared to",
    "leads", "leading", "provides", "provided", "enables", "enabled",
)


@dataclass
class LabelCandidate:
    """A dropped entity name awaiting promotion to the active GLiNER label set."""

    label: str               # surface form injected into GLiNER (e.g. "BM25")
    display_name: str        # original name as seen in the relation (e.g. "BM25")
    inferred_type: str = ""  # GLiNER type if known, else "" (unknown)
    occurrences: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    doc_ids: Set[str] = field(default_factory=set)


def _is_plausible_entity(name: str) -> bool:
    """Heuristic: is this a likely entity surface form, not a relation phrase?

    Accepts: capitalized names, single technical tokens, short noun phrases
    (<=3 words). Rejects: multi-word phrases containing a relation-verb
    lexicon, or phrases longer than 3 words.
    """
    s = (name or "").strip()
    if not s or len(s) > 80:
        return False
    low = s.lower()
    # Reject if it contains a relation-verb fragment anywhere.
    if any(v in low for v in _RELATION_VERBS):
        return False
    words = s.split()
    if len(words) > 3:
        return False
    # Accept single words and capitalized phrases.
    if len(words) == 1:
        return True
    # Multi-word: require the first word to be capitalized (proper noun
    # or technical compound), reject lowercase-starting phrases.
    return s[0].isupper()


class LabelProvider:
    """Per-domain dynamic label accumulator with promotion + eviction + TTL.

    Phase 1 (record-only) and Phase 2 (promotion/eviction/reload) are both
    implemented here. All public methods are thread-safe.
    """

    def __init__(self, domain: str, cfg: dict):
        self.domain = domain
        dl = (cfg or {}).get("dynamic_labels", {}) or {}
        self.enabled = dl.get("enabled", True)
        self.max_labels = dl.get("max_per_domain", 20)
        self.promotion_threshold = dl.get("promotion_threshold", 3)
        self.ttl_docs = dl.get("ttl_docs", 50)
        self._candidates: Dict[str, LabelCandidate] = {}
        self._active: Set[str] = set()
        self._doc_counter = 0
        self._current_doc = ""
        self._lock = threading.Lock()
        self._load_state()

    # -- Public API (hot path) -------------------------------------------------

    def get_active(self) -> list:
        """Return active dynamic labels. O(1) in-memory access -- no I/O."""
        if not self.enabled:
            return []
        with self._lock:
            return list(self._active)

    def record_unknown(self, name: str, inferred_type: str = "") -> None:
        """Record a dropped entity name after E2B parse.

        Called from ``_parse_and_validate`` whenever an edge references an
        entity GLiNER did not detect. Thread-safe; never raises.

        Filters out relation-phrase fragments that E2B sometimes emits as
        bogus ``source``/``target`` values (e.g. "offers control over",
        "is calculated between"). Only plausible entity surface forms are
        recorded as dynamic-label candidates.
        """
        if not self.enabled or not name:
            return
        if not _is_plausible_entity(name):
            return
        key = name.strip().lower()
        if not key:
            return
        with self._lock:
            if key not in self._candidates:
                self._candidates[key] = LabelCandidate(
                    label=name.strip(),
                    display_name=name.strip(),
                    inferred_type=inferred_type or "",
                )
            c = self._candidates[key]
            c.occurrences += 1
            c.last_seen = time.time()
            c.doc_ids.add(self._current_doc or "")
        logger.debug("recorded unknown entity '%s' (occ=%d)",
                     name, self._candidates[key].occurrences)

    def step_document(self, doc_id: str = "") -> None:
        """Advance TTL, promote candidates, evict stale labels, persist, reload.

        Call once per document (after extraction). Non-fatal on any I/O error.
        """
        if not self.enabled:
            return
        self._current_doc = doc_id
        with self._lock:
            self._doc_counter += 1
            # Promotion: candidates seen in >= threshold docs -> active
            for key, cand in list(self._candidates.items()):
                if cand.occurrences >= self.promotion_threshold:
                    if cand.label not in self._active:
                        self._promote(key)
            # Eviction: labels unseen for ttl_docs documents -> drop
            for key, cand in list(self._candidates.items()):
                if self._doc_counter - cand.last_seen > self.ttl_docs:
                    self._active.discard(cand.label)
                    del self._candidates[key]
            active_snapshot = list(self._active)
            candidates_snapshot = {
                k: {
                    "label": c.label,
                    "display_name": c.display_name,
                    "inferred_type": c.inferred_type,
                    "occurrences": c.occurrences,
                    "first_seen": c.first_seen,
                    "last_seen": c.last_seen,
                }
                for k, c in self._candidates.items()
            }
            doc_counter = self._doc_counter
        # Persistence + daemon reload outside the lock (slow I/O).
        self._save_state(doc_counter, active_snapshot, candidates_snapshot)
        self._reload_daemon()

    # -- Internal -------------------------------------------------------------

    def _promote(self, key: str) -> None:
        if len(self._active) >= self.max_labels:
            self._evict_lru()
        label = self._candidates[key].label
        self._active.add(label)
        logger.info("promoted dynamic label '%s' for domain '%s' "
                    "(active=%d)", label, self.domain, len(self._active))

    def _evict_lru(self) -> None:
        """Evict the active label with the oldest last_seen timestamp."""
        oldest_key = None
        oldest_ts = None
        for key, cand in self._candidates.items():
            if cand.label in self._active:
                if oldest_ts is None or cand.last_seen < oldest_ts:
                    oldest_ts = cand.last_seen
                    oldest_key = key
        if oldest_key:
            self._active.discard(self._candidates[oldest_key].label)
            logger.info("evicted LRU dynamic label '%s'",
                        self._candidates[oldest_key].label)

    def _reload_daemon(self) -> None:
        """Push updated config to GPU daemon via existing /admin/reload.

        Fire-and-forget -- blocks no requests; failure is non-fatal (the daemon
        reads domain_config.yaml which does NOT carry dynamic labels, so reload
        is a no-op safety sync; dynamic labels live only in this process).
        """
        try:
            requests.post("http://localhost:8000/admin/reload", timeout=5)
        except Exception:  # noqa: BLE001 -- best-effort, never fatal
            pass

    def _state_path(self) -> str:
        return os.path.join(_STATE_DIR, f"{self.domain}_dynamic.json")

    def _save_state(self, doc_counter, active_snapshot, candidates_snapshot) -> None:
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            state = {
                "domain": self.domain,
                "doc_counter": doc_counter,
                "active": active_snapshot,
                "candidates": candidates_snapshot,
            }
            with open(self._state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to persist label state: %s", e)

    def _load_state(self) -> None:
        try:
            path = self._state_path()
            if not os.path.exists(path):
                return
            with open(path) as f:
                state = json.load(f)
            self._doc_counter = state.get("doc_counter", 0)
            self._active = set(state.get("active", []))
            for key, data in state.get("candidates", {}).items():
                self._candidates[key] = LabelCandidate(**data)
            logger.info("loaded dynamic labels for '%s': %d active, %d candidates",
                        self.domain, len(self._active), len(self._candidates))
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to load label state: %s", e)


# -- Module-level singleton registry ------------------------------------------

_providers: Dict[str, LabelProvider] = {}
_registry_lock = threading.Lock()


def get_provider(domain: str, cfg: Optional[dict] = None) -> LabelProvider:
    """Return the per-domain LabelProvider singleton (creates on first use)."""
    with _registry_lock:
        if domain not in _providers:
            if cfg is None:
                import domain_loader
                cfg = domain_loader.get_domain(domain)
            _providers[domain] = LabelProvider(domain, cfg)
        return _providers[domain]


def reset_provider(domain: str) -> None:
    """Drop a domain's cached provider (used by tests / manual rollback)."""
    with _registry_lock:
        _providers.pop(domain, None)
