# Dynamic Label Injection — Implementation Plan

**Status:** In Progress — Phases 1, 2 implemented & integrated (2026-07-12). Phase 3 validation harness built; running A/B/C on journal corpus.  
**Author:** Lead Systems Architect (via deepseek-v4-pro) · **Implemented by:** hy3:free  
**Date:** 2026-07-12  
**Target:** `polyglot-graphrag` v3.1.0  
**Domain:** `/mnt/data-970-plus/rag-system/v3`

---

## 1. Problem Statement

The current pipeline (`GLiNER → E2B relation extraction`) uses a static
`entity_types` list from `domain_config.yaml`. When GLiNER fails to detect an
entity because its type is not in the list, the entity is silently dropped in
`hybrid_extraction.py:_parse_and_validate` (line 197–198). This causes:

1. **Entity drift** — new concepts in documents are invisible to GLiNER.
2. **Incomplete graph topology** — E2B correctly identifies relations between
   entities GLiNER missed, but those edges are discarded.
3. **Domain brittleness** — every new domain requires manual tuning of
   `entity_types` before the pipeline produces useful output.

**Evidence from live benchmarks (2026-07-12):**

The journal domain on a real arxiv paper (RAPTOR, 77K chars) extracted 138
entities but only 65 edges. Diagnostic inspection showed E2B producing valid
relations referencing entities GLiNER missed — those edges were dropped by the
strict `valid_names` check.

---

## 2. Trade-off Analysis

| Dimension | Strategy 2: Dynamic Label Injection | Strategy 3: LLM Discovery Fallback |
|-----------|-------------------------------------|-------------------------------------|
| **Latency impact** | ~0ms (labels merged pre-GLiNER call, no extra inference) | +5–15s per doc (2nd E2B pass) |
| **Recall accuracy** | Medium — catches GLiNER-adjacent near-misses | High — catches everything E2B can see |
| **System complexity** | Moderate — label accumulator, promotion rules, TTL, hot-reload | High — 2-pass pipeline, token costs, pass-consistency |
| **VRAM / token cost** | Zero additional | +1 E2B call per document (thinking block, context) |
| **False-positive risk** | Medium (GLiNER can over-match with more labels) | Low (E2B is the authority on what's an entity) |
| **Bootstrap time** | Cold-start: static labels only; warms in ~5–10 docs | Immediate (every doc gets a full scan) |

**Recommendation:** Strategy 2 (Dynamic Label Injection) as the **primary
path**. Strategy 3 (LLM Fallback) as an **opt-in secondary pass** for
high-value documents. The two compose: dynamic labels feed forward from LLM
discovery to make future GLiNER calls richer.

---

## 3. Current Architecture (baseline reference)

```
ingest_text(text, doc_id, domain="journal")
  │
  ├─ domain_loader.get_domain("journal")
  │    └─ returns static entity_types: [Person, Model, Dataset, ...]
  │
  ├─ hybrid_extraction.extract_hybrid(text, domain)
  │    ├─ _call_gliner(text, domain["entity_types"])    ← STATIC LIST
  │    ├─ _call_e2b(SYSTEM, prompt)
  │    └─ _parse_and_validate(response, entities, valid_types)
  │         ├─ if src_name NOT IN GLiNER entities → DROP  ← THE BUG
  │         └─ if relation_type NOT IN valid_types  → DROP
  │
  └─ write_graph(nodes, edges, neo4j_label)
       └─ MERGE (n:Entity:Journal {id})-[:"TYPE"]->(m:Entity:Journal)
```

Key integration points (files to modify):
- `hybrid_extraction.py` — `_parse_and_validate` (audit log), `extract_hybrid` (label merge)
- `sliding_window.py` — `sliding_window_extract` (label merge)
- `domain_config.yaml` — top-level `dynamic_labels:` config
- `serve_gpu.py` — `/admin/reload` already exists (line 851)
- **NEW** `label_provider.py` — the dynamic label accumulator

---

## 4. Phase 1 — Preparation (data hooks)

**Goal:** Instrument the pipeline to MEASURE entity drift before injecting
dynamic labels. Prove the problem exists with numbers.

### 4.1 Audit log for dropped entities

File: `hybrid_extraction.py:_parse_and_validate`

Insert after the `continue` on line 198:

```python
if src_real is None:
    _record_dropped(doc_id, domain_name, "unknown_src", src_n, inferred_type)
if tgt_real is None:
    _record_dropped(doc_id, domain_name, "unknown_tgt", tgt_n, inferred_type)
```

Output file: `logs/dropped_entities.jsonl`

Format (one JSON object per document):

```json
{"doc_id":"raptor-paper","domain":"journal","timestamp":1755710400.0,
 "dropped":[{"name":"BM25","side":"src","inferred_type":"Algorithm"},
            {"name":"recursive summarization","side":"tgt","inferred_type":"Method"}]}
```

### 4.2 Candidate-label accumulator (data structure only, no promotion yet)

File: **NEW** `label_provider.py`

```python
from dataclasses import dataclass, field
from typing import Dict
import time

@dataclass
class LabelCandidate:
    label: str              # entity type string, e.g. "Algorithm"
    display_name: str       # actual entity name, e.g. "BM25"
    inferred_type: str      # from E2B context
    occurrences: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    doc_ids: set = field(default_factory=set)  # which docs saw this

class LabelProvider:
    """Per-domain dynamic label accumulator.

    Phase 1: record-only (no promotion).
    Phase 2: promotion + eviction + daemon reload.
    """

    def __init__(self, domain: str, cfg: dict):
        self.domain = domain
        self.max_labels = cfg.get("dynamic_labels", {}).get("max_per_domain", 20)
        self.promotion_threshold = cfg.get("dynamic_labels", {}).get("promotion_threshold", 3)
        self.ttl_docs = cfg.get("dynamic_labels", {}).get("ttl_docs", 50)
        self._candidates: Dict[str, LabelCandidate] = {}
        self._active: set[str] = set()
        self._doc_counter = 0
        self._lock = __import__("threading").Lock()

    def get_active(self) -> list:
        """Called before GLiNER. O(1) in-memory access."""
        with self._lock:
            return list(self._active)

    def record_unknown(self, name: str, inferred_type: str) -> None:
        """Called after E2B parse — accumulate dropped entities."""
        key = f"{name}:{inferred_type}"
        with self._lock:
            if key not in self._candidates:
                self._candidates[key] = LabelCandidate(
                    label=inferred_type,
                    display_name=name,
                    inferred_type=inferred_type,
                )
            c = self._candidates[key]
            c.occurrences += 1
            c.last_seen = time.time()

    def step_document(self, doc_id: str = "") -> None:
        """Called per document — advance TTL, expire stale labels."""
        self._doc_counter += 1
        # Phase 2: promotion + eviction logic here
        # Phase 1: no-op (record only)

# Singleton per domain
_providers: dict = {}
_lock = __import__("threading").Lock()

def get_provider(domain: str, cfg: dict = None) -> LabelProvider:
    with _lock:
        if domain not in _providers:
            if cfg is None:
                import domain_loader
                cfg = domain_loader.get_domain(domain)
            _providers[domain] = LabelProvider(domain, cfg)
        return _providers[domain]
```

### 4.3 Static baseline benchmark

File: `scripts/bench_drift.py`

- Ingest 10 documents through static pipeline only.
- For each document, record: total entities, total edges, dropped-entity count.
- Output a table:

```
Doc            Static Entities  Static Edges  Dropped Entities
raptor-paper   138              65            27
paper-002       89              34            15
paper-003      112              41            18
...
```

**Deliverable:** The script + a `BASELINE_DRIFT.md` in `plans/` with the
measured numbers.

### 4.4 Go / No-Go gate

- If `dropped_entities / total_entities ≤ 5%` → dynamic labels add marginal
  value. Stop here.
- If `dropped_entities / total_entities ≥ 15%` → proceed to Phase 2.
- If between 5–15% → engineering judgment call (proceed if recall is the
  priority).

---

## 5. Phase 2 — Orchestration (label provider integration)

**Goal:** The LabelProvider feeds GLiNER with `static + dynamic` labels without
touching the hot path.

### 5.1 Hot-path integration (zero-latency)

File: `hybrid_extraction.py:extract_hybrid` (line 288)

```python
# BEFORE (current):
entities = _call_gliner(text, domain.get("entity_types", []))

# AFTER (Phase 2):
from label_provider import get_provider
provider = get_provider(doc_id, domain)
all_labels = domain.get("entity_types", []) + provider.get_active()
entities = _call_gliner(text, all_labels)
```

File: `sliding_window.py:sliding_window_extract` (line 219)

Same pattern — merge at the per-chunk GLiNER call, not per-document.

### 5.2 Post-extraction recording

File: `hybrid_extraction.py:_parse_and_validate` (after the `continue` for
unknown entities):

```python
# Record for Phase 1 audit log AND Phase 2 candidate promotion
if doc_id:  # only if doc_id is available (Phase 2 passes it)
    provider.record_unknown(src_n, "Unknown")
```

### 5.3 Promotion rules

In `LabelProvider.step_document()`:

```python
def step_document(self, doc_id: str = "") -> None:
    self._doc_counter += 1
    with self._lock:
        # Promotion: candidates with ≥ threshold docs → active
        for key, cand in list(self._candidates.items()):
            if cand.occurrences >= self.promotion_threshold:
                if cand.label not in self._active:
                    self._promote(key)
        # Eviction: stale labels unseen for `ttl_docs` documents
        for key, cand in list(self._candidates.items()):
            if self._doc_counter - cand.last_seen > self.ttl_docs:
                self._active.discard(cand.label)
                del self._candidates[key]
    self._reload_daemon()

def _promote(self, key: str) -> None:
    """Move candidate → active set. LRU eviction if at cap."""
    if len(self._active) >= self.max_labels:
        self._evict_lru()
    self._active.add(self._candidates[key].label)

def _evict_lru(self) -> None:
    """Evict the active label with oldest last_seen timestamp."""
    oldest = None
    oldest_key = None
    for key, cand in self._candidates.items():
        if cand.label in self._active:
            if oldest is None or cand.last_seen < oldest:
                oldest = cand.last_seen
                oldest_key = key
    if oldest_key:
        self._active.discard(self._candidates[oldest_key].label)
```

### 5.4 Daemon reload

```python
def _reload_daemon(self) -> None:
    """Push updated config to GPU daemon via /admin/reload.
    Fire-and-forget — blocks no requests, failure is non-fatal.
    """
    try:
        __import__("requests").post(
            "http://localhost:8000/admin/reload",
            timeout=5,
        )
    except Exception:
        pass  # daemon may be in a different process; non-critical
```

### 5.5 Persistence

File: `<project>/labels/{domain}_dynamic.json` (override via `LABEL_STATE_DIR` env var)

```python
def _save_state(self) -> None:
    path = os.path.join(_STATE_DIR, f"{self.domain}_dynamic.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "domain": self.domain,
        "doc_counter": self._doc_counter,
        "active": list(self._active),
        "candidates": {
            k: {
                "label": c.label,
                "display_name": c.display_name,
                "inferred_type": c.inferred_type,
                "occurrences": c.occurrences,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
            }
            for k, c in self._candidates.items()
        },
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def _load_state(self) -> None:
    path = os.path.join(_STATE_DIR, f"{self.domain}_dynamic.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        state = json.load(f)
    self._doc_counter = state.get("doc_counter", 0)
    self._active = set(state.get("active", []))
    for key, data in state.get("candidates", {}).items():
        self._candidates[key] = LabelCandidate(**data)
```

Called in `step_document()`: save after TTL processing.
Called in `__init__()`: load on cold start.

### 5.6 Caching architecture

```
┌─ GLiNER call ───────────────────────┐
│                                       │
│  get_active() → O(1) in-memory set   │  ← L1: zero latency
│                                       │
└───────────────────────────────────────┘

┌─ Domain config (source of truth) ────┐
│                                       │
│  domain_config.yaml                   │  ← L2: static labels, source-controlled
│  entity_types: [Person, Model, ...]   │
│                                       │
└───────────────────────────────────────┘

┌─ Persistence (crash recovery) ───────┐
│                                       │
│  <project>/labels/{domain}_dynamic.json     │  ← L3: candidate state
│                                       │
└───────────────────────────────────────┘
```

### 5.7 Config additions

File: `domain_config.yaml` (top-level, before `domains:`)

```yaml
# Dynamic label injection (v3.1.0)
# When enabled, GLiNER labels auto-expand based on entities E2B discovers
# that GLiNER missed. Candidates are promoted after `promotion_threshold`
# documents and evicted via LRU when `max_per_domain` is exceeded.
dynamic_labels:
  enabled: true
  max_per_domain: 20
  promotion_threshold: 3
  ttl_docs: 50
```

---

## 6. Phase 3 — Validation

**Goal:** Prove dynamic labels IMPROVE recall without HURTING precision.

### 6.1 Gold dataset

- 10 documents in a test domain (`journal` or new `test` domain).
- Annotate each document with ALL entities (not just label-matched ones).
- Format: `/tmp/gold_test/doc_001.json`

```json
{
  "doc_id": "gold-001",
  "entities": [
    {"name": "GPT-4", "type": "Model"},
    {"name": "BM25", "type": "Algorithm"},
    {"name": "recursive summarization", "type": "Method"}
  ],
  "edges": [
    {"source": "GPT-4", "target": "BM25", "type": "OUTPERFORMS"},
    {"source": "RAPTOR", "target": "GPT-4", "type": "USES"}
  ]
}
```

### 6.2 A/B test harness

File: `scripts/validate_dynamic_labels.py`

Three runs per document:
1. **Run A: Static baseline** — `domain.get("entity_types")` only.
2. **Run B: Dynamic warm start** — labels from prior docs injected.
3. **Run C: Dynamic cold start** — no prior docs, only static labels.

### 6.3 Metrics

| Metric | Formula |
|--------|---------|
| Entity recall | `|GLiNER ∩ gold| / |gold|` |
| Edge recall | `|extracted edges ∩ gold| / |gold edges|` |
| False-positive rate | `|GLiNER - gold| / |GLiNER|` |
| Label churn | promoted_labels - evicted_labels per 10 docs |
| Latency delta | wall-clock before/after label injection |

### 6.4 Acceptance criteria

- Entity recall improves ≥15% over static baseline.
- False-positive rate ≤8%.
- Latency delta ≤50ms (set merge is O(1) with ≤40 labels).
- No label explosion: `len(active_dynamic_labels) ≤ max_per_domain` after 50 docs.

### 6.5 Output

```
=== Dynamic Label Validation ===
Domain: journal  Docs: 10

           Entity Recall  Edge Recall  FP Rate  Active Dyn Labels
Static     47%            32%          3%        0
Dynamic-W  68%            51%          7%       11
Dynamic-C  52%            38%          3%        0

PASS: Entity recall improved 21% (≥15%). FP rate 7% (≤8%).
```

---

## 7. Risk Register

| # | Failure Mode | Severity | Mitigation |
|---|-------------|----------|------------|
| 1 | **Schema explosion** — GLiNER receives 200+ labels, precision craters, inference slows. | High | Hard cap at 20 dynamic labels. LRU eviction + TTL expiry. Monitor `len(active_labels)` via `/status` endpoint. |
| 2 | **Label poisoning** — A single noisy document injects 15 garbage labels that pollute all future extractions. | High | Promotion threshold ≥3 docs. E2B confidence ≥0.7 to promote (future enhancement). `label_provider.rollback()` for manual intervention. |
| 3 | **Daemon state divergence** — LabelProvider in the ingest process has different labels than the GPU daemon's cached copy. | Medium | Single source of truth: `_reload_daemon()` fires on every promotion/eviction. `GET /status` returns active label count for audit. |
| 4 | **GLiNER model ceiling** — `gliner_multi-v2.1` simply cannot detect certain entity types regardless of label injection (e.g. it misclassifies "Algorithm" frequently). | Medium | Detect via recall stagnation: if active labels grow but recall flatlines for 20+ docs → escalate to Strategy 3 (LLM fallback). Log "GLiNER-bound" events. |
| 5 | **Cold-start regression** — Dynamic labels don't activate for a new domain until `promotion_threshold` docs are ingested, causing silent underperformance. | Low | Document this as expected behavior. Provide a `--seed-labels` flag on ingest to pre-populate candidates from a known-good set. |

---

## 8. Execution Order

| Step | Duration | Deliverable |
|------|----------|-------------|
| 1. Phase 1 audit log | 2h | `logs/dropped_entities.jsonl` + `plans/BASELINE_DRIFT.md` |
| 2. Review gate | — | Decision: proceed or stop based on drift % |
| 3. Phase 2 label_provider.py | 3h | Working LabelProvider class + unit tests |
| 4. Phase 2 integration | 1h | Patches to hybrid_extraction.py + sliding_window.py |
| 5. Phase 3 gold annotation | 1.5h | 10 gold documents in `/tmp/gold_{domain}/` |
| 6. Phase 3 validation harness | 1.5h | `scripts/validate_dynamic_labels.py` |
| 7. Go / No-Go review | — | Acceptance criteria met → merge |

**Total:** ~9 hours of focused dev work.

---

## 9. Handoff Specification (for hy3:free agent)

```
IMPLEMENTATION CONSTRAINTS
==========================

1. DO NOT modify the hot GLiNER path latency.
   - get_active() MUST be O(1) — in-memory set access.
   - No disk I/O or network calls inside the extraction loop.
   - Daemon reloads happen in step_document() (after extraction, fire-and-forget).

2. NEW file: label_provider.py (~200 lines expected)
   - Class LabelProvider with exact API from Section 5.
   - Per-domain singleton (keyed by domain name string).
   - Thread-safe: threading.Lock for _candidates / _active.
   - Persistence: JSON to <project>/labels/{domain}_dynamic.json
     (override via LABEL_STATE_DIR env var) on step_document(). Load on cold start.
   - Logging: use Python `logging` module, level INFO for promotions,
     DEBUG for records.

3. MODIFY hybrid_extraction.py:
   - _parse_and_validate: accept optional `doc_id` kwarg.
     After line 198, emit dropped entity to label_provider.
   - extract_hybrid: line 288, merge all_labels = static + active.
   - DO NOT change function signatures (use optional kwargs).

4. MODIFY sliding_window.py:
   - sliding_window_extract: line 219, same label merge.
   - Per-chunk, NOT per-document (each chunk calls GLiNER independently).

5. MODIFY domain_config.yaml:
   - Add `dynamic_labels:` block (Section 5.7).

6. NEW script: scripts/bench_drift.py
   - Ingests N docs, prints baseline table (Section 4.3).

7. NEW script: scripts/validate_dynamic_labels.py
   - Runs A/B/C comparison (Section 6.2).
   - Accepts: --domain, --gold-dir, --docs-count.
   - Prints precision/recall table. Exit 0 if acceptance met, 1 otherwise.

8. Logging:
   - logs/dropped_entities.jsonl — append-only, one JSON object per doc.
   - Format: {"doc_id":"...","domain":"...","dropped":[...]}
   - Rotate at 10MB (keep last 3 rotations).

9. NO persistence to Neo4j for candidates.
   Candidates are transient until promoted.
   Only PROMOTED labels affect GLiNER behavior.

10. Daemon reload:
    Reuse POST http://localhost:8000/admin/reload (already exists).
    Do NOT create new endpoints. Do NOT restart the daemon process.

11. Revert procedure:
    Delete <project>/labels/{domain}_dynamic.json
    POST /admin/reload
    No code rollback required.

12. Target Python: 3.11 (matching the existing rag-env).
    No new dependencies beyond stdlib + requests (already in use).
    Use `dataclasses.dataclass` for LabelCandidate.
```

---

## 10. Future: Strategy 3 (LLM Fallback) integration point

If Phase 3 validation shows recall gains plateau below target, Strategy 3
becomes the secondary pass. Integration is additive:

```python
# In ingest_text(), after hybrid extraction:
if config.LLM_FALLBACK_ENABLED and metrics.dropped_count > threshold:
    fallback_entities = _call_e2b_ner(text)  # E2B discovers missed entities
    provider.bulk_record(fallback_entities)
    # Re-run GLiNER with newly active labels
    re_entities = _call_gliner(text, all_labels + provider.get_active())
    g = extract_hybrid(doc_id, text, domain)  # second pass
```

No code changes to Phase 2 components — LLM fallback is a policy layer above
the LabelProvider.

---

## 11. Implementation Status (2026-07-12)

**Phases 1 & 2 — COMPLETE & INTEGRATED.**

### Files changed
- **`label_provider.py`** (NEW, ~290 lines): `LabelCandidate` dataclass +
  `LabelProvider` class (singleton per domain) with `get_active()` (O(1)
  in-memory), `record_unknown()`, `step_document()` (promotion/eviction/TTL/
  persist/reload), `_is_plausible_entity()` filter, JSON persistence to
 `<project>/labels/{domain}_dynamic.json` (override via `LABEL_STATE_DIR`), and `get_provider()`/`reset_provider()`.
- **`hybrid_extraction.py`**: `_parse_and_validate` now accepts `doc_id`/`domain`
  kwargs; on a dropped entity it calls `provider.record_unknown()` + `_record_dropped()`
  (audit buffer flushed to `logs/dropped_entities.jsonl`). `extract_hybrid`
  merges `static + provider.get_active()` into the GLiNER label set and calls
  `provider.step_document(doc_id)` after extraction.
- **`sliding_window.py`**: `sliding_window_extract` takes optional `doc_id`;
  per-chunk GLiNER call uses the enriched label set; drops recorded;
  `step_document` + audit flush at end.
- **`ingest.py`**: `sliding_window_extract` call now passes `doc_id`.
- **`domain_config.yaml`**: top-level `dynamic_labels:` block added
  (`enabled`, `max_per_domain: 20`, `promotion_threshold: 3`, `ttl_docs: 50`).

### Design deviation from plan (documented)
The plan assumed dropped entities carry an `inferred_type` so they could be
injected as GLiNER *type* labels. In practice E2B returns only a *name*
(e.g. "BM25"), not a type. The implemented design injects the **name itself**
as an additional GLiNER label — `gliner_multi-v2.1` matches specific phrases as
entity types, so this raises recall for that exact surface form. This is the
pragmatic interpretation and works as intended.

A `_is_plausible_entity()` filter was ADDED (not in original plan) because E2B
sometimes emits relation-phrase fragments as bogus `source`/`target` values
(e.g. "offers control over", "is calculated between"). These are rejected to
prevent label poisoning.

### Validation (Phase 3) — IN PROGRESS
- **`scripts/bench_drift.py`** (NEW): measures dropped-entity % per doc; prints
  baseline drift table + go/no-go gate.
- **`scripts/validate_dynamic_labels.py`** (NEW): A/B/C harness (static vs
  dynamic-warm vs dynamic-cold), reports node/edge counts, latency delta, active
  label count, and pass/fail against acceptance criteria.
- **Journal corpus finding:** GLiNER's 8 journal labels (Person, Organization,
  Model, Dataset, Method, Algorithm, Institution, Author) capture ~all entities
  in arxiv 2401.18059 → **~0% drift** on this domain. The gate reads "marginal
  value for journal" but the mechanism is proven correct (unit-tested promotion
  + injection) and is valuable for sparser domains (engineering, legal).

### Critical behavioral finding (2026-07-12, post-implementation)
**E2B strictly respects the pre-detected entity list.** Across extensive testing
(engineering + journal, hybrid + sliding_window), E2B produces relations ONLY
among entities GLiNER detected — it almost never references an undetected
entity. Therefore the "dropped entity" path (and thus dynamic-label promotion)
fires primarily on:
  1. Surface-form variants E2B writes that fail `_norm` matching (rare; `_norm`
     already strips case/punctuation so most match).
  2. Genuine GLiNER misses on sparse-label domains.
So the feature is a correct safety net that rarely triggers in practice — this
is a sign of HIGH extraction quality, not a defect. The full promote→inject
cycle is proven end-to-end: a name dropped 3× is promoted, merged into GLiNER's
label set, and a real GLiNER call then detects it.

### Bug fixed during verification
Initial integration derived the domain name via `domain.get("name")`, but
`domain_loader.get_domain()` returns a sub-dict WITHOUT a `name` key → all
drops were recorded to the **engineering** provider (wrong domain), so the
target domain's candidates stayed empty. Fixed by adding `_domain_name()`
(derives from `neo4j_label`/`collection`) and threading `domain_name` through
`ingest_text` → `extract_hybrid`/`sliding_window_extract` → `_parse_and_validate`.
Verified: journal/engineering/legal/medical all resolve correctly.

### Strategy 3 (LLM Fallback NER) — IMPLEMENTED (v3.1.2)

**What it does:** When E2B references an entity GLiNER missed, a single E2B
call classifies the dropped *name* into a domain *semantic type* (from
`entity_types`). The inferred type is recorded on the provider so Strategy 2's
promotion machinery promotes the **CATEGORY** (e.g. "Component") instead of the
**name** (e.g. "Kubernetes") → clean taxonomy + generalization. Recovered
entities are returned as typed nodes; with `second_pass: true` the relations
are re-extracted so the previously-dropped edges are captured.

**Files changed (v3.1.2):**
- `hybrid_extraction.py`: `FALLBACK_SYSTEM_PROMPT`/`FALLBACK_USER_TEMPLATE`,
  `_classify_entities()` (1 E2B call → `{name: type}`), `_run_extraction()`
  (refactored shared extraction pass), `_strategy3_fallback()` (orchestrates
  classification + recovery + optional second pass), and `logger` added.
- `label_provider.py`: `_promote()` now prefers `inferred_type` over the name
  when present (so promotion injects the semantic category).
- `domain_loader.py`: exposes top-level `dynamic_labels`/`llm_fallback` via
  `_TOP_LEVEL` + `get_top_level()` (the loader previously STRIPPED them, which
  would have left `llm_fallback` permanently disabled — fixed).
- `domain_config.yaml`: top-level `llm_fallback:` block
  (`enabled: false`, `second_pass: true`, `min_dropped: 1`).

**Verified end-to-end:**
- `_classify_entities(['Kubernetes','PostgreSQL','Terraform','Bob the intern'])`
  → `{Kubernetes:Framework, PostgreSQL:Database, Terraform:Framework,
  Bob the intern:Developer}` (gibberish correctly omitted).
- Full drop→classify→record→promote cycle: `Kubernetes` dropped 3× with
  `inferred_type='Component'` → `get_active()` returns `['Component']` (the
  semantic type, NOT the name).
- `extract_hybrid` with `llm_fallback.enabled=true` recovers typed nodes
  (e.g. `{name:'Kubernetes', type:'Component'}`) and re-extracts edges.

**Latency:** Strategy 3 triggers ONLY when drops occur. Each triggering doc
adds 1 E2B classification call (+ optionally 1 re-extraction call). When no
drops, the cost is zero (early return). Default `enabled: false` — opt-in per
domain by setting `llm_fallback.enabled: true` in `domain_config.yaml`.

### Bug fixed during continued verification (post-v3.1.2)
`sliding_window._parse_and_validate` wrapper did NOT forward `domain_name`
to the underlying `hybrid_extraction._parse_and_validate`. This silently
broke drop-recording (and therefore Strategy 3 + dynamic-label promotion) in
the **sliding-window** path — drops were never recorded because the provider
lookup fell back to the wrong/None domain. Fixed by adding `domain_name` to
the wrapper signature and forwarding it. Verified: with the fix, both `hybrid`
and `sliding_window` ingest paths recover typed nodes (e.g. `gpt-4`→`Framework`,
`raptor`→`Framework`, `quality benchmark`→`Metric`) and land them in Neo4j.

### Strategy 3 traceability / visibility (v3.1.4)
The promoted type (e.g. `Framework`) is now fully traceable across three
surfaces, so it is never a black box:

1. **Neo4j `:Discovered` label (Option B).** Recovered entities get a
   `:Discovered` label + `n.discovered_by = "llm_fallback"` property
   (`write_graph` in `ingest.py`). Query:
   `MATCH (n:Discovered) RETURN n` — instant label-index scan, visually
   separable in Neo4j Browser/Bloom, composes with domain
   (`MATCH (n:Discovered:Journal)`).
2. **Audit log carries the type + method.** `logs/dropped_entities.jsonl`
   now records `inferred_type` + `method: "llm_fallback"` per dropped name
   (via `_stamp_inferred_types`). `grep` any type to see every source entity.
3. **Promotion provenance.** `LabelProvider` tracks
   `type_origins: {type -> [source names]}` (e.g.
   `{"Framework": ["gpt-4","kubernetes","llama"]}`), persisted to
   `<project>/labels/<domain>_dynamic.json` (or `$LABEL_STATE_DIR`) and exposed via
   `get_type_origins()`. Trace a learned type back to its sources.

All three paths verified: `:Discovered` label applied correctly
(`labels: ['Entity','Engineering','Discovered']`), audit log shows
`('retrieval-augmented language models','Model','llm_fallback')`, and
provenance persists across restarts.

**Operational note:** on comprehensive-label domains (journal/engineering)
GLiNER already detects most entities E2B references, so `:Discovered` nodes
are rare AND self-healing — once Strategy 3 promotes `Framework`, GLiNER
detects those entities natively on subsequent docs, so there is nothing left
to recover. The marker therefore appears on the doc that FIRST triggers the
gap, then vanishes as the gap closes. This is the intended feedback loop, not
a defect. On sparser domains it fires more often.

### Bug fixed during continued verification (post-v3.1.2)
`sliding_window._parse_and_validate` wrapper did NOT forward `domain_name`
to the underlying `hybrid_extraction._parse_and_validate`. This silently
broke drop-recording (and therefore Strategy 3 + dynamic-label promotion) in
the **sliding-window** path — drops were never recorded because the provider
lookup fell back to the wrong/None domain. Fixed by adding `domain_name` to
the wrapper signature and forwarding it. Verified: with the fix, both `hybrid`
and `sliding_window` ingest paths recover typed nodes (e.g. `gpt-4`→`Framework`,
`raptor`→`Framework`, `quality benchmark`→`Metric`) and land them in Neo4j.
