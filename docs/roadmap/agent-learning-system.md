# Plan: Self-Improving Knowledge Loop — Agent Learning System

**Status:** planned (not implemented)  
**Target:** v3.0.0  
**Created:** 2026-07-10, updated 2026-07-11

---

## Goal

Turn the RAG system from a static knowledge base into a **self-improving learning
system**. Every agent interaction feeds back into the KB — capturing what works,
what doesn't, corrections, and problem-solving patterns. The system gets smarter
with use, not just through manual document ingestion.

## The Loop

```
USER: "fix the checkout timeout bug"
  │
AGENT: [researches via /ask, tries approaches, solves it]
  │
AGENT: [captures the experience]
  ├─ POST /learn   → "checkout timeout caused by slow payment gateway → added circuit breaker"
  ├─ POST /feedback → "BUG-204 answer was accurate, no correction needed"
  └─ POST /feedback → "ADR-014 answer was wrong → actual migration used Kafka, not RabbitMQ"
       └─ correction ingested into KB → future answers reflect Kafka
  │
NEXT TIME:
  USER: "how do we handle payment gateway timeouts?"
  └─ /ask returns: "Use circuit breaker pattern (learned from checkout incident) [lesson-2026-07-10-001]"
     + references original BUG-204
     + references ADR-014 (now corrected to Kafka)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT (Hermes)                               │
│  ┌──────────┐   ┌───────────┐   ┌─────────────┐               │
│  │ Research │→→→│ Take      │→→→│ Capture     │               │
│  │ (/ask)   │   │ action    │   │ results     │               │
│  └──────────┘   └───────────┘   └──────┬──────┘               │
│                                         │                       │
│                    POST /feedback       │     POST /learn       │
└─────────────────────────────────────────┼──────────────────────┘
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RAG DAEMON (:8000)                              │
│                                                                  │
│  POST /feedback          POST /learn          GET /analytics    │
│  ├─ rating (good/bad)    ├─ problem           ├─ query_stats    │
│  ├─ correction_text      ├─ what_worked       ├─ gap_analysis   │
│  ├─ query_id             ├─ what_failed       ├─ top_failures   │
│  └─ notes                ├─ lessons_learned   └─ growth_report  │
│                          └─ tags                               │
│                                                                  │
│  STORES:                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │ query_log        │  │ lessons           │  │ corrections  │   │
│  │ (Qdrant)         │  │ (Qdrant + Neo4j)  │  │ (Qdrant)     │   │
│  │                  │  │                    │  │              │   │
│  │ every /ask call  │  │ problem→solution  │  │ user-provided│   │
│  │ with outcome     │  │ patterns, tagged  │  │ overrides    │   │
│  └─────────────────┘  └──────────────────┘  └──────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ EXISTING KB (unchanged)                                   │   │
│  │ engineering_chunks (Qdrant) + Entity nodes (Neo4j)        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## API contract

### `POST /feedback` — rate a query result

```json
// Request
{
  "query_id": "abc123",           // returned in /ask response
  "query": "who reported BUG-204?",
  "rating": "good",               // good | bad | partial
  "correction": null,             // only for bad/partial
  "notes": "answer was accurate"
}

// Response (good rating)
{
  "status": "ok",
  "action": "reinforced",
  "cache_boosted": true
}

// Response (bad rating with correction)
{
  "status": "ok", 
  "action": "corrected",
  "correction_ingested": true,
  "doc_id": "correction-20260710-001",
  "chunks": 3
}
```

### `POST /learn` — share a lesson learned

```json
// Request
{
  "problem": "Checkout 5xx cascade during payment gateway latency spike",
  "context_doc_ids": ["bug-204"],   // related existing docs
  "what_was_tried": [
    "Increased timeout thresholds → masked the problem",
    "Added retry with exponential backoff → made it worse under load"
  ],
  "what_worked": "Circuit breaker pattern: fail fast after 3 consecutive timeouts, fall back to cached pricing",
  "resolution_summary": "Deployed circuit breaker on checkout→gateway path. P99 latency dropped from 12s to 0.4s.",
  "tags": ["checkout", "gateway", "circuit-breaker", "resilience"],
  "severity": "SEV-2",
  "related_pr": "PR-483"
}

// Response
{
  "status": "ok",
  "doc_id": "lesson-20260710-001",
  "doc_type": "Lesson",
  "chunks_ingested": 8,
  "entities_extracted": 5,
  "linked_to": ["bug-204", "adr-014"]
}
```

### `GET /analytics` — knowledge health report

```json
// Response
{
  "period": "last_7d",
  "total_queries": 342,
  "ratings": {
    "good": 291,
    "bad": 34,
    "partial": 17
  },
  "accuracy": 0.85,
  "top_failures": [
    {"query": "how does the new auth work?", "failures": 12},
    {"query": "what's the deployment process?", "failures": 8}
  ],
  "knowledge_gaps": [
    "auth system", "deployment pipeline", "monitoring setup"
  ],
  "lessons_learned": 15,
  "corrections_applied": 7,
  "kb_growth": {
    "documents": "5→7",
    "lessons": "0→15", 
    "corrections": "0→7"
  }
}
```

## Data model

### Qdrant collections (new)

| Collection | Purpose | Dimension | Dense? |
|-----------|---------|-----------|--------|
| `query_log` | Every /ask call | 1024 | yes (query vector for similarity search) |
| `lessons` | Agent-learned patterns | 1024 | yes |
| `corrections` | User/agent-provided fixes | 1024 | yes |

### Query log payload

```json
{
  "query_id": "abc123",
  "query": "who reported BUG-204?",
  "timestamp": "2026-07-10T14:22:00Z",
  "source": "llm",                  // llm | cache
  "path": "hybrid",
  "rerank_scores": [0.977, 0.358],
  "n_contexts": 5,
  "rating": null,                   // filled by /feedback
  "correction_applied": false
}
```

### Lesson payload

```json
{
  "doc_id": "lesson-20260710-001",
  "doc_type": "Lesson",
  "problem": "Checkout 5xx cascade during payment gateway latency spike",
  "what_worked": "Circuit breaker pattern...",
  "tags": ["checkout", "circuit-breaker"],
  "severity": "SEV-2",
  "linked_docs": ["bug-204", "adr-014"],
  "ingested_at": "2026-07-10T14:30:00Z",
  "authority": "high"              // agent-learned lessons get higher authority
}
```

### Correction payload

```json
{
  "doc_id": "correction-20260710-001",
  "original_query_id": "abc123",
  "original_answer": "ADR-014 migration used RabbitMQ",
  "corrected_answer": "ADR-014 migration used Kafka (not RabbitMQ)",
  "correction_context": "...",
  "authority": "override",          // override > high > normal
  "linked_docs": ["adr-014"],
  "ingested_at": "2026-07-10T14:35:00Z"
}
```

## How it changes `/ask` behavior

### Today (v2.2.0)

```
query → cache check → Qdrant + Neo4j → rerank → synth → answer
```
Sources: only `engineering_chunks` (original KB).

### After v3.0.0

```
query → cache check → Qdrant + Neo4j → rerank → synth → answer
                        ↑              ↓
                  lessons included   corrections override reranker
                  in candidate pool  (higher auth = boosted score)

POST /ask response (new fields):
{
  ...existing fields...,
  "query_id": "abc123",           // for feedback correlation
  "confidence": 0.87,             // weighted avg of rerank scores
  "sources_used": {
    "kb_chunks": 3,               // from original KB
    "lessons": 1,                 // from agent-learned patterns
    "corrections": 1              // from user-provided fixes
  },
  "low_confidence": false         // true if confidence < 0.5 → trigger auto-enrichment
}
```

**Rerank weighting by authority:**

| Source | Authority | Boost |
|--------|-----------|-------|
| `correction` | `override` | 1.5× |
| `correction` | `high` | 1.25× |
| `lesson` | `high` | 1.2× |
| `lesson` | `normal` | 1.0× |
| `kb_chunk` (engineering_chunks) | `normal` | 1.0× |

Corrections override original KB. Lessons supplement it.

## Implementation phases

### Phase 1 — Basic feedback + query log (~2 hours)

1. Add `query_id` to every `/ask` response (uuid or hash-based).
2. Add `POST /feedback` endpoint.
3. Add `query_log` Qdrant collection.
4. Log every `/ask` call (query, answer, contexts, scores, timestamps).
5. Store ratings against query logs.

### Phase 2 — Learn + corrections (~2 hours)

1. Add `POST /learn` endpoint — ingest lessons into `lessons` collection + extract entities into Neo4j.
2. Add `corrections` collection.
3. Modify `/ask` rerank step to include lessons + corrections from Qdrant search (separate query against `lessons` collection, parallel to existing search).
4. Add authority-based weighting in reranker.
5. `POST /feedback` with correction → ingests into `corrections` collection.

### Phase 3 — Analytics (~1 hour)

1. Add `GET /analytics` endpoint.
2. Query log aggregation (success_rate, top_failures, knowledge_gaps).
3. Knowledge gaps identified by: queries with low rerank scores + negative ratings.

### Phase 4 — Auto-enrichment trigger (~2 hours)

1. When `/ask` returns `low_confidence: true`, agent can trigger auto-enrichment.
2. `POST /enrich {query, context_ids}` → agent researches externally, ingests results.
3. Re-query after enrichment — should now have high confidence.

## Integration with Hermes

### Hermes memory + RAG feedback working together

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Hermes       │────→│ /ask            │────→│ answer +     │
│ agent        │     │ (with query_id) │     │ confidence    │
│              │←────│                  │     │ + query_id   │
└──────┬───────┘     └─────────────────┘     └──────────────┘
       │
       │ Low confidence or wrong?
       ├─→ POST /feedback {"query_id":"...","rating":"bad","correction":"..."}
       │
       │ Solved a problem?
       └─→ POST /learn {"problem":"...","what_worked":"..."}
```

The agent's session memory (hermes memory tool) captures the *contextual* knowledge
("we fixed the checkout bug using circuit breakers"). The RAG feedback system
captures the *searchable* knowledge ("how to fix checkout gateway timeouts").

### What goes where

| Knowledge type | Store in | Why |
|---------------|----------|-----|
| "User prefers Celsius over Fahrenheit" | Hermes memory | Personal preference, not searchable KB |
| "Circuit breaker fixed checkout timeout" | RAG /learn | Team-wide searchable knowledge |
| "Port 8082 is E2B, 8084 is E4B" | Hermes memory | Environment config |
| "ADR-014 uses Kafka, not RabbitMQ" | RAG /feedback + corrections | Factual correction to KB |
| "Always test with synthesize=false first" | Hermes skill | Procedural workflow |

## VRAM impact

Minimal. The query_log, lessons, and corrections are stored in Qdrant (on-disk).
During `/ask`, the lessons + corrections search adds one extra Qdrant query
(milliseconds, parallel with existing search). No new models loaded.

| New collection | Vector dim | Additional search time | New models |
|---------------|------------|----------------------|------------|
| `query_log` | 1024 | 0 (write-only during /ask, read-only in analytics) | none |
| `lessons` | 1024 | ~0.03s (parallel Qdrant query) | none |
| `corrections` | 1024 | ~0.03s (parallel Qdrant query) | none |

## The learning flywheel

```
     ┌──────────────────────────────────┐
     │  MORE USAGE = SMARTER SYSTEM     │
     │                                  │
     │  Daily queries → identify gaps   │
     │  Corrections → fix wrong answers │
     │  Lessons → capture solutions     │
     │  Analytics → prioritize gaps     │
     │                                  │
     │  After 30 days:                  │
     │  - Accuracy: 85% → 94%          │
     │  - KB: 5 docs → 5 docs +        │
     │    23 lessons + 9 corrections    │
     │  - Zero-answer rate: 15% → 4%   │
     └──────────────────────────────────┘
```

## Files to create/modify

| File | Action |
|------|--------|
| `v2/feedback.py` | New: feedback handling, learn ingestion, analytics queries |
| `v2/serve_gpu.py` | Add `/feedback`, `/learn`, `/analytics` endpoints, modify `/ask` to include query_id, confidence, multi-source search |
| `v2/config.py` | Add `COLL_QUERY_LOG`, `COLL_LESSONS`, `COLL_CORRECTIONS`, authority weights |
| `docs/agent-learning-system.md` | This document |
| `CHANGELOG.md` | v3.0.0 entry |
| `v2/README.md` | New endpoints, updated architecture diagram |

## Dependencies

- No new Python packages.
- No new models.
- No VRAM increase.
- Qdrant `query_points` supports batched parallel queries (already used for lessons + corrections search).
- Neo4j for lesson entities (same extraction pipeline as KB docs).
