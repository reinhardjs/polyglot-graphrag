# /ask pipeline (full flow)

End-to-end request path for `POST /ask`. One HTTP call: cache → route →
retrieve → fuse/rerank → (optional) differential persistence → synthesize.

## Mermaid

```mermaid
flowchart TD
    A[POST /ask] --> B{skip_cache?}
    B -- no --> C{Qdrant cache hit?}
    C -- yes --> Z[return cached result source=cache]
    C -- no --> D
    B -- yes --> D

    D{req.crag?} -->|yes| CR[CRAG pipeline: adaptive route + corrective retrieval]
    CR --> ROUTE
    D -->|no| E{domain == snomed?}

    E -- yes --> F[SNOMED hybrid branch]
    F --> F1[snomed.retrieve / retrieve_temporal term-match + IDF]
    F --> F2[clinical_prose.retrieve dense semantic via daemon embed]
    F1 --> G
    F2 --> G

    E -- no (engineering/medical/...) --> H[resolve collections from profile]
    H --> H1[Qdrant search vec + Neo4j subgraph parallel threads]
    H1 --> G

    ROUTE[assemble pool: q_res + g_res] --> G
    G[normalise to records + tag _signal]
    G --> G2{dual-signal? tag _dual_signal}
    G2 --> I{dual_signal_only OR mode=differential?}
    I -- dual_only hard filter --> J[keep dual candidates only fallback if none]
    I -- no --> K
    J --> K

    K[cross-encoder rerank bge-reranker-v2-m3] --> K2[+ DUAL_SIGNAL_BOOST on dual]
    K2 --> K3[confidence = de-boosted raw score bands: high>=0.5 medium>=0.15]
    K3 --> L{min_confidence floor? drop below, keep >=1}
    L --> M[ranked contexts + contexts_meta + diagnoses]

    M --> N{mode == differential?}
    N -- yes --> O[persist to Neo4j: Differential -HAS_CANDIDATE-> DxCandidate -MATCHES_CONCEPT-> SnomedDescription]
    N -- no --> P
    O --> P{synthesize? or mode==differential}
    P -- yes --> Q[E2B synthesis -> ranked differential answer]
    P -- no --> R[return contexts only]
    Q --> S[return result + differential_persisted flag]
    R --> S
    S --> T[optional: store in Qdrant cache if synthesize & !skip_cache]
```

## Stage notes

| Stage | What | Key code |
|-------|------|----------|
| Cache | Qdrant `query_cache` lookup by `hash(query)` | `ask()` top |
| Route | `domain=snomed` → hybrid (graph+prose); else collections from profile | `ask()` branch |
| Retrieve | SNOMED term-match + clinical_prose dense; or Qdrant+Neo4j parallel | `get_retriever` |
| Fuse | normalise records, tag `_signal`, compute `_dual_signal` | fusion block |
| Rerank | cross-encoder bge-reranker-v2-m3, +0.15 dual boost | `_reranker.predict` |
| Confidence | raw (de-boosted) score → high/medium/low | `_confidence()` |
| Filter | `min_confidence` floor (keep ≥1) | fusion block |
| Persist | Neo4j Differential/DxCandidate (mode=differential only) | `_persist_differential` |
| Synthesize | E2B (Gemma) → answer; forced on for differential | `rag.synthesize` |

## Tuning knobs (config.py, no code edits)

- `CONFIDENCE_HIGH_THRESHOLD = 0.50`
- `CONFIDENCE_MEDIUM_THRESHOLD = 0.15`
- `DUAL_SIGNAL_BOOST = 0.15`
- live via `GET /config`

## Query params

- `domain` — snomed | engineering | medical | clinical_prose | ...
- `mode: "differential"` — ranked Dx + forced synthesis + Neo4j persist
- `dual_signal_only: true` — keep only dual-confirmed
- `min_confidence: "low"|"medium"|"high"` — floor the differential
- `synthesize: false` — return contexts only
- `skip_cache: true` — bypass/avoid cache
- `top_k` — candidates returned (default 5)
