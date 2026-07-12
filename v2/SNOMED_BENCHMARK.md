# SNOMED CT Extraction Benchmark — Results (2026-07-12)

**Source:** `/mnt/data-970/backup/SnomedCT_Germany-EditionRelease_PRODUCTION_20240515T120000Z.zip`
(518 MB → 3.6 GB uncompressed, 86 files, SNOMED CT Germany Edition 2024-05-15)

**Test goal (user-selected):** Can the E2B LLM extraction pipeline
(hybrid + sliding_window) extract medical entity relations from the
SNOMED prose (TextDefinition / Description files)?

## VERDICT: NO — LLM extraction is the WRONG tool for SNOMED CT

| Mode | Entities | Edges | Time | Notes |
|------|----------|-------|------|-------|
| `hybrid` (GLiNER+E2B) | **2** | **0** | 8.7s | GLiNER found only 2 (misclassified) |
| `sliding_window` | **2** | **0** | 16.7s | Same — no medical entities |

### Why it failed

1. **GLiNER vocabulary mismatch.** The `engineering` domain has entity
   types `[Microservice, Database, API, Metric, Developer, Framework,
   Component, Bug, PR, ADR]`. SNOMED prose ("A decrease in lower leg
   circumference due to recurrent ulceration...") contains NONE of these.
   GLiNER detected only 2 entities ("celiac plexus", "aortic plexus"),
   both misclassified as "Component".

2. **Relationships are already explicit.** SNOMED stores relations in
   `sct2_Relationship_*.txt` as TAB-separated
   `sourceId → destinationId via typeId` (e.g. `116680003` = associated
   finding). Natural-language extraction is redundant — the structure is
   already machine-readable.

3. **Definitions are noun-phrases, not prose.** `sct2_TextDefinition`
   rows are short phrases ("Introduction of a substance to the body"),
   not relationship-rich sentences ("X DEPENDS_ON Y").

### Gold check (structured tables)

From `sct2_Relationship_Snapshot` (active only):
- Of 60 sampled English TextDefinitions, only **1 concept**
  (`9000000000207008`) had active relationships — all pointing to
  `9000000000445007` (ASSOCIATED_WITH, likely "Clinical finding" root).
- SNOMED relationships are **sparse + hierarchical** (IS-A parent links),
  not the dense "service A depends on service B" graph our pipeline expects.

## RECOMMENDATION

Load SNOMED CT as a **structured knowledge graph**, NOT via LLM:

```
sct2_Concept_*.txt        → (:Concept {id, effectiveTime, active, moduleId})
sct2_Description_*.txt    → (:Description {term, languageCode}) -[:DESCRIBES]->(:Concept)
sct2_Relationship_*.txt  → (:Concept)-[:REL {typeId}]->(:Concept)
sct2_TextDefinition_*.txt  → (:Concept)-[:DEFINED_AS]->(text)
```

This needs:
- A **new domain** in `domain_config.yaml` (e.g. `medical`) with
  SNOMED entity/relation types — OR skip GLiNER entirely and parse
  the TSVs directly.
- A **TSV loader** (not ingest_text) that maps columns → Neo4j nodes/edges.
- Jina embeddings on Description terms for hybrid vector search.

The E2B pipeline is correct for its design domain (engineering
prose). SNOMED CT is a different data class (structured ontology)
and needs a different ingestion path.

## Artifacts
- `/tmp/snomed_sample.md` — 60 English TextDefinition rows (corpus used)
- `snomed_bench_prep.py` — loads TSVs, builds gold from relationships
- `snomed_bench_run.py` — runs hybrid + sliding_window on the corpus
