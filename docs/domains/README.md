# Domain Profiles — GraphRAG v2.6.0

Domain profiles are TOML files in `v2/domains/` (one per domain). They make the
system drop-in usable for any knowledge domain without code changes — chunking
strategy, entity/edge vocabulary, extraction & synthesis prompts, Neo4j labels,
entry strategy, and metadata schema are all declared here and read at runtime by
`config.load_domain_profile()`.

## Selecting a domain

Pass `domain` on any API call. The profile drives collection routing, chunking,
extraction prompt, synthesis prompt tone, graph labels, and entry strategy.

```bash
# Ingest into the medical domain (routes to medical_chunks, section chunking,
# medical extraction prompt, hybrid entry)
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"...","doc_id":"enc-1","domain":"medical",
       "metadata":{"patient_id":"P-42","title":"Encounter note"}}'

# Query the medical domain (hybrid entry, clinical-tone synthesis)
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"what treats hypertension?","domain":"medical","synthesize":true}'
```

If `domain` is omitted, the system behaves like v2.5.0 (engineering defaults:
sentence chunking, generic extraction/synthesis prompts, keyword entry,
`engineering_chunks` collection).

## Profile schema

```toml
[domain]
name = "medical"                 # key used by `domain=`; also the Neo4j label base
collection = "medical_chunks"    # Qdrant collection this domain writes/reads
neo4j_label = "Medical"          # :Entity:Medical label for graph isolation
description = "..."              # free text

[chunking]                       # REQ-3 — pluggable chunking
strategy = "section"             # sentence | paragraph | section | fixed
chunk_size = 512                 # token budget per chunk (for paragraph/section/fixed grouping)
overlap = 64                     # sliding-window overlap (fixed strategy)
section_header_prefix = "##"     # header line prefix preserved (section strategy)

[graph_schema]                   # entity/edge vocabulary
entity_types = ["Symptom", "Diagnosis", "Medication", "Lab", "Procedure", "BodyPart"]
edge_types = ["TREATS", "INDICATES", "CAUSES", "ASSOCIATED_WITH"]

[extraction]                     # REQ-4 — E2B extraction prompt (LLM primary path)
prompt = """
Extract a medical knowledge graph from the document below.
Entities: extract names EXACTLY as they appear — do NOT normalize.
Return ONLY valid JSON, no prose, no markdown:
{"nodes":[{"id":"ExactEntityName","type":"Symptom|Diagnosis|Medication|Lab|Procedure|BodyPart"}],
 "edges":[{"source":"entity_a","target":"entity_b","type":"TREATS|INDICATES|CAUSES|ASSOCIATED_WITH"}]}
Document ({doc_id}):
{text}
"""

[synthesis]                      # REQ-5 — E4B answer prompt (shared via prompts.py)
role = "clinical assistant"
prompt = """
You are a clinical assistant. Answer using ONLY the provided clinical documents.
Always cite the source [n]. If the context lacks the answer, say so. Be concise.

Context:
{context}

Question: {query}

Answer (cite clinical documents):
"""

[metadata_schema]                # REQ-7 — optional domain metadata fields
fields = ["patient_id", "title", "visit_date", "author", "doc_type"]

[neo4j_entry]                    # REQ-6 — graph entry strategy
strategy = "hybrid"              # keyword | vector | hybrid
max_k_hops = 2
```

### Field reference

| Section | Field | Meaning |
|---------|-------|---------|
| `[domain]` | `name` | Domain key. Must match the filename stem (`medical.toml` → `name="medical"`). |
| | `collection` | Qdrant collection for this domain's chunks. |
| | `neo4j_label` | Neo4j `:Entity:<Label>` label — isolates graph traversal per domain. |
| | `description` | Free text shown in `GET /profiles`. |
| `[chunking]` | `strategy` | `sentence` (1 chunk/sentence), `paragraph` (1/para), `section` (split on `section_header_prefix`, header kept), `fixed` (sliding window of `chunk_size` tokens, `overlap` overlap). |
| | `chunk_size` | Token budget. For `fixed` it sizes the window; for `paragraph`/`section` it only sub-splits an oversized unit. |
| | `overlap` | Sliding-window overlap in tokens (fixed strategy). |
| | `section_header_prefix` | Markdown heading prefix marking section boundaries (section strategy). |
| `[graph_schema]` | `entity_types` | Allowed entity types (drives GLiNER fallback labels). |
| | `edge_types` | Allowed relationship types (documentation/validation aid). |
| `[extraction]` | `prompt` | Full E2B prompt. May contain `{doc_id}` and `{text}` (safe-substituted). Literal `{...}` JSON is allowed (won't break — we do not use `str.format`). |
| `[synthesis]` | `role` | Persona used when no full `prompt` template is given. |
| | `prompt` | Full E4B prompt with `{context}` and `{query}` placeholders (safe-substituted). |
| `[metadata_schema]` | `fields` | Declared metadata keys. Unknown keys passed at ingest are **warned (not dropped)** so ingestion never fails on schema drift. |
| `[neo4j_entry]` | `strategy` | `keyword` (fast token overlap; auto-falls back to vector when zero overlap), `vector` (embed query+names, cosine), `hybrid` (both in parallel; keyword when overlap≥2 else vector). |
| | `max_k_hops` | Graph traversal depth hint (currently informational; traversal uses `config.GRAPH_HOPS`). |

## Available profiles

| Domain | Collection | Chunking | Entry | Notes |
|--------|-----------|----------|-------|-------|
| `engineering` | `engineering_chunks` | sentence | keyword | Default; legacy-compatible. |
| `medical` | `medical_chunks` | section | hybrid | Clinical-tone synthesis. |
| `legal` | `legal_chunks` | section | hybrid | Statute/case vocabulary. |
| `accounting` | `accounting_chunks` | paragraph | keyword | Financial statements. |
| `hospitality` | `hospitality_chunks` | section | hybrid | Menus, SOPs, guest data. |

## Adding a new domain

1. Copy `v2/domains/engineering.toml` to `v2/domains/<name>.toml`.
2. Set `name`, `collection`, `neo4j_label` (unique per domain).
3. Tune `[chunking]`, `[extraction]`, `[synthesis]`, `[metadata_schema]`, `[neo4j_entry]`.
4. Create the Qdrant collection (auto-created on first ingest) — or pre-create via `docker compose exec qdrant ...`.
5. Ingest with `domain=<name>`. No code changes, no restart.

## API: list profiles

```
GET /profiles      # returns all loaded domain profiles (name, collection, label, description)
```
