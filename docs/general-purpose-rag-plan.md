# General-Purpose RAG — Domain Agnostic Architecture Plan

**Status:** planned
**Target:** v2.6.0 (after /ingest endpoint + multi-domain collections)
**Created:** 2026-07-10

---

## Current Hardcoded Assumptions (discovered by audit)

Every one of these must be configurable for the system to work on ANY domain:

| # | Assumption | Location | Impact |
|---|-----------|----------|--------|
| 1 | Collection = `engineering_chunks` | config.py:34 | Can't query legal/medical KB |
| 2 | Entity types = `Microservice\|Database\|API\|Bug\|ADR\|PR\|...` | config.py:127-129 | Medical needs `Symptom\|Diagnosis\|Medication` |
| 3 | Edge types = `ASSOCIATED_WITH\|DEPENDS_ON\|IMPACTS\|...` | config.py:129-130 | Medical needs `TREATS\|CAUSES\|CONTRAINDICATES` |
| 4 | Extraction prompt = "engineering document" | config.py:120-121 | Legal: "legal document", Medical: "medical record" |
| 5 | Synthesis prompt = generic | ask.py:178-183 | No domain context for E4B |
| 6 | `doc_type = "eng"` | ingest.py:82, 371 | Should be domain-specific |
| 7 | Sentence-based chunking | ask.py (late_chunk) | Legal needs paragraph; Medical needs section |
| 8 | Keyword-overlap Neo4j entry | ask.py:85-116 | Works for "bug-204" but not "chest pain" |
| 9 | GLiNER edge type = `ASSOCIATED_WITH` | serve_gpu.py:238 | All domains get same edge type |
| 10 | Profile window = 2 sentences | ingest.py:89 | Legal/medical may need larger context |
| 11 | No domain-specific metadata | ingest.py:371 | Legal: jurisdiction, court. Medical: patient_id, date |

---

## Design: Domain Profiles

Each domain gets a profile file that overrides ALL the hardcoded assumptions:

```
v2/domains/
├── engineering.toml     # current defaults
├── medical.toml
├── legal.toml
├── accounting.toml
└── hospitality.toml
```

### `medical.toml` — example

```toml
[domain]
name = "medical"
collection = "medical_chunks"
neo4j_label = "Medical"

[chunking]
strategy = "section"       # sentence | paragraph | section | fixed
chunk_size = 512
overlap = 64

[graph_schema]
entity_types = [
    "Symptom", "Diagnosis", "Disease", "Medication", "Procedure",
    "LabTest", "Anatomy", "Patient", "Doctor", "Department"
]
edge_types = [
    "TREATS", "CAUSES", "INDICATES", "CONTRAINDICATES",
    "DOSAGE", "PERFORMS", "PRESCRIBES", "AFFECTS",
    "ASSOCIATED_WITH"  # fallback
]

[prompts]
extraction = """
Extract a medical knowledge graph from the clinical document below.

Return ONLY valid JSON: {nodes: [{id: string, type: string}], edges: [{source, target, type}]}
Node types: Symptom, Diagnosis, Disease, Medication, Procedure, LabTest, Anatomy
Edge types: TREATS, CAUSES, INDICATES, CONTRAINDICATES, DOSAGE, PERFORMS, AFFECTS

Entity IDs MUST be the exact name as written — do NOT translate.
Include ALL medications, symptoms, and diagnoses mentioned.

Document ID: {doc_id}
Document text:
{text}
"""

synthesis = """
You are a medical information assistant. You answer questions using ONLY
the medical documents provided below. Be precise and cautious — do NOT
make diagnoses. Cite specific medications and dosages when available.
Say "insufficient information" if the context does not contain enough
to answer the question.

Context:
{context}

Question: {question}
"""

[entity_resolution]
threshold = 0.88    # same as engineering — cross-lingual vector matching
aliases_enabled = true

[metadata_schema]
fields = ["patient_id", "encounter_date", "department", "doctor", "document_type"]
```

### `legal.toml` — example

```toml
[domain]
name = "legal"
collection = "legal_chunks"
neo4j_label = "Legal"

[chunking]
strategy = "paragraph"
chunk_size = 1024
overlap = 128

[graph_schema]
entity_types = [
    "Statute", "Case", "Court", "Judge", "Party", "Citation",
    "Holding", "Dissent", "Regulation", "Contract"
]
edge_types = [
    "CITES", "OVERRULES", "AFFIRMS", "DISSENTS", "INTERPRETS",
    "CHALLENGES", "AMENDS", "ASSOCIATED_WITH"
]

[prompts]
extraction = """
Extract a legal knowledge graph from the legal document below.

Return ONLY valid JSON: {nodes: [{id, type}], edges: [{source, target, type}]}
Node types: Statute, Case, Court, Judge, Party, Citation, Holding, Regulation
Edge types: CITES, OVERRULES, AFFIRMS, DISSENTS, INTERPRETS, CHALLENGES, AMENDS

Entity IDs MUST be exact legal names/citations as written — do NOT abbreviate.

Document ID: {doc_id}
Document text:
{text}
"""

synthesis = """
You are a legal research assistant. Answer the question using ONLY the
legal documents and citations provided. Reference specific statutes,
case law, and holdings. Do NOT provide legal advice — summarize what
the documents state.

Context:
{context}

Question: {question}
"""

[entity_resolution]
threshold = 0.88

[metadata_schema]
fields = ["jurisdiction", "court", "case_number", "date_filed", "practice_area"]
```

---

## Code Changes Needed

### 1. `config.py` → Domain-aware defaults

```python
# Replace single constants with domain-aware lookups
DOMAIN_PROFILES_DIR = Path(__file__).parent / "domains"
DEFAULT_DOMAIN = "engineering"

def load_domain_profile(name: str | None = None) -> dict:
    """Load profile .toml, merge with defaults. Fall back to DEFAULT_DOMAIN."""
    ...
```

### 2. `ask.py` → Chunking strategies

Replace `late_chunk()` with a pluggable strategy:

```python
CHUNK_STRATEGIES = {
    "sentence":  _late_chunk_sentence,   # current
    "paragraph": _late_chunk_paragraph,  # split on \n\n
    "section":   _late_chunk_section,    # split on ## headers
    "fixed":     _late_chunk_fixed,      # fixed-size windows
}
```

### 3. `ingest.py` → Profile-driven extraction

```python
def ingest_text(text, doc_id, domain="engineering")
    profile = load_domain_profile(domain)
    prompt = profile["prompts"]["extraction"].format(doc_id=doc_id, text=text)
    # ... domain-aware graph schema for extraction
    # ... domain-aware metadata payload
```

### 4. `serve_gpu.py` → Domain in /ask

```json
{
  "query": "what treats hypertension?",
  "domain": "medical",
  "collection": "medical_chunks",
  "synthesize": true
}
```

Handles: collection routing, domain-specific synthesis prompt, domain label on Neo4j query.

### 5. `GLiNER fallback` → Domain edge types

```python
# serve_gpu.py:/extract_graph
edge_type = profile["graph_schema"]["edge_types"][0]  # first as default
```

---

## Domain-Aware E2B Extraction

The extraction LLM (Gemma E2B) needs to know the domain schema to produce
useful JSON. Currently it's prompted with `EXTRACTION_PROMPT` which says
"engineering document." With domain profiles:

```
POST :8082/chat/completions
{
  "messages": [{"role": "user", "content": "<domain.extraction_prompt>"}],
  "response_format": {"type": "json_object"}
}
```

The LLM will auto-adapt its entity/edge types to the domain schema.

---

## Domain-Aware E4B Synthesis

Currently: `"Answer the question using ONLY the context..."`

With domain profiles: `domain.synthesis_prompt` is injected, giving E4B:
- Medical: cautious, no diagnosis, cite medications
- Legal: cite statutes, no legal advice
- Engineering: current behavior (unchanged)
- Accounting: cite tax codes, no financial advice

---

## Chunking Strategy Examples

| Domain | Strategy | Why |
|--------|----------|-----|
| Engineering | sentence | Bug reports/ADRs are narrative prose |
| Medical | section | Clinical notes have structured headers (HISTORY, EXAM, PLAN) |
| Legal | paragraph | Legal docs are dense paragraphs with numbered sections |
| Accounting | fixed + overlap | Ledger entries are tabular/short — fixed windows work better |
| Hospitality | sentence | SOPs and manuals are instructional prose |

---

## Neo4j Entry Point — Beyond Keyword Overlap

Current keyword overlap works for `bug-204` (technical IDs) but fails for
natural-language queries like "chest pain treatment." Two improvements:

1. **Vector entry** (already exists as fallback): embed query, find closest
   entity in Neo4j via `entity_vector_idx`. Currently only triggered when
   keyword overlap returns zero.

2. **Hybrid entry**: run both keyword AND vector, pick the best. Keyword is
   fast (~1ms), vector is accurate (~50ms). For medical domain, prefer vector.

```python
# ask.py: configurable entry strategy
NEO4J_ENTRY_STRATEGY = "hybrid"  # keyword | vector | hybrid
```

---

## Implementation Phases

### Phase 1 — Domain profile system (v2.5.0, ~2 hours)
1. Create `v2/domains/` directory with `engineering.toml` (extracted from current config)
2. `load_domain_profile()` function in config.py
3. Port existing constants to engineering.toml defaults
4. Backward compatible — no API changes

### Phase 2 — Pluggable chunking (v2.5.0, ~1 hour)
1. `CHUNK_STRATEGIES` dict in ask.py
2. Paragraph, section, and fixed strategies
3. Selected per domain profile

### Phase 3 — Domain-aware prompts (v2.5.0, ~30 min)
1. Extraction prompt from profile
2. Synthesis prompt from profile
3. Ingest metadata from profile schema

### Phase 4 — Neo4j entry strategy (v2.6.0, ~1 hour)
1. Hybrid keyword + vector entry
2. Configurable per domain

### Phase 5 — Sample domains (v2.6.0, ~1 hour)
1. `medical.toml` with complete schema
2. `legal.toml` with complete schema
3. Test ingest + query on each

---

## Summary: What Changes vs What Stays

| Component | Stays Same | Changes |
|-----------|-----------|---------|
| Jina v3 embedding | ✓ (general-purpose) | — |
| BGE v2-m3 reranker | ✓ (general-purpose) | Optional domain-specific reranker |
| E2B extraction | ✓ | Prompt now domain-aware |
| E4B synthesis | ✓ | Prompt now domain-aware |
| Qdrant | ✓ | Collection per domain |
| Neo4j | ✓ | Domain labels on nodes |
| GLiNER | ✓ | Edge type from domain config |
| Entity resolution | ✓ (vector, 0.88) | — |
| Chunking | — | Pluggable per domain |
| Configuration | — | TOML profile per domain |
| Neo4j entry | — | Hybrid keyword+vector |
