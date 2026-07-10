# General-Purpose RAG — Domain Agnostic Architecture Plan

**Status:** planned
**Target:** v2.6.0 (after /ingest endpoint + multi-domain collections)
**Created:** 2026-07-10, updated 2026-07-11

> [!IMPORTANT]
> **Review amendments (2026-07-11):** This plan has been updated to address
> findings from codebase review. Key changes:
> - Chunking refactor target corrected: `serve_gpu.py:embed_late` + `serve_cpu.py:embed_late` (not `ask.py`)
> - `_build_profile()` must be chunking-strategy-aware (not hardcoded 2-sentence window)
> - Synthesis prompt consolidation: currently duplicated in 3 files (`ask.py`, `serve_gpu.py`, `retrieve_json.py`)
> - Profile window for Neo4j entity context should adapt to chunking strategy

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
| 7 | Sentence-based chunking | **serve_gpu.py:embed_late** + **serve_cpu.py:embed_late** | Legal needs paragraph; Medical needs section |
| 8 | Keyword-overlap Neo4j entry | ask.py:85-116 | Works for "bug-204" but not "chest pain" |
| 9 | GLiNER edge type = `ASSOCIATED_WITH` | serve_gpu.py:238 | All domains get same edge type |
| 10 | Profile window = 2 sentences | ingest.py:89 | Legal/medical may need larger context |
| 11 | No domain-specific metadata | ingest.py:371 | Legal: jurisdiction, court. Medical: patient_id, date |
| 12 | Synthesis prompt duplicated in 3 files | ask.py:177, serve_gpu.py:318, retrieve_json.py:27 | All 3 must be updated for domain-aware prompts |

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

### 2. `serve_gpu.py` + `serve_cpu.py` → Pluggable chunking strategies

> [!WARNING]
> **Corrected target.** The plans previously referenced `ask.py`'s `late_chunk()`,
> but no such function exists. The actual chunking logic is the regex
> `re.split(r'(?<=[.!?])\s+', ...)` in `serve_gpu.py:embed_late` (line 174)
> and `serve_cpu.py:embed_late` (line 121). Both must be refactored.

Replace hardcoded sentence split with a pluggable strategy:

```python
CHUNK_STRATEGIES = {
    "sentence":  _chunk_sentence,      # current regex split
    "paragraph": _chunk_paragraph,     # split on \n\n
    "section":   _chunk_section,       # split on ## headers
    "fixed":     _chunk_fixed,         # fixed-size windows with overlap
}
```

### 2b. `ingest.py` → `_build_profile()` must be chunking-strategy-aware

> [!WARNING]
> `_build_profile()` (ingest.py:89-111) uses a hardcoded 2-sentence window
> for Neo4j entity profiles. When medical docs use `section` chunking, a
> 2-sentence window loses the section header (e.g., "DIAGNOSIS:") that
> provides critical context.

**Fix:** Adapt the profile extraction to the chunking strategy:
- `sentence` strategy → current 2-sentence window (unchanged)
- `section` strategy → use the full section containing the entity
- `paragraph` strategy → use the full paragraph
- `fixed` strategy → use the fixed-size window containing the entity

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

### 5. Synthesis prompt consolidation + domain-aware prompts

> [!WARNING]
> The synthesis prompt is currently **duplicated in 3 files:**
> 1. `ask.py:synthesize()` (line 177-181) — CLI client
> 2. `serve_gpu.py:/ask` (line 317-318) — calls `rag.synthesize()` which is `ask.synthesize()`
> 3. `retrieve_json.py:_synthesize_nonstream()` (line 27-30) — Hermes path
>
> All three must be updated when switching to domain-aware prompts.

**Fix:** Extract prompt construction into a single function in `ask.py` (or a
new `prompts.py` module) that accepts the domain profile. `retrieve_json.py`
should import and reuse it instead of maintaining its own copy.

```python
# prompts.py (new)
def build_synthesis_prompt(query: str, contexts: list, profile: dict = None) -> str:
    """Build the synthesis prompt, domain-aware if profile is provided."""
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    if profile and profile.get("prompts", {}).get("synthesis"):
        return profile["prompts"]["synthesis"].format(context=ctx, question=query)
    # Default (engineering) prompt
    return (
        "Answer the question using ONLY the context. "
        "Cite source numbers. If missing, say so.\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"
    )
```

### 5b. GLiNER fallback → Domain edge types

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
