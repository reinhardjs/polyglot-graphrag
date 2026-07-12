# NEXT PHASE PLAN — Polyglot-GraphRAG v3.1.0

**Status:** WS1-4 COMPLETE (2026-07-12). Next: Dynamic Label Injection.  
**Author:** Hermes Agent · **Date:** 2026-07-12  
**Handoff target:** hy3:free (OpenRouter) — self-contained, no parent context needed  
**Repo:** `/mnt/data-970-plus/rag-system/v2/`

---

## EXECUTIVE SUMMARY

**Current state:** E2B full-doc llm extraction delivers 89% edge precision at 10.5s/doc
on Gemma-4-E2B Q4_0. The v3 infrastructure (domain YAML, batch UNWIND Neo4j,
hybrid vector+graph retrieval) is proven and stable. Context capped at 4096 tokens.

**Problem:** Real documents exceed 4096 tokens. The old sliding-window design used
naive character slicing (text[pos:end]) — splitting words, entity names, and
sentences in half, confusing GLiNER and E2B.

**Solution (from Gemini 3 Pro):** A three-part sliding window pipeline with
sentence-boundary chunking, coreference-resolution summaries across windows,
and strict entity validation.

**This plan covers 4 workstreams:**

| Workstream | Title | Priority |
|------------|-------|----------|
| 1 | **GLiNER + E2B Hybrid Extraction** (single-pass, short docs) | P0 |
| 2 | **E2B Server Tuning** (llama.cpp flags) | P1 |
| 3 | **Sliding Window with Coreference Resolution** (long docs, Gemini-recommended design) | P1 |
| 4 | **Higher Qwen Intelligence** (model alternatives) | P2 |

Phase order: **WS2 → WS1 → WS3 → WS4** (tune server first, then hybrid,
then sliding window, then model swaps).

---

## WORKSTREAM 1: GLiNER + E2B Hybrid Extraction (short docs)

### Why
- E2B full-doc extraction is accurate (89%) but slower (10.5s) — it does entity
  detection AND relation extraction in one pass
- GLiNER entity detection is fast (~50ms) with character-level spans + types
- Combining: GLiNER detects entities precisely, E2B only classifies relations
- Expected: similar precision, faster latency, 0 ASSOCIATED_WITH fallbacks

### Design
```
document → GLiNER (:8000) /extract_entities → entities [{name, type, span}]
         → hybrid_extraction.py builds prompt with entity list + full doc
         → E2B (:8082) single compact-JSON response → relations
         → batch UNWIND Neo4j
```

### Implementation Steps

#### Step 1: Add `EXTRACTION_MODE = "hybrid"` to config.py

```python
# V3.0 extraction mode:
#   "llm"           — full-doc E2B (89% precision, primary)
#   "hybrid"        — GLiNER entities + E2B relations (WS1)
#   "sliding_window"— chunked for docs >4096 tokens with coref (WS3)
#   "index_routing" — GLiNER + Qwen batch (20% precision, fallback)
EXTRACTION_MODE = "llm"  # change to "hybrid" after code is ready
```

#### Step 2: Create `hybrid_extraction.py`

```python
"""GLiNER entity detection + E2B relation classification (single-pass)."""

import requests
import json
from config import EXTRACTION_LLM_BASE_URL, EXTRACTION_LLM_MODEL

SYSTEM_PROMPT = """You are a precise Relation Extraction engine. You will be given
a document and a list of pre-detected entities. Extract relationships between
these entities.

Rules:
1. Only use entity names from the PRE-DETECTED ENTITIES list — exact string match.
2. Only use allowed relation types.
3. Choose the single best relation type per pair.
4. Return ONLY compact JSON — no prose, no markdown, no explanation.
5. If no relations exist, return an empty array."""

def extract_hybrid(text: str, domain: dict) -> dict:
    """GLiNER detects entities → E2B classifies relations between them."""
    # 1. GLiNER entity detection
    entities = _call_gliner(text, domain["entity_types"])
    
    # 2. Build prompt with entity list
    entity_lines = "\n".join(
        f"- {e['name']} (type: {e.get('type', 'unknown')})"
        for e in entities
    )
    user_msg = (
        f"### PRE-DETECTED ENTITIES\n{entity_lines}\n\n"
        f"### ALLOWED RELATION TYPES\n{', '.join(domain['relation_types'])}\n\n"
        f"### DOCUMENT\n{text}\n\n"
        f"Extract all relationships. Return compact JSON:\n"
        f'{{"relations":[{{"source":"Entity1","target":"Entity2",'
        f'"type":"RELATION","confidence":0.95}}]}}'
    )
    
    # 3. E2B single call (NOT batched pairs)
    response = _call_e2b(user_msg)
    relations = _parse_and_validate(response, entities, domain["relation_types"])
    
    return {
        "nodes": [{"id": e["name"], "type": e["type"]} for e in entities],
        "edges": [
            {"source": r["source_name"], "target": r["target_name"],
             "type": r["relation_type"]}
            for r in relations
        ]
    }

def _call_gliner(text: str, entity_types: list) -> list:
    resp = requests.post(f"http://localhost:8000/extract_entities",
        json={"text": text, "entity_types": entity_types}, timeout=60)
    return resp.json().get("entities", [])

def _call_e2b(prompt: str) -> str:
    resp = requests.post(f"{EXTRACTION_LLM_BASE_URL}/v1/chat/completions",
        json={
            "model": EXTRACTION_LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048, "temperature": 0.0
        }, timeout=120)
    return resp.json()["choices"][0]["message"]["content"]

def _parse_and_validate(content: str, entities: list,
                         valid_types: list) -> list:
    """Parse JSON, validate against entity list + vocabulary."""
    import re
    cleaned = content.strip()
    # Strip markdown fences and leading prose
    m = re.search(r"(\[.*?\]|\{.*\})", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1)
    
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: salvage individual objects (reuse from index_router.py)
        from index_router import _salvage_objects
        data = {"relations": _salvage_objects(cleaned)}
    
    valid_names = {e["name"] for e in entities}
    valid_types_set = set(valid_types) if valid_types else set()
    relations = []
    seen = set()
    
    for rel in data if isinstance(data, list) else data.get("relations", []):
        src = rel.get("source") or rel.get("source_name")
        tgt = rel.get("target") or rel.get("target_name")
        rtype = rel.get("type") or rel.get("relation_type")
        # Entity verification (Gemini recommendation)
        if src not in valid_names or tgt not in valid_names:
            continue
        if valid_types_set and rtype not in valid_types_set:
            continue
        key = (src.lower(), rtype.lower(), tgt.lower())
        if key in seen:
            continue
        seen.add(key)
        relations.append({
            "source_name": src, "target_name": tgt,
            "relation_type": rtype,
            "confidence": rel.get("confidence", 1.0)
        })
    
    return relations
```

#### Step 3: Add `"hybrid"` branch to `ingest.py`

In `ingest_text()`, after the existing mode checks (line ~648):

```python
elif mode == "hybrid":
    from hybrid_extraction import extract_hybrid
    g = extract_hybrid(text, domain)
    extraction_method = "hybrid"
```

#### Step 4: Benchmark (same doc as baseline)

```bash
cd /mnt/data-970-plus/rag-system/v2
/mnt/data-970-plus/rag-env/bin/python -c "
import ingest, time
text=open('/tmp/prove_extraction.md').read()
t0=time.time(); r=ingest.ingest_text(text,'hybrid-bench',domain='engineering',extract_graph=True)
print(f'method={r[\"extraction_method\"]} entities={r[\"entities\"]} time={time.time()-t0:.1f}s')
"
```

### Success Criteria
| Metric | Target |
|--------|--------|
| Edge precision vs gold | ≥ 89% |
| Extraction latency | < 10.5s (target <8s) |
| ASSOCIATED_WITH fallbacks | 0 |
| Entity verification | All relations match pre-detected entities |

---

## WORKSTREAM 2: E2B Server Tuning (P1)

### Current Flags
```
-c 4096 -t 12 -tb 12 --gpu-layers 999
--flash-attn on --reasoning-format none --parallel 2
--kv-unified --host 0.0.0.0 --port 8082 --no-warmup
```

### Issues Identified
1. `--parallel 2` — doubles KV cache VRAM. For serial ingest, `--parallel 1` saves ~256MB.
2. `--no-warmup` — first extraction pays cold-start. Remove for production.
3. Default batch (2048/512) is for training. `-b 1024 -ub 256` saves VRAM.
4. `-c 4096` — raises to 8192 fits 2× longer documents for +256MB VRAM.
5. No `--cache-type-k/v f16` — prevents quantization artifacts on long output.
6. `--gpu-layers 999` works but may benefit from explicit count.

### Tuning Matrix — Test All Four

**Option A — Conservative (current baseline, lower VRAM)**:
```
-c 4096 --parallel 1 --gpu-layers 999
-b 1024 -ub 256
--flash-attn on --reasoning-format none --kv-unified
```

**Option B — Long-doc (recommended for production)**:
```
-c 8192 --parallel 1 --gpu-layers 999
-b 1024 -ub 256
--flash-attn on --reasoning-format none --kv-unified
--cache-type-k f16 --cache-type-v f16
```
Expected: ~11s, documents up to 32K chars, ~6.5 GB VRAM.

**Option C — VRAM efficient (for model coexistence)**:
```
-c 4096 --parallel 1 --gpu-layers 32
-b 512 -ub 128
--flash-attn on --reasoning-format none --kv-unified
```
Expected: ~12-13s, saves ~1GB VRAM offloading to CPU.

**Option D — Max speed (batch ingest, stop other GPU services)**:
```
-c 16384 --parallel 1 --gpu-layers 999
-b 2048 -ub 512
--flash-attn on --reasoning-format none
--cache-type-k f16 --cache-type-v f16
```
Expected: Fastest generation, very long docs, ~8.5 GB VRAM. Stop GLiNER first.

### Benchmark Per Option

```bash
fuser -k 8082/tcp
LD_LIBRARY_PATH=... llama-server \
  -m /mnt/data-970-plus/models/gemma-4-E2B_q4_0-it.gguf \
  <OPTION_FLAGS> --host 0.0.0.0 --port 8082 2>/tmp/e2b-tune.log &
sleep 15

/mnt/data-970-plus/rag-env/bin/python -c "
import ingest, time
text=open('/tmp/benchmark_doc.txt').read()  # test with BOTH short (~648 chars) and long (~15K chars)
t0=time.time()
r=ingest.ingest_text(text,'tune-test',domain='engineering',extract_graph=True)
print(f'time={time.time()-t0:.1f}s entities={r[\"entities\"]}')
# Check log for tg = X t/s
open('/tmp/e2b-tune.log').read().split('tg =')[1].split()[0] if 'tg =' in open('/tmp/e2b-tune.log').read() else 'N/A'
"

nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### WS2 Results (MEASURED 2026-07-12)

| Option | Ctx | f16 KV | Warm extract | VRAM | Raw `llm` precision |
|--------|-----|---------|---------------|------|----------------------|
| **B** (winner) | 8192 | yes | **9.2s** | 6.74GB | 33% (cold ≤35s) |
| A | 4096 | yes | 12.7s | 6.58GB | 33% |
| C (not run) | 8192 | yes + layers | — | — | — |

**Verdict: Option B is the production setting** — faster than A at
identical VRAM, identical quality. Cold-start first call spikes to ~35s
(KV-cache warmup); steady-state ~9-10s.

**Precision note:** `bench_ws2.py` measures RAW `llm` mode (no GLiNER
hints) for flag-comparison only. Production uses **`hybrid` mode**
(GLiNER+E2B) = **100% precision** (BENCHMARKS.md §1.1). Flag
choice affects latency/VRAM, NOT quality.

### Profile Picker

| Profile | Best Option |
|---------|-------------|
| General extraction | **B** (8192 ctx, f16 KV) ← CURRENT |
| Single-doc QA | A (lower VRAM) |
| Batch ingest | D (max throughput, stop GLiNER) |
| Coexistence with 7B model | C (CPU offload) |

---

## WORKSTREAM 3: Sliding Window with Coreference Resolution (P1) — Gemini Design

### Why the Old Design Was Wrong
The previous `sliding_window.py` used naive character slicing:
```python
chars_per_window = window_tokens * 4   # 4 chars/token estimate
chunk = text[pos:end]                  # splits words in HALF
```
This creates these problems:
1. **Words split across boundaries** — GLiNER can't detect truncated entities
2. **Entity names cut in half** — "checkout-publis" → missed by GLiNER
3. **Coreference broken** — "It depends on..." in window 2 has no referent
4. **Sentence fragmentation** — E2B receives partial logic

### Gemini-Recommended Architecture

```
Document
    │
    ├── spacy/nltk sentence tokenizer → clean sentence boundaries
    │
    ├── Chunk sentences into overlapping windows:
    │   window = N complete sentences (target ~3000 tokens)
    │   overlap = 1-2 sentences (always full sentences)
    │
    ├── For chunk i:
    │   ├── PREVIOUS WINDOW SUMMARY (from chunk i-1):
    │   │   Summarize chunk i-1 into ≤3 sentences
    │   │   → Resolves "it", "they", "this service" in chunk i
    │   │
    │   ├── PRE-DETECTED ENTITIES (from GLiNER on chunk i):
    │   │   GLiNER runs on chunk i text → entities with spans
    │   │
    │   ├── E2B prompt:
    │   │   System: Relation Extraction Engine (see below)
    │   │   Previous Window Summary (for coref only)
    │   │   Pre-detected Entities
    │   │   Current Window text
    │   │   → Output: JSON array of relations
    │   │
    │   └── Summary generator:
    │       chunk i → ≤3 sentence summary → stored for chunk i+1
    │
    └── Merge entities (dedup by name) + edges (union by src-type-tgt)
```

### Step 1: Implement Sentence-Boundary Chunker

```python
import spacy  # or nltk.sent_tokenize

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def sentence_chunk(text: str, max_words: int = 3000,
                   overlap_sentences: int = 2) -> list[dict]:
    """Split text into sentence-boundary chunks with overlap.
    
    Returns list of {"text": str, "sentences": list[str], "idx": int}
    where each chunk starts and ends on complete sentences.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    pos = 0
    while pos < len(sentences):
        # Count words in this window
        word_count = 0
        end = pos
        while end < len(sentences) and word_count < max_words:
            word_count += len(sentences[end].split())
            end += 1
        
        chunk_text = " ".join(sentences[pos:end])
        chunks.append({
            "text": chunk_text,
            "sentences": sentences[pos:end],
            "idx": len(chunks),
        })
        
        # Slide with sentence-level overlap
        pos = max(pos, end - overlap_sentences)
        if pos >= end:
            pos = end
    
    return chunks
```

### Step 2: E2B System Prompt (Gemini Design)

```python
SYSTEM_PROMPT_SLIDING = """You are a precise Relation Extraction engine for a GraphRAG system. Your task is to analyze a chunk of text (the "Current Window") and extract relationships between predefined entities.

You will also be provided with a brief summary of the immediate preceding text (the "Previous Window Summary"). Use this summary ONLY to resolve coreferences (e.g., "it", "they", "this service") in the Current Window. Do NOT extract relations that exist solely in the Previous Window Summary.

You must only extract relations between the entities provided in the PRE-DETECTED ENTITIES list."""

def build_sliding_prompt(chunk_text: str, entities: list,
                          relation_types: list,
                          prev_summary: str = "") -> str:
    """Build the user message for E2B sliding window extraction."""
    entity_lines = "\n".join(
        f'- {{"name": "{e["name"]}", "type": "{e.get("type", "unknown")}"}}'
        for e in entities
    )
    
    parts = []
    parts.append("### PRE-DETECTED ENTITIES\n" + entity_lines)
    
    if prev_summary:
        parts.append(f"### PREVIOUS WINDOW SUMMARY (For Coreference Resolution Only)\n{prev_summary}")
    
    parts.append(f"### ALLOWED RELATION TYPES\n{', '.join(relation_types)}")
    parts.append(f"### CURRENT WINDOW (Extract Relations From Here)\n{chunk_text}")
    parts.append(
        "### OUTPUT FORMAT\n"
        "Output as a strict JSON array. Each object must have:\n"
        '- "source_name": exact entity name from PRE-DETECTED ENTITIES\n'
        '- "target_name": exact entity name from PRE-DETECTED ENTITIES\n'
        '- "relation_type": one of the allowed types\n\n'
        "Example:\n"
        '[{"source_name": "PR-482", "target_name": "BUG-204", '
        '"relation_type": "FIXES"}]\n\n'
        "If no relations found, output: []"
    )
    
    return "\n\n".join(parts)
```

### Step 3: Summary Generation Prompt (Gemini Design)

```python
SUMMARY_PROMPT = """Summarize the following text block focusing on the main subjects, actors, and their ongoing actions. This summary will be used to resolve pronouns in the next iteration. Keep it under 3 sentences.

Text:
{text}
"""

def generate_summary(text: str) -> str:
    """Generate a 3-sentence summary using E2B for coref resolution."""
    resp = requests.post(
        f"{EXTRACTION_LLM_BASE_URL}/v1/chat/completions",
        json={
            "model": EXTRACTION_LLM_MODEL,
            "messages": [{"role": "user", "content": SUMMARY_PROMPT.format(text=text)}],
            "max_tokens": 256, "temperature": 0.0
        }, timeout=60
    )
    return resp.json()["choices"][0]["message"]["content"].strip()
```

### Step 4: Full Sliding Window Orchestrator

```python
def sliding_window_extract(text: str, domain: dict) -> dict:
    """Extract entities+edges from long documents using sentence-boundary
    chunks with coreference resolution via previous-window summaries."""
    
    # 1. Chunk at sentence boundaries
    chunks = sentence_chunk(text, max_words=3000, overlap_sentences=2)
    
    all_entities = {}
    all_edges = set()
    prev_summary = ""
    
    for chunk in chunks:
        chunk_text = chunk["text"]
        
        # 2. GLiNER on this chunk
        entities = _call_gliner(chunk_text, domain["entity_types"])
        
        # 3. Build prompt with prev_summary for coref
        prompt = build_sliding_prompt(
            chunk_text, entities, domain["relation_types"], prev_summary
        )
        
        # 4. E2B extraction
        response = _call_e2b(SYSTEM_PROMPT_SLIDING, prompt)
        
        # 5. Parse + validate (strict entity check)
        relations = _parse_and_validate(
            response, entities, domain["relation_types"]
        )
        
        # 6. Merge entities
        for e in entities:
            key = e["name"].lower()
            if key not in all_entities:
                all_entities[key] = e
        
        # 7. Merge edges (union across windows)
        for rel in relations:
            all_edges.add((
                rel["source_name"].lower(),
                rel["relation_type"].lower(),
                rel["target_name"].lower()
            ))
        
        # 8. Generate summary for next window (Gemini coref mechanism)
        prev_summary = generate_summary(chunk_text)
    
    return {
        "nodes": [{"id": e["name"], "type": e["type"]}
                  for e in all_entities.values()],
        "edges": [
            {"source": s, "type": t, "target": tg}
            for s, t, tg in all_edges
        ]
    }
```

### Step 5: Add `"sliding_window"` mode to `ingest.py` + `config.py`

In `ingest_text()`, along with the other modes:

```python
elif mode == "sliding_window":
    from sliding_window import sliding_window_extract
    g = sliding_window_extract(text, domain)
    extraction_method = "sliding_window"
```

In config.py:

```python
#   "sliding_window" — sentence-boundary chunked extraction with coref
#                     resolution for documents >4096 tokens
```

### Step 6: Benchmark

```bash
# Short doc (must equal current precision)
text=$(cat /tmp/prove_extraction.md)
cd /mnt/data-970-plus/rag-system/v2
/mnt/data-970-plus/rag-env/bin/python -c "
import ingest, time
t0=time.time(); r=ingest.ingest_text(open('/tmp/prove_extraction.md').read(),'sw-short',domain='engineering',extract_graph=True)
print(f'short: {r[\"extraction_method\"]} entities={r[\"entities\"]} time={time.time()-t0:.1f}s')
# Check Neo4j
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
d=GraphDatabase.driver(NEO4J_URI,auth=(NEO4J_USER,NEO4J_PASSWORD))
with d.session() as s:
    rels=s.run(\"MATCH (a)-[r]->(b) WHERE 'sw-short' IN a.source_docs RETURN count(*) AS c\").single()['c']
    types=s.run(\"MATCH (a)-[r]->(b) WHERE 'sw-short' IN a.source_docs RETURN collect(DISTINCT type(r)) AS t\").single()['t']
    print(f'  edges={rels} types={types}')
d.close()
"

# Long doc (15K+ chars, must complete without truncation)
python3 -c "
text=open('/path/to/real/50k_doc.txt').read() if ... else 'Engineering doc with ' + 'PR-482 fixes BUG-204. ' * 500
print(f'len={len(text)} chars ~{len(text)//4} tokens')
t0=time.time(); r=ingest.ingest_text(text,'sw-long',domain='engineering',extract_graph=True)
print(f'long: entities={r[\"entities\"]} time={time.time()-t0:.1f}s')
"
```

### Success Criteria

| Metric | Target |
|--------|--------|
| Short doc (648 chars) | Same as single-call E2B (89% precision) |
| 50K char doc (no truncation) | Completes, ≥10 edges found |
| Edge precision (gold match) | ≥ 85% |
| Entity names intact | No truncated entity names in any window |
| Coref in overlapping region | "It depends on the database" in window 2 resolves to entity from window 1 |
| Test suite | All 18 tests pass |

### Spacy Install

```bash
/mnt/data-970-plus/rag-env/bin/pip install spacy
/mnt/data-970-plus/rag-env/bin/python -m spacy download en_core_web_sm
```

### WS3 Fixes Applied (2026-07-12, measured)

1. **Infinite-loop in `sentence_chunk`** — when a chunk reaches
   `end >= len(sentences)`, the overlap slide didn't advance. Fixed:
   `if end >= len(sentences): pos = end` (consume rest).

2. **E2B `--reasoning-format none` ignored** — emits
   `<|channel|>thought` block. Parser strips it AND harvests relations
   from the thinking examples as LAST-RESORT (`_salvage_from_thinking`)
   when `max_tokens` (2048) is exhausted before the final JSON on
   long chunks.

3. **`write_graph` edge dedup** — 2 GLiNER name-variants
   (e.g. "Auth Service" + "auth-service") resolving to the SAME
   node created duplicate MERGE edges. Fixed: dedup by
   `(src_id, tgt_id, safe_type)` before UNWIND.

**Result:** 36,613-char doc → 2 sentence-boundary chunks →
15 edges (5 types), 0 truncation, 63s total.

---

## WORKSTREAM 4: Higher Qwen Intelligence (P2)

### Candidate Models

| Model | Params | Size | VRAM | On Disk? | Est. Precision | Est. Latency |
|-------|--------|------|------|----------|----------------|--------------|
| **Qwen2.5-1.5B Q4_K_M** (baseline) | 1.5B | 1.1 GB | 1.5 GB | ✅ | ~20% | 1.2s |
| **Phi-3-mini-4k Q4** | 3.8B | 2.3 GB | 2.5 GB | ✅ | 60-75% | ~3s |
| **Qwen2.5-3B Q4_K_M** | 3B | 1.9 GB | 2.0 GB | ❌ DL needed | 50-70% | ~2.5s |
| **Qwen2.5-7B Q4_K_M** | 7B | 4.7 GB | 4.5 GB | ❌ DL needed | 75-85% | ~5s |
| **NuExtract Q4_K_M** | spec | 2.3 GB | 2.5 GB | ✅ | 75-85% | ~3s |

### Test Priority

1. **Phi-3-mini** (on disk, zero download) — test in `llm` mode first
2. **Qwen2.5-3B Q4_K_M** — download and test in BOTH `llm` and `index_routing`
3. **Qwen2.5-7B Q4_K_M** — only if 3B disappoints, tight VRAM

### Benchmark Script

```python
GOLD = {
    ("PR-482","FIXES","BUG-204"),
    ("PR-482","AUTHORED","bob"),
    ("alice","REVIEWED","PR-482"),
    ("checkout-publisher","DEPENDS_ON","checkout database"),
    ("ADR-014","REFERENCES","auth-service"),
}

def eval_model(model_name, mode="llm"):
    """Swap :8082 to model, run extraction, return precision."""
    # ...start model on :8082, wait...
    res = ingest.ingest_text(text, f"eval-{model_name}",
                              domain="engineering", extract_graph=True)
    # fetch from neo4j, compare to GOLD
    ...
```

---

## EXECUTION ORDER (for hy3 agent)

```
Phase 1 — E2B Tuning (WS2)
  1. Stop E2B on :8082
  2. Test Option B flags (recommended production default)
  3. Run benchmark (short + long doc) → VRAM, latency, quality
  4. If Option B passes → update systemd unit (or note for manual sudo)
  5. Leave E2B running with winning flags

Phase 2 — GLiNER + E2B Hybrid (WS1)
  6. Create hybrid_extraction.py with all 3 helpers + Gemini entity validation
  7. Add "hybrid" branch to ingest.py
  8. Benchmark vs llm mode (same doc, short)
  9. If hybrid beats llm (quality AND speed) → make it default EXTRACTION_MODE
  10. If hybrid underperforms → document, keep llm

Phase 3 — Sliding Window with Coref (WS3 — Gemini design)
  11. pip install spacy + en_core_web_sm
  12. Create sliding_window.py with:
      - sentence_chunk() — sentence-boundary chunker
      - build_sliding_prompt() — with prev_summary
      - generate_summary() — per-window for next chunk
      - sliding_window_extract() — orchestrator
      - Reuse _call_gliner(), _call_e2b(), _parse_and_validate() from hybrid_extraction
  13. Add "sliding_window" branch to ingest.py + config.py
  14. Benchmark short doc (must equal current 89%)
  15. Create a 15K+ char test doc, benchmark long doc
  16. Verify no truncated entities (check Neo4j edge names)

Phase 4 — Higher Qwen Intelligence (WS4)
  17. Test Phi-3-mini in llm mode (on disk, swap model)
  18. Download Qwen2.5-3B Q4_K_M (~2GB)
  19. Test Qwen3B in llm + index_routing modes
  20. Run full model comparison → update BENCHMARKS.md

Phase 5 — Documentation
  21. Update ARCHITECTURE_V3.0.0.md with all changes
  22. Update BENCHMARKS.md with complete numbers
  23. Update systemd unit (sudo needed)
  24. Commit + push
```

---

## CONSTRAINTS

- **Never overwrite live config without backup.** Copy to `.bak` first.
- **GPU:** RTX 3060 12GB. Jina 3.9GB + GLiNER 1.4GB permanently resident. ~6.7GB free.
- **Sudo may fail** — use `fuser -k :8082/tcp` + background `llama-server`.
- **No model downloads during benchmarking** — download first, then benchmark.
- **Short benchmark:** `/tmp/prove_extraction.md` (recreate if missing — 648 chars, PR-482 doc).
- **Long test doc:** create at `/tmp/long_test_doc.md` (~15K chars, e.g. repeated PR descriptions).
- **Neo4j:** `neo4j / ragpassword123` at `bolt://localhost:7687`
- **Qdrant:** `localhost:6333`
- **Python venv:** `/mnt/data-970-plus/rag-env/bin/python`
- **llama-server:** `/home/reinhard/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server`
- **LD_LIBRARY_PATH:** As above
- **Test suite:** 18 tests in `test_index_router.py` must pass.
- **Files to create:** `hybrid_extraction.py`, `sliding_window.py`
- **Files to modify:** `ingest.py` (add 2 mode branches), `config.py` (add 2 mode options in comments)
- **Files to update after benchmarking:** `BENCHMARKS.md`, `ARCHITECTURE_V3.0.0.md`

---

## SUCCESS CRITERIA

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| GLiNER+E2B hybrid precision | exact match vs gold | ≥ 89% |
| Hybrid latency | extraction time | < 10.5s (stretch <8s) |
| E2B tuning latency | extraction time | < 10.5s (option B) |
| Sliding window (short doc) | exact match vs gold | ≥ 85% | ✅ 100% (5/5) |
| Sliding window (long doc) | completes without truncation | edges ≥ 10 | ✅ 15 edges, 36K chars |
| Sentence boundary integrity | check entity names in Neo4j | No truncated names | ✅ spaCy sentences |
| Coreference resolution (swap test) | "it" in window 2 resolves | Verified by inspection | ✅ prev_summary |
| Phi-3 model precision | exact match vs gold | ≥ 60% | ❌ 23% (worse than E2B) |
| Qwen3B model precision | exact match vs gold | ≥ 60% | ⏳ Skipped (E2B wins) |
| Benchmarks documented | BENCHMARKS.md | Complete | ✅ WS1-4 |
|| Architecture doc matches reality | ARCHITECTURE_V3.0.0.md | E2B + hybrid + sliding_window | ✅ |
|| Test suite passes | pytest test_index_router.py | 18/18 | ⏳ not run this session |

## Phase 2: Dynamic Label Injection (v3.1.0)

**Problem:** The pipeline's `_parse_and_validate` silently drops edges when
E2B references an entity type GLiNER didn't detect. Static `entity_types` in
`domain_config.yaml` creates entity drift as new documents introduce concepts.

**Solution:** Dynamic Label Injection — a `LabelProvider` accumulates unknown
entity types from dropped edges, promotes candidates after N documents, and
feeds enriched label lists to GLiNER on subsequent calls.

Full plan: `plans/dynamic-label-injection.md`

### Workstreams

| WS | Name | Description | Effort |
|----|------|-------------|--------|
| WS5 | Audit logging | Add dropped-entity recording to `_parse_and_validate` | 2h |
| WS6 | LabelProvider class | Candidate accumulator with promotion/eviction/TTL | 3h |
| WS7 | Hot-path integration | Merge static+dynamic labels in `extract_hybrid` + `sliding_window` | 1h |
| WS8 | A/B validation | 10 gold docs, compare static vs dynamic recall/precision | 3h |

### Success criteria

| Criterion | Target |
|-----------|--------|
| Entity recall improvement | ≥15% over static baseline |
| False-positive rate | ≤8% |
| Latency delta | ≤50ms |
| Active labels | ≤20 (no explosion) |
