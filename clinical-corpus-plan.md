# Clinical Prose → Qdrant: Hybrid Diagnosis Upgrade

## Why this changes the game

SNOMED has concept names + relationships — it can tell you "fever IS A pyrexia"
but NOT "Influenza causes fever, cough, body aches." For that you need **prose**.

A Wikipedia medicine article for measles says:
> "Measles presents with fever, cough, runny nose, conjunctivitis,
> and a characteristic red-brown maculopapular rash."

Now when someone asks "fever and rash and cough", vector search on that
article matches 3/3 symptoms — meanwhile the SNOMED concept "Measles (disorder)"
contains NONE of those words in its name.

## Data sources (free, high-quality)

| Source | Size | Quality | License |
|--------|------|---------|---------|
| Wikipedia Medicine | ~25K articles (FA/GA) | Peer-reviewed | CC-BY-SA |
| NCBI Bookshelf | ~10K textbooks/chapters | Medical review | Varies (mostly free) |
| WHO fact sheets | ~500 disease briefs | Official | Public domain |
| MSD Manual (consumer) | ~3K topics | Professional review | Free access |

## Pipeline (single script)

```
Download → Clean → Chunk (512 tokens) → Embed (Jina-v3) → Upsert Qdrant
```

The existing Jina-v3 embedder (1024-dim, already loaded on GPU) handles
embedding. A new Qdrant collection `clinical_prose` stores the chunks.

## How /ask changes

Currently (term-match only):

```
Query → parse_keywords → CONTAINS match on SnomedDescription.term → score by IDF
```

Hybrid (after adding prose):

```
Query ─┬─→ [CLINICAL PROSE] Jina embed → Qdrant semantic search (top 30)
        │    Returns: article chunks describing disease presentations
        │
        └─→ [SNOMED GRAPH]  term-match on SnomedDescription.term  (top 30)
             Returns: concept names matching symptom keywords
         
         └─→ Merge / deduplicate → rerank → E4B synthesize

```

The combined context feeds E4B with BOTH:
- "Candidiasis of mouth co-occurrent with HIV (SNOMED 713497004)"  [graph]
- "AIDS is characterized by fever, lymphadenopathy, weight loss, and
  opportunistic infections such as oral candidiasis"  [prose]

E4B can then actually INFER: "The patient's progressive symptoms over
3 days match the natural history of HIV/AIDS as described in the
clinical literature, with oral candidiasis as a sentinel infection."

## Implementation steps

1. Fetch ~5K Wikipedia medicine articles (curated list: diseases+symptoms)
2. Clean HTML → markdown → plain text
3. Chunk at 512 tokens with 64-token overlap (using the existing chunker in domain_config.yaml)
4. Embed via Jina-v3 (already loaded in serve_gpu.py, 1024-dim)
5. Upsert to new Qdrant collection `clinical_prose`
6. Add `clinical_prose:` domain in domain_config.yaml
7. For `domain=snomed`: run BOTH Qdrant search AND Cypher traversal, merge results
8. Synthesis prompt updated: "You have SNOMED concept matches AND clinical
   literature describing the disease presentation. Produce a differential."

## Expected improvement

| Query | Current (term-match only) | Hybrid (SNOMED + prose) |
|-------|--------------------------|-------------------------|
| "fever and rash" | Fever concepts (finding) + rash concepts | Measles, Dengue, Scarlet fever, SLE |
| "chest pain and shortness of breath" | Chest pain (finding), Dyspnea (finding) | Angina pectoris, MI, PE, Pneumonia |
| Temporal AIDS case | Oral candidiasis variants | HIV/AIDS ranked #1, with supporting evidence |
| "fever and joint pain" | Fever (finding), Pain concepts | Lyme disease, Rheumatic fever, RA |

## What stays same

- Neo4j unchanged (SNOMED graph + engineering graph)
- Jina embedder unchanged
- E4B synthesis unchanged (just gets richer context)
- Reranker unchanged
- All existing domains still work
- No new infrastructure
