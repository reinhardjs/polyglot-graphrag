# Sprint 2: Better Proxy Metrics (P1)

> Target: v1.0.7 → v1.1.0
> Goal: Fix the structural biases in local proxy metrics so they reflect actual RAG quality, not embedding artifacts.

--------------------------------------------------------------------
## Problem

The current local proxy metrics have structural biases that obscure real improvements:

| Metric | Bias | Effect |
|--------|------|--------|
| `context_precision_local` | Cosine ≥ 0.35 vs ground_truth | 512-token chunks vs 50-char summaries → structurally low cosine |
| `answer_relevancy_local` | Cosine(question, answer) | Template questions ("what does X describe?") vs novel answer content → low cosine |
| `context_recall_local` | Term-level Jaccard | Reasonable, no major bias |

These biases mean:
- We can't tell if a real improvement happens (AR stuck at 0.56 even with better answers)
- The release gate floors context_precision at 0.25 (admitting the bias)
- Fine-tuning the system feels blind

--------------------------------------------------------------------
## 2a: Sentence-Level Context Precision

### Current implementation (evaluate_pipeline.py:144-157)
```python
def context_precision_local(question, ground_truth, contexts):
    ref_v = _embed([ground_truth])[0]
    ctx_vs = _embed(contexts)
    rel = sum(1 for cv in ctx_vs if _cos(ref_v, cv) >= 0.35)
    return rel / len(contexts)
```

**Problem**: A 512-token chunk's embedding is an average over all its tokens. If only 2 sentences in that chunk are about the ground_truth topic, the chunk-level embedding may still be far from the ground_truth vector. The 0.35 threshold is too strict for this structural mismatch.

### Proposed implementation
```python
def context_precision_local_sentence_level(question, ground_truth, contexts):
    """A context is 'relevant' if ANY of its sentences has cosine ≥ 0.35 vs ground_truth."""
    import nltk  # or use simple sentence splitting with regex
    ref_v = _embed([ground_truth])[0]
    relevant = 0
    for ctx in contexts:
        sentences = _sentences(ctx)  # existing helper
        if not sentences:
            continue
        sent_vs = _embed(sentences)
        if any(_cos(ref_v, sv) >= 0.35 for sv in sent_vs):
            relevant += 1
    return relevant / len(contexts) if contexts else 0.0
```

### Expected effect
- Enterprise: cp 0.47 → ~0.65-0.80 (most chunks have at least one sentence on-topic)
- Ora-et-labora: cp 0.65 → ~0.85-0.95
- The metric becomes more stable and responsive to real retrieval quality

### Implementation steps
1. Add sentence splitting to `_sentences()` in evaluate_pipeline.py (currently regex-based)
2. Add `context_precision_local_sentence_level()` (or modify `context_precision_local`)
3. Create `context_recall_local_sentence_level()` using same approach (sentence-level recall = fraction of ground_truth topic covered)
4. Update `evaluate_local()` to use the new functions
5. Re-baseline both golden sets
6. Update release gate thresholds (cp floor can move from 0.25 to 0.40)

### Verification
- Run both golden sets (enterprise + oraetlabora) before and after
- Confirm metrics are higher AND more stable across repeated runs
- Confirm no regression in faithfulness (should stay at 1.0)

--------------------------------------------------------------------
## 2b: Decomposition-Based Answer Relevancy

### Current implementation (evaluate_pipeline.py:136-141)
```python
def answer_relevancy_local(question, answer):
    qv, av = _embed([question, answer])
    return max(0.0, _cos(qv, av))
```

**Problem**: For "what does X describe?" template questions, the question embedding is a generic "document describe" vector, while the answer is specific document content. Cosine similarity is structurally ~0.50-0.55 regardless of answer quality.

### Proposed implementation
```python
def answer_relevancy_local_skip_based(question, answer):
    """Decompose question into sub-questions; check answer coverage per sub-Q."""
    # Step 1: Extract key noun from "what does X describe?" → "X"
    import re
    m = re.search(r'what does the (.+?) (document|file|note) describe', question.lower())
    doc_name = m.group(1) if m else question
    
    # Step 2: Check that the answer addresses the sub-questions
    # Sub-Q1: Does the answer mention the document name?
    # Sub-Q2: Does the answer describe the document's purpose/function?
    # Sub-Q3: Does the answer include concrete details?
    answer_lower = answer.lower()
    checks = 0
    total = 3
    
    # Sub-Q1: document name mentioned in answer
    if doc_name in answer_lower:
        checks += 1
    
    # Sub-Q2: purpose-related terms in answer
    purpose_terms = ['describe', 'document', 'pipeline', 'system', 'process', 
                     'provides', 'handles', 'manages', 'enables']
    if any(t in answer_lower for t in purpose_terms):
        checks += 1
    
    # Sub-Q3: concrete details (numbers, dates, ports, config values)
    has_detail = bool(re.search(r'\b\d+', answer))  # any number
    if has_detail:
        checks += 1
    
    return checks / total
```

### For general questions (non-template)
Use a more general approach:
1. Extract wh-terms (what, how, why, when, who)
2. Check answer against each wh-aspect
3. For "how" questions: check for mechanism/step-by-step language
4. For "why" questions: check for because/due-to/since language

### Expected effect
- Enterprise: ar 0.56 → ~0.70-0.80
- Ora-et-labora: ar 0.65 → ~0.75-0.85
- The metric reflects actual answer quality (detail, coverage, specificity)

### Implementation steps
1. Add `answer_relevancy_local_skip_based()` to evaluate_pipeline.py
2. Create a question-type detector (template vs open)
3. Route to appropriate strategy based on question type
4. Re-baseline both golden sets
5. Update release gate thresholds

--------------------------------------------------------------------
## 2c: Updated Release Gate Thresholds (After Sprint 2)

| Metric | Current Gate | After Sprint 2 | Rationale |
|--------|-------------|----------------|-----------|
| Faithfulness | ≥ 0.85 | ≥ 0.85 | Unchanged (already 1.0) |
| Context Precision | ≥ 0.25 (floored) | ≥ 0.50 | Sentence-level should raise this |
| Context Recall | ≥ 0.50 | ≥ 0.75 | Already at 0.88 with top_k=12 |
| Answer Relevancy | ≥ 0.30 | ≥ 0.55 | Decomposition should raise to 0.7+ |

--------------------------------------------------------------------
## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Sentence splitting adds latency to evaluation | Split is fast (regex); embedding per sentence is 1 batch call |
| Metric values change → release gate thresholds need recalibration | Run both golden sets, set thresholds at mean - 2σ |
| Template-specific logic doesn't generalize to other question types | Add a question-type detector; fall back to current cosine for unknown types |
| Over-engineering the metric defeats the purpose of a "proxy" | Keep changes minimal (2 new functions, <50 lines each) |
