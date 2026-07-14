# Rerank calibration — bge-reranker-v2-m3 score bands

## Model
Rerank stage uses a **cross-encoder**: `BAAI/bge-reranker-v2-m3`
(`config.RERANK_MODEL_NAME`), loaded as `CrossEncoder(...)` in `serve_gpu.py`.

It is NOT a bi-encoder. Retrieval (Qdrant dense) uses the bi-encoder
(`Jina-v3`); rerank uses the cross-encoder. Two-stage: bi-encoder retrieves
~20-50 candidates, cross-encoder re-scores the final top_k.

## Why calibrate?
The `confidence` label (low/medium/high) was initially derived from the
**post-boost** score, which meant a weak dual-signal candidate (raw ~0.05 +
0.15 boost = 0.20) could read as "medium" purely for being dual. That conflates
*relevance* with *corroboration*. Fixed: confidence is computed on the
**de-boosted raw cross-encoder score**; `dual_signal` stays a separate flag.

## Observed distribution (12 clinical queries, GPU fp16)
Sample queries: flat symptom sets, temporal presentations, symptom-only, and
two noise controls ("purple elephant trumpet symphony", "quantum entanglement
migraine").

| Tier | Raw score range | Example |
|------|-----------------|---------|
| noise / irrelevant | 0.000 – 0.01 | noise controls, endo (weak) |
| weak single match | 0.02 – 0.15 | endo 0.02-0.09 |
| moderate | 0.15 – 0.50 | flu 0.26-0.54, rheum, systemic 0.16-0.41 |
| strong (dual) | >= 0.50 | cardio 0.64-1.11, GI 0.61-1.09, AIDS 0.48-0.90 |

## Bands (applied to raw score, `serve_gpu._confidence`)
```
  raw >= 0.50  -> "high"
  raw >= 0.15  -> "medium"
  else         -> "low"
```

Rationale: 0.15 cleanly separates noise/weak (max 0.09-0.15) from real moderate
matches; 0.50 separates moderate from the strong dual-signal tier. Noise queries
(~0.00) correctly land as "low" — and would surface as `path: qdrant`
(no SNOMED graph match), so a clinician sees "no terminology match" cleanly.

## Caveats
- fp16 on RTX 3060. Distribution would shift under CPU fp32; if reranker ever
  runs on CPU, re-calibrate.
- Scores are query-relative, not absolute probabilities. "high" = strong
  keyword+semantic overlap evidence, NOT clinical probability of the disease.
- dual_signal is independent corroboration: a "low"/"medium" candidate that is
  dual-signal is still weakly relevant per the model but backed by two sources.
