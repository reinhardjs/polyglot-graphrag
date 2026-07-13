"""GLiNER-only graph extraction (no LLM / E2B dependency).

Zero-shot NER via GLiNER + co-occurrence heuristic edges
(ASSOCIATED_WITH between entities sharing a sentence). This is the
stable, E2B-free extraction path used when EXTRACT_WITHOUT_E2B=1
(e.g. bulk-loading large corpora where the Gemma-4-E2B extraction
LLM is unavailable or unstable on the host GPU).

Extracted from serve_gpu.py's /extract_graph endpoint so both the
HTTP endpoint and ingest_text() can share it.
"""

import re

# Populated lazily by the daemon (avoids importing gliner at module load).
_gliner = None
_gliner_labels = None
_gliner_threshold = 0.4


def _load_gliner():
    """Lazy-load GLiNER on first call. Imported here to avoid a hard dep at
    module import time and to reuse the daemon's already-loaded instance."""
    global _gliner
    if _gliner is None:
        from gliner import GLiNER
        import config as C
        import torch
        _gliner = GLiNER.from_pretrained(C.GLINER_MODEL_NAME).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        ).half()
        _gliner.predict_entities("warm", C.GLINER_LABELS)  # warmup
    return _gliner


def extract_gliner_graph(text: str, labels=None, threshold=None):
    """Return {"nodes":[...], "edges":[...]} using GLiNER + co-occurrence.

    No LLM is contacted, so this never depends on the E2B extraction server.
    """
    global _gliner_labels, _gliner_threshold
    gliner = _load_gliner()
    if labels is None:
        import config as C
        labels = C.GLINER_LABELS
    if threshold is None:
        import config as C
        threshold = C.GLINER_THRESHOLD
    preds = gliner.predict_entities(text, labels, threshold=threshold)

    nodes, edges, seen = {}, [], set()
    for sent in [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]:
        s_low = sent.lower()
        local = []
        for p in preds:
            if p["text"].lower() in s_low:
                nodes[p["text"].strip().lower()] = p["label"]
                local.append(p["text"].strip().lower())
        for a in range(len(local)):
            for b in range(a + 1, len(local)):
                pair = tuple(sorted((local[a], local[b])))
                if pair not in seen:
                    seen.add(pair)
                    edges.append({
                        "source": pair[0], "target": pair[1],
                        "type": "ASSOCIATED_WITH",
                    })
    return {
        "nodes": [{"id": k, "type": v} for k, v in nodes.items()],
        "edges": edges,
    }
