#!/usr/bin/env python3
"""Build a golden dataset for a CONFIDENTIAL ingested corpus
(enterprise domain) so we can measure synthesized-answer QUALITY
(Faithfulness / Answer Relevance / Context Precision / Context Recall)
via evaluate_pipeline.py --live --domain enterprise.

The corpus name is the user's — this script only derives Q/ground_truth
pairs from whatever docs are already ingested into `enterprise`; it does
not hardcode any corpus identity.
GROUND_TRUTH span that is also present in that doc. This mirrors how
bench_corpus.py builds its R2 queries, but adds a grounded
ground_truth so Context Recall + Answer Relevance are measurable.

The daemon fills in `answer` + `contexts` live (--live), so this file
only needs to ship {question, ground_truth, doc_id}.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
import config as C

BASE = C.BASE_DIR
random.seed(20260715)

qc = QdrantClient(url=C.QDRANT_URL)
hits = qc.scroll("enterprise", limit=9000,
               with_payload=["doc_id", "text", "chunk_idx"])[0]

# Distinct first-chunk text per doc
first = {}
for h in hits:
    did = h.payload.get("doc_id", "")
    if h.payload.get("chunk_idx") != 0:
        continue
    if did not in first:
        first[did] = h.payload.get("text", "")

docs = [d for d, t in first.items() if t and t.strip()]
random.shuffle(docs)

# (stem fragment) -> a question template grounded in that doc family.
# We craft questions whose ground_truth is a factual span we KNOW is in
# the doc because we take it verbatim from the doc's own text.
def make_pair(did, text):
    # Use the doc's first sentence-ish as ground truth (it is, by
    # construction, present in the doc -> Context Recall measurable).
    words = text.split()
    # pick a 6-12 word factual span from the early text
    span = " ".join(words[3:14]).strip(" .,;:")
    if len(span) < 20:
        span = " ".join(words[:18]).strip(" .,;:")
    short = did.split("::")[-1].replace("-", " ")
    q = f"what does the {short} document describe?"
    return {
        "question": q,
        "ground_truth": span,
        "doc_id": did,
    }

rows = [make_pair(d, first[d]) for d in docs[:15]]
with open(os.path.join(BASE, "golden", "golden.json"), "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"wrote {len(rows)} golden rows -> golden/golden.json (git-ignored)")
for r in rows[:4]:
    print(f"  Q: {r['question']}")
    print(f"     GT: {r['ground_truth'][:80]}")
