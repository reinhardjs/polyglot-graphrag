"""domains/clinical_prose/ingest.py — populate the clinical-prose corpus.

SNOMED alone cannot do differential diagnosis: it stores concept names +
IS-A hierarchy, but NOT free-text disease descriptions. This ingestor pulls
a curated set of medical articles (Wikipedia Medicine extracts, which are
free CC-BY-SA) and embeds them into a dedicated Qdrant collection
(``clinical_prose``) so /ask can do SEMANTIC symptom->disease matching.

Why semantic prose instead of SNOMED graph edges?
  A Wikipedia "Measles" article says: "fever, cough, runny nose, conjunctivitis,
  and a characteristic red-brown maculopapular rash." Vector search on that
  chunk matches 3/3 of {fever, rash, cough}. The SNOMED concept "Measles
  (disorder)" contains NONE of those words in its name -> term-match misses it.

Design
------
* Embedding is delegated to the resident GPU daemon (serve_gpu.py) via the
  HTTP /embed_late endpoint, so we never load a second copy of Jina-v3.
* Upsert schema is identical to ingest.py (dense 1024-dim + sparse hash vector)
  so the existing rerank/synthesize pipeline can consume it unchanged.
* Idempotent: each article URL is hashed to a stable doc_id; re-running only
  adds new articles.

Usage
-----
    ./venv/bin/python -m domains.clinical_prose.ingest --limit 500
    # or via the daemon HTTP API once registered:
    curl -X POST http://localhost:8000/ingest_domain \
         -H 'Content-Type: application/json' \
         -d '{"domain":"clinical_prose","limit":500}'
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time

import requests

# Allow running as a script or module; import project root helpers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import config as C

# The resident GPU daemon embeds for us (no second model load).
DAEMON_EMBED_LATE = C.DAEMON_EMBED_LATE
COLLECTION = "clinical_prose"
CHUNK_SIZE = 512          # tokens (approx, by words/0.75)
CHUNK_OVERLAP = 64
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Curated starting set: high-yield disease + symptom articles. The ingestor
# also follows the "See also" / linked disease pages up to --limit, so this
# list is just the seed.
SEED_ARTICLES = [
    # Infectious
    "HIV/AIDS", "Measles", "Influenza", "Dengue fever", "Malaria", "Tuberculosis",
    "COVID-19", "Pneumonia", "Meningitis", "Hepatitis", "Chickenpox", "Rubella",
    "Mumps", "Polio", "Typhoid fever", "Cholera", "Lyme disease", "Syphilis",
    # Autoimmune / chronic
    "Systemic lupus erythematosus", "Rheumatoid arthritis", "Type 1 diabetes",
    "Multiple sclerosis", "Crohn's disease", "Ulcerative colitis", "Psoriasis",
    # Oncologic
    "Leukemia", "Lymphoma", "Lung cancer", "Breast cancer", "Colorectal cancer",
    # Cardiovascular
    "Myocardial infarction", "Angina pectoris", "Heart failure", "Atrial fibrillation",
    "Pulmonary embolism", "Hypertension", "Stroke",
    # Respiratory
    "Asthma", "Chronic obstructive pulmonary disease", "Bronchitis",
    # GI
    "Appendicitis", "Gastroenteritis", "Cirrhosis", "Pancreatitis", "Peptic ulcer disease",
    # Neuro
    "Migraine", "Epilepsy", "Parkinson's disease", "Alzheimer's disease",
    # Endocrine
    "Hypothyroidism", "Hyperthyroidism", "Addison's disease", "Cushing's syndrome",
    # Renal
    "Chronic kidney disease", "Nephrotic syndrome", "Urinary tract infection",
    # Symptoms (so symptom queries resolve too)
    "Fever", "Rash", "Chest pain", "Shortness of breath", "Fatigue", "Weight loss",
    "Cough", "Headache", "Abdominal pain", "Joint pain", "Night sweats",
    "Lymphadenopathy", "Oral candidiasis", "Hemoptysis", "Hematuria",
    # Opportunistic (HIV context)
    "Pneumocystis pneumonia", "Candidiasis", "Toxoplasmosis", "Cytomegalovirus",
]


def _http_json(params: dict) -> dict:
    headers = {
        "User-Agent": "polyglot-graphrag/1.0 (clinical corpus ingest; "
                      "contact: local-dev@example.com)"
    }
    r = requests.get(WIKI_API, params=params, timeout=30, headers=headers)
    r.raise_for_status()
    return r.json()


def fetch_extract(title: str):
    # type: ignore[return-value]
    """Fetch the plain-text extract of a Wikipedia article (intro + sections).

    Returns (text, links) on success, or None if no usable extract.
    """
    params = {
        "action": "query", "format": "json", "prop": "extracts|links",
        "explaintext": 1, "exsectionformat": "plain", "titles": title,
        "pllimit": "50", "redirects": 1,
    }
    data = _http_json(params)
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        text = page.get("extract")
        if text and len(text) > 200:
            links = [l["title"] for l in page.get("links", [])
                     if ":" not in l["title"] and "Talk:" not in l["title"]]
            return text, links
    return None


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Word-based chunking with overlap (lightweight; mirrors chunking.py)."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


def _embed_chunks(text: str) -> list:
    """Embed via the resident daemon /embed_late (returns list of vectors)."""
    chunks = _chunk_text(text)
    if not chunks:
        return []
    resp = requests.post(DAEMON_EMBED_LATE, json={
        "text": text, "doc_id": "tmp", "strategy": "fixed",
        "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json().get("chunks", [])


def ingest(limit: int = 500, seed: list | None = None) -> dict:
    """Seed from SEED_ARTICLES, follow links up to `limit` articles, embed + upsert.

    Returns a summary dict {ingested, skipped, failed, collection}.
    """
    from qdrant_client import QdrantClient, models

    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=C.VECTOR_DIM, distance=models.Distance.COSINE),
            sparse_vectors_config={"text": models.SparseVectorParams()},
        )
        print(f"[clinical_prose] created collection '{COLLECTION}'", flush=True)

    # Reuse ingest.py's sparse-vector helper for schema parity.
    from ingest import _sparse

    queue = list(seed or SEED_ARTICLES)
    seen = set()
    stats = {"ingested": 0, "skipped": 0, "failed": 0}
    total = 0

    while queue and total < limit:
        title = queue.pop(0)
        if title in seen:
            continue
        seen.add(title)
        total += 1
        try:
            res = fetch_extract(title)
            if not res:
                stats["skipped"] += 1
                continue
            text, links = res
            doc_id = "wiki:" + hashlib.md5(title.encode()).hexdigest()[:12]
            chunks = _embed_chunks(text)
            if not chunks:
                stats["skipped"] += 1
                continue
            pts = []
            for ch in chunks:
                if not ch.get("vector"):
                    continue
                payload = {
                    "doc_id": doc_id, "chunk_idx": ch["chunk_idx"],
                    "text": ch["text"], "doc_type": "clinical_prose",
                    "metadata": {"title": title, "source": "wikipedia",
                                 "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"},
                }
                pts.append(models.PointStruct(
                    id=abs(hash(f"{doc_id}:{ch['chunk_idx']}")) % (2 ** 63),
                    vector={"": ch["vector"], "text": _sparse(ch["text"])},
                    payload=payload,
                ))
            if pts:
                qc.upsert(COLLECTION, pts, wait=True)
                stats["ingested"] += 1
                print(f"[clinical_prose] + {title} ({len(pts)} chunks) "
                      f"[{total}/{limit}]", flush=True)
            # Enqueue linked disease-ish pages to grow the corpus.
            for l in links:
                if l not in seen and l not in queue:
                    queue.append(l)
        except Exception as e:
            stats["failed"] += 1
            print(f"[clinical_prose] ! {title}: {e}", flush=True)
        time.sleep(0.2)  # be polite to Wikipedia

    stats["collection"] = COLLECTION
    print(f"[clinical_prose] done: {stats}", flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", nargs="*", default=None,
                    help="Override seed article titles")
    args = ap.parse_args()
    ingest(limit=args.limit, seed=args.seed)
