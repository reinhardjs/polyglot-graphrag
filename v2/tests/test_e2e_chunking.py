"""End-to-end tests for v2.6.0 domain profiling + chunking (REQ-1/2/3).

These hit the live /ask + /ingest endpoints on :8000 (assumes rag-gpu-daemon
is running). They are SLOW (involve LLM extraction + embed) — run separately:

    cd /mnt/data-970-plus/rag-system/v2
    /mnt/data-970-plus/rag-env/bin/python -m pytest tests/test_e2e_chunking.py -q

Requires: rag-gpu-daemon ( :8000 ) + gemma-4-e2b ( :8082 ) + gemma-4-e4b ( :8084 ).
"""
import sys, os, time, json, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://127.0.0.1:8000"


def _ingest_wait(doc_id, domain=None, text=None, collection=None, timeout=40):
    body = {"text": text or f"# {doc_id}\nFirst sentence here. Second sentence there.",
            "doc_id": doc_id}
    if domain:
        body["domain"] = domain
    if collection:
        body["collection"] = collection
    r = requests.post(f"{BASE}/ingest", json=body)
    assert r.status_code in (200, 202), r.text
    task_id = r.json()["task_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = requests.get(f"{BASE}/ingest/status/{task_id}").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(1)
    raise TimeoutError(f"ingest {task_id} did not finish")


def _delete(doc_id, collection):
    requests.delete(f"{BASE}/ingest/{doc_id}?collection={collection}")


def test_medical_domain_routes_to_medical_chunks():
    """REQ-2/3: domain=medical → medical_chunks collection + section chunking."""
    text = ("# Encounter\nChief complaint: chest pain.\n\n"
            "## Assessment\nHypertension noted.\n\n"
            "## Plan\nStart lisinopril.")
    try:
        st = _ingest_wait("e2e-med-sec", domain="medical", text=text)
        assert st["status"] == "done"
        # Verify stored in medical_chunks (not engineering_chunks)
        import ingest as I
        qc = I.get_qdrant()
        pts, _ = qc.scroll("medical_chunks", limit=20, with_payload=True)
        med = [p for p in pts if p.payload.get("doc_id") == "e2e-med-sec"]
        assert len(med) >= 2, "expected section chunks in medical_chunks"
        # Section headers preserved
        assert any(p.payload["text"].startswith("##") for p in med)
    finally:
        _delete("e2e-med-sec", "medical_chunks")


def test_engineering_default_uses_sentence_chunking():
    """Backward compat: no domain → engineering_chunks, sentence chunks."""
    try:
        st = _ingest_wait("e2e-eng-sent",
                          text="Alpha sentence one. Beta sentence two. Gamma three.")
        assert st["status"] == "done"
        import ingest as I
        qc = I.get_qdrant()
        pts, _ = qc.scroll("engineering_chunks", limit=50, with_payload=True)
        eng = [p for p in pts if p.payload.get("doc_id") == "e2e-eng-sent"]
        assert len(eng) >= 2, "sentence strategy should yield >=2 chunks"
    finally:
        _delete("e2e-eng-sent", "engineering_chunks")


def test_domain_ask_routes_collection():
    """REQ-2: /ask with domain=medical (no collection) queries medical_chunks."""
    # ingest a tiny medical doc first
    _ingest_wait("e2e-med-ask", domain="medical",
                 text="## Drug\nLisinopril treats hypertension.")
    try:
        r = requests.post(f"{BASE}/ask", json={
            "query": "what treats hypertension?", "domain": "medical",
            "synthesize": False})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["sources"], "expected sources bibliography"
        assert any(s["doc_id"] == "e2e-med-ask" for s in d["sources"])
    finally:
        _delete("e2e-med-ask", "medical_chunks")
