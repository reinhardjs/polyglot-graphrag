"""Unit/integration tests for the /ingest HTTP contract (v2.5.0 REQ-1 fixes).

Verifies:
  - POST /ingest returns 202 Accepted (non-blocking)
  - Re-ingest with matching `if_checksum` returns 304 unchanged (synchronous)
  - Empty text returns 400

These hit the live :8000 daemon. Run with the full suite or standalone:
    cd <project-root>
    ./venv/bin/python -m pytest tests/test_ingest_contract.py -q
"""
import sys, os, time, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://127.0.0.1:8000"


def _wait(doc_id, timeout=40):
    # poll status by listing — simpler than tracking task_id for these checks
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/ingest?collection=engineering_chunks")
        docs = r.json().get("documents", [])
        if any(d["doc_id"] == doc_id for d in docs):
            return True
        time.sleep(0.5)
    return False


def _cleanup(doc_id):
    requests.delete(f"{BASE}/ingest/{doc_id}?collection=engineering_chunks")


def test_post_ingest_returns_202():
    """REQ-1: POST /ingest is non-blocking and returns 202 Accepted."""
    doc_id = "contract-202"
    try:
        r = requests.post(f"{BASE}/ingest", json={
            "text": "Contract test doc for 202 status.", "doc_id": doc_id,
            "domain": "engineering"})
        assert r.status_code == 202, f"expected 202, got {r.status_code}: {r.text}"
        assert "task_id" in r.json()
    finally:
        _cleanup(doc_id)


def test_checksum_match_returns_304():
    """REQ-1: re-ingest with matching if_checksum → 304 unchanged (sync)."""
    doc_id = "contract-304"
    text = "Stable content for checksum test."
    try:
        # first ingest (202)
        r1 = requests.post(f"{BASE}/ingest", json={
            "text": text, "doc_id": doc_id, "domain": "engineering"})
        assert r1.status_code == 202
        assert _wait(doc_id), "doc never appeared"

        # fetch stored checksum
        r2 = requests.get(f"{BASE}/ingest?collection=engineering_chunks")
        cs = next((d["checksum"] for d in r2.json()["documents"]
                   if d["doc_id"] == doc_id), None)
        assert cs, "no checksum stored"

        # re-ingest with same checksum → 304
        r3 = requests.post(f"{BASE}/ingest", json={
            "text": text, "doc_id": doc_id, "domain": "engineering",
            "if_checksum": cs})
        assert r3.status_code == 304, f"expected 304, got {r3.status_code}: {r3.text}"
        # 304 is intentionally body-less (RFC 9110) — status alone confirms skip
    finally:
        _cleanup(doc_id)


def test_empty_text_returns_400():
    """REQ-1: empty text → 400 Bad Request."""
    r = requests.post(f"{BASE}/ingest", json={"text": "", "doc_id": "x"})
    assert r.status_code == 400
