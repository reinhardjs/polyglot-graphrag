"""Unit tests for REQ-7 domain metadata schema.

No GPU / daemon. Tests:
  - validate_metadata(): unknown field warns but is kept (no drop); declared
    schema with all-known fields yields no warnings; no schema = accept all.
  - write_vectors(): full meta dict (incl. domain metadata) stored under the
    Qdrant payload 'metadata' key (mocked client).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
import ingest as I


ENG_PROFILE = C.load_domain_profile("engineering")


def test_validate_metadata_unknown_field_warns_kept():
    meta = {"title": "My Doc", "bogus_field": "x"}
    accepted, warnings = I.validate_metadata(ENG_PROFILE, meta)
    assert any("bogus_field" in w for w in warnings)
    assert accepted.get("bogus_field") == "x"  # kept, not dropped
    assert accepted.get("title") == "My Doc"


def test_validate_metadata_known_fields_no_warning():
    # engineering schema declares only author + doc_type
    meta = {"author": "bob", "doc_type": "eng"}
    accepted, warnings = I.validate_metadata(ENG_PROFILE, meta)
    assert warnings == []
    assert accepted == meta


def test_validate_metadata_no_schema_accepts_all():
    # profile without metadata_schema -> accept everything
    prof = {"domain": {"name": "x"}}
    accepted, warnings = I.validate_metadata(prof, {"anything": 1})
    assert warnings == []
    assert accepted["anything"] == 1


def test_validate_metadata_none_profile_accepts_all():
    accepted, warnings = I.validate_metadata(None, {"a": 1})
    assert warnings == []
    assert accepted["a"] == 1


def test_write_vectors_stores_metadata():
    # Mock the qdrant client used by get_qdrant()
    captured = {}

    class _Pt:
        def __init__(self, id, vector, payload):
            captured.setdefault("pts", []).append((id, payload))

    class _QC:
        def upsert(self, collection, pts, wait=False):
            captured["collection"] = collection
            captured["pts"] = [(p.id, p.payload) for p in pts]

    orig = I.get_qdrant
    I.get_qdrant = lambda: _QC()
    try:
        chunks = [{"chunk_idx": 0, "text": "abc", "vector": [0.1] * 4}]
        meta = {"doc_type": "med", "domain": "medical", "patient_id": "P-9",
                "title": "Encounter"}
        n = I.write_vectors("d1", chunks, meta, checksum="cs",
                            collection="medical_chunks")
    finally:
        I.get_qdrant = orig
    assert n == 1
    pl = captured["pts"][0][1]
    assert pl["metadata"]["patient_id"] == "P-9"
    assert pl["metadata"]["title"] == "Encounter"
    assert pl["metadata"]["domain"] == "medical"
    assert pl["doc_id"] == "d1"
