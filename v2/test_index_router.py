"""test_index_router.py — V3.0 Hybrid Index-Routing unit + integration tests.

Pure-logic tests (no external services required):
  - build_route_request pairing strategies
  - _salvage_objects (truncated-JSON recovery)
  - _parse_relations (fence stripping, vocabulary enforcement, validation)
  - domain_loader YAML schema integrity

Integration tests (require live GLiNER daemon + Qwen) — gated by
IR_LIVE=1 env var so CI can run the fast suite without GPUs.

Run:  python -m pytest test_index_router.py -q
Live: IR_LIVE=1 python -m pytest test_index_router.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from index_router import (
    build_route_request,
    extract_entities,
    _format_pairs,
    _salvage_objects,
    _parse_relations,
    query_routes,
    extract,
)

DAEMON_URL = os.environ.get("IR_DAEMON_URL", "http://localhost:8000")
QWEN_URL = os.environ.get("IR_QWEN_URL", "http://localhost:8082/v1/chat/completions")
LIVE = os.environ.get("IR_LIVE", "0") == "1"

ENG_TEXT = (
    "PR-482 authored by bob fixes BUG-204. "
    "The checkout-publisher microservice depends on the checkout database. "
    "ADR-014 references the auth-service. alice reviewed PR-482."
)


# ── Pure logic: build_route_request ──────────────────────────────────────────
def test_build_route_request_all_pairs():
    entities = [{"name": "A", "type": "X"}, {"name": "B", "type": "Y"},
                {"name": "C", "type": "Z"}]
    payload = build_route_request(
        "text here", entities, ["REL1", "REL2"], "all_pairs")
    assert len(payload["pairs"]) == 3  # 3 choose 2
    assert set(payload["relation_types"]) == {"REL1", "REL2"}
    assert payload["text"] == "text here"


def test_build_route_request_same_sentence():
    entities = [{"name": "A", "type": "X"}, {"name": "B", "type": "Y"},
                {"name": "C", "type": "Z"}]
    # "A and B" share one sentence; C is in another.
    text = "A relates to B. C is separate."
    payload = build_route_request(
        text, entities, ["REL1", "REL2"], "same_sentence")
    # Only A-B co-occur in the first sentence → 1 pair (not all 3 choose 2)
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0] == {"source": 0, "target": 1}


def test_build_route_request_empty():
    payload = build_route_request("t", [], ["R"], "all_pairs")
    assert payload["pairs"] == []


def test_format_pairs_indices():
    entities = [{"name": "A", "type": "X"}, {"name": "B", "type": "Y"}]
    payload = build_route_request("t", entities, ["R"], "all_pairs")
    out = _format_pairs(payload)
    assert "(X) 'A'" in out
    assert "(Y) 'B'" in out


def test_same_sentence_fewer_than_all_pairs():
    entities = [{"name": "A", "type": "X"}, {"name": "B", "type": "Y"},
                {"name": "C", "type": "Z"}]
    text = "A relates to B. C is separate."
    allp = build_route_request(text, entities, ["R"], "all_pairs")
    sames = build_route_request(text, entities, ["R"], "same_sentence")
    # same_sentence must produce strictly fewer pairs than all_pairs
    assert len(sames["pairs"]) < len(allp["pairs"])
    # and at least the intra-sentence A-B pair must be present
    assert {"source": 0, "target": 1} in sames["pairs"]


# ── _salvage_objects (truncated JSON recovery) ───────────────────────────────
def test_salvage_complete_objects():
    text = '{"relations":[{"source":0,"target":1,"type":"FIXES","confidence":0.9}]}'
    objs = _salvage_objects(text)
    assert len(objs) == 1
    assert objs[0]["type"] == "FIXES"


def test_salvage_truncated():
    # Simulate max_tokens cutoff mid-array
    text = ('[{"source":0,"target":1,"type":"FIXES","confidence":0.9},'
            '{"source":2,"target":3,"type":"AUTHORED","confidence":0.8},'
            '{"source":4,"target":5,"type":"DEPENDS_ON"')
    objs = _salvage_objects(text)
    types = {o["type"] for o in objs}
    assert "FIXES" in types
    assert "AUTHORED" in types
    assert "DEPENDS_ON" not in types  # incomplete, discarded


# ── _parse_relations ─────────────────────────────────────────────────────────
def test_parse_strips_fences():
    content = '```json\n{"relations":[{"source":0,"target":1,"type":"FIXES","confidence":0.9}]}\n```'
    payload = {"text": "t", "entities": [{"name": "a"}, {"name": "b"}],
               "pairs": [], "relation_types": ["FIXES"]}
    rels = _parse_relations(content, payload, min_confidence=0.5)
    assert len(rels) == 1
    assert rels[0]["type"] == "FIXES"


def test_parse_vocabulary_enforcement():
    """Unknown relation type must be rejected."""
    content = '{"relations":[{"source":0,"target":1,"type":"NOT_A_TYPE","confidence":0.9}]}'
    payload = {"text": "t", "entities": [{"name": "a"}, {"name": "b"}],
               "pairs": [], "relation_types": ["FIXES"]}
    rels = _parse_relations(content, payload, min_confidence=0.5)
    assert rels == []


def test_parse_confidence_gate():
    content = ('[{"source":0,"target":1,"type":"FIXES","confidence":0.3},'
               '{"source":0,"target":1,"type":"FIXES","confidence":0.95}]')
    payload = {"text": "t", "entities": [{"name": "a"}, {"name": "b"}],
               "pairs": [], "relation_types": ["FIXES"]}
    rels = _parse_relations(content, payload, min_confidence=0.6)
    assert len(rels) == 1
    assert rels[0]["confidence"] == 0.95


def test_parse_index_bounds():
    """Out-of-range entity indices must be rejected."""
    content = '{"relations":[{"source":0,"target":99,"type":"FIXES","confidence":0.9}]}'
    payload = {"text": "t", "entities": [{"name": "a"}, {"name": "b"}],
               "pairs": [], "relation_types": ["FIXES"]}
    rels = _parse_relations(content, payload, min_confidence=0.5)
    assert rels == []


def test_parse_self_loop_rejected():
    content = '{"relations":[{"source":0,"target":0,"type":"FIXES","confidence":0.9}]}'
    payload = {"text": "t", "entities": [{"name": "a"}, {"name": "b"}],
               "pairs": [], "relation_types": ["FIXES"]}
    rels = _parse_relations(content, payload, min_confidence=0.5)
    assert rels == []


def test_parse_dedup():
    content = ('[{"source":0,"target":1,"type":"FIXES","confidence":0.9},'
               '{"source":0,"target":1,"type":"FIXES","confidence":0.9}]')
    payload = {"text": "t", "entities": [{"name": "a"}, {"name": "b"}],
               "pairs": [], "relation_types": ["FIXES"]}
    rels = _parse_relations(content, payload, min_confidence=0.5)
    assert len(rels) == 1


# ── domain_loader ─────────────────────────────────────────────────────────────
def test_domain_loader_all_present():
    import domain_loader
    domains = domain_loader.list_domains()
    assert set(domains) == {"engineering", "legal", "medical",
                            "accounting", "hospitality"}


def test_domain_loader_schema_integrity():
    import domain_loader
    for name in domain_loader.list_domains():
        d = domain_loader.get_domain(name)
        assert d.get("entity_types"), f"{name} missing entity_types"
        assert d.get("relation_types"), f"{name} missing relation_types"
        assert d.get("collection"), f"{name} missing collection"
        assert d.get("pair_strategy") in ("all_pairs", "same_sentence", "window")
        assert 0.0 < d.get("min_confidence", 0) <= 1.0


# ── Integration (live services) ───────────────────────────────────────────────
@pytest.mark.skipif(not LIVE, reason="set IR_LIVE=1 for live GLiNER+Qwen tests")
def test_extract_entities_live():
    ents = extract_entities(ENG_TEXT,
                            ["PR", "Bug", "Developer", "Microservice", "Database", "ADR"],
                            daemon_url=DAEMON_URL)
    names = {e["name"] for e in ents}
    assert "PR-482" in names
    assert "BUG-204" in names
    # spans must be correct
    for e in ents:
        assert ENG_TEXT[e["start"]:e["end"]] == e["name"]


@pytest.mark.skipif(not LIVE, reason="set IR_LIVE=1 for live GLiNER+Qwen tests")
def test_extract_live_engineering():
    result = extract(ENG_TEXT, "engineering",
                     daemon_url=DAEMON_URL, qwen_url=QWEN_URL)
    assert len(result["entities"]) >= 3
    # At least one typed relation, all within vocabulary
    assert len(result["relations"]) >= 1
    dom = __import__("domain_loader").get_domain("engineering")
    for r in result["relations"]:
        assert r["type"] in dom["relation_types"]


@pytest.mark.skipif(not LIVE, reason="set IR_LIVE=1 for live GLiNER+Qwen tests")
def test_query_routes_batched():
    entities = extract_entities(ENG_TEXT,
                                ["PR", "Bug", "Developer", "Microservice", "Database", "ADR"],
                                daemon_url=DAEMON_URL)
    payload = build_route_request(ENG_TEXT, entities,
                                  ["FIXES", "AUTHORED", "DEPENDS_ON", "REFERENCES"],
                                  "all_pairs")
    rels = query_routes(payload, qwen_url=QWEN_URL, batch_size=6)
    # every relation valid: indices in range, type in vocab
    n = len(payload["entities"])
    for r in rels:
        assert 0 <= r["source"] < n
        assert 0 <= r["target"] < n
        assert r["type"] in payload["relation_types"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
