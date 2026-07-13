"""Unit tests for normalize_graph — extractor JSON schema normalization.

Ensures swapping the extraction LLM (which may emit different key names) does
not silently produce an empty graph. Pure logic, no GPU / daemon.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingest as I


def test_passthrough_canonical():
    d = {"nodes": [{"id": "BUG-204", "type": "Bug"}],
         "edges": [{"source": "bob", "target": "BUG-204", "type": "REPORTED"}]}
    assert I.normalize_graph(d) == d


def test_mobile_variant_id_label_type():
    # google/gemma-4-E2B-it-qat-mobile emits {id,label,type}
    d = {"nodes": [{"id": "BUG-204", "label": "BUG-204", "type": "Bug"}],
         "edges": [{"source": "bob", "target": "alice", "type": "ASSOCIATED_WITH"}]}
    out = I.normalize_graph(d)
    assert out["nodes"][0]["id"] == "BUG-204"
    assert out["nodes"][0]["type"] == "Bug"
    assert out["edges"][0]["source"] == "bob"
    assert out["edges"][0]["target"] == "alice"
    assert out["edges"][0]["type"] == "ASSOCIATED_WITH"


def test_name_entity_variants():
    d = {"nodes": [{"name": "checkout", "entity_type": "Service"},
                   {"entity": "billing", "node_type": "Database"}],
         "edges": []}
    out = I.normalize_graph(d)
    ids = {n["id"] for n in out["nodes"]}
    assert ids == {"checkout", "billing"}
    types = {n["type"] for n in out["nodes"]}
    assert types == {"Service", "Database"}


def test_edge_relation_synonyms():
    d = {"nodes": [], "edges": [
        {"from": "a", "to": "b", "relation": "USES"},
        {"subject": "c", "object": "d", "relationship": "DEPENDS_ON"},
        {"head": "e", "tail": "f"},  # no rel -> defaults RELATED
    ]}
    out = I.normalize_graph(d)
    assert out["edges"][0] == {"source": "a", "target": "b", "type": "USES"}
    assert out["edges"][1]["source"] == "c"
    assert out["edges"][2]["type"] == "RELATED"


def test_drops_entities_without_name():
    d = {"nodes": [{"type": "Bug"}, {"id": "BUG-204"}], "edges": []}
    out = I.normalize_graph(d)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["id"] == "BUG-204"


def test_drops_edges_missing_endpoint():
    d = {"nodes": [], "edges": [{"source": "a"}, {"target": "b"}]}
    out = I.normalize_graph(d)
    assert out["edges"] == []


def test_non_dict_passthrough():
    assert I.normalize_graph("not a dict") == "not a dict"
    assert I.normalize_graph(None) is None
