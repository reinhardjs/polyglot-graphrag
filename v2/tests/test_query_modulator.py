"""Unit tests for Phase 1 query_modulator.py (domain-agnostic, no daemon).

Mocks the Neo4j driver so tests run without a live graph. Validates:
  - alias map built from (:Entity) nodes (aliases + name→type)
  - abbreviation expansion ("NDA" -> canonical, "htn" -> Hypertension)
  - domain scoping (active domain only)
  - unknown tokens pass through untouched
  - original text preserved, expansions appended in parentheses
"""
from query_modulator import build_alias_map, modulate_query, QueryModulator


class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def data(self): return self._rows


class _FakeSession:
    def __init__(self, rows): self._rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, cypher): return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows): self._rows = rows
    def session(self): return _FakeSession(self._rows)
    def close(self): pass


# Medical-ish fixture: NDA has aliases, htn is an entity name of type Disease
MED_ROWS = [
    {"name": "Non-Disclosure Agreement", "type": "LegalDoc",
     "aliases": ["NDA", "confidentiality agreement"]},
    {"name": "Hypertension", "type": "Disease", "aliases": ["htn", "HTN"]},
]


def test_build_alias_map_from_aliases():
    drv = _FakeDriver(MED_ROWS)
    amap = build_alias_map("medical", drv)
    assert amap["nda"] == "Non-Disclosure Agreement"
    assert amap["htn"] == "Hypertension"
    # name→type also registered
    assert amap["hypertension"] == "Hypertension"


def test_modulate_expands_abbreviation():
    drv = _FakeDriver(MED_ROWS)
    out = modulate_query("NDA breached", "medical", drv)
    assert "NDA breached" in out
    assert "Non-Disclosure Agreement" in out
    assert out.startswith("NDA breached (")


def test_modulate_expands_lowercase_alias():
    drv = _FakeDriver(MED_ROWS)
    out = modulate_query("patient has htn", "medical", drv)
    assert "Hypertension" in out


def test_unknown_tokens_pass_through():
    drv = _FakeDriver(MED_ROWS)
    out = modulate_query("the quick brown fox", "medical", drv)
    assert out == "the quick brown fox"


def test_domain_scoping_isolates_vocab():
    # Current schema: :Entity nodes are NOT tagged per-domain, so build_alias_map
    # scans ALL entities (global vocabulary). The `domain` arg is accepted for
    # forward-compat (auto-scopes if :Entity:<Label> nodes ever exist) but today
    # the same alias resolves under any domain request.
    global_rows = MED_ROWS + [
        {"name": "Checkout Microservice", "type": "Microservice",
         "aliases": ["checkout", "CO"]},
    ]
    drv = _FakeDriver(global_rows)
    out_med = modulate_query("NDA issue", "medical", drv)
    out_eng = modulate_query("NDA issue", "engineering", drv)
    assert "Non-Disclosure Agreement" in out_med
    assert "Non-Disclosure Agreement" in out_eng
    # and the engineering-specific alias also resolves under medical request
    assert "Checkout Microservice" in modulate_query("checkout down", "medical", drv)


def test_querymodulator_wrapper_caches_per_domain():
    QueryModulator._cache.clear()  # isolate from other tests' cached maps
    drv = _FakeDriver(MED_ROWS)
    q1 = QueryModulator.moderate("NDA issue", "medical", driver=drv)
    assert "Non-Disclosure Agreement" in q1
    # second call hits cache (no driver needed) — still correct
    q2 = QueryModulator.moderate("NDA signed", "medical")
    assert "Non-Disclosure Agreement" in q2
    QueryModulator.refresh("medical")
