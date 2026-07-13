"""Unit tests for REQ-4 domain-aware extraction prompt selection.

No GPU / LLM required — we monkeypatch the OpenAI client and the GLiNER HTTP
call to assert the right prompt + labels reach the model.

Architecture note (v0.x): domain schemas live in domain_config.yaml (loaded by
domain_loader). The extraction prompt is config.EXTRACTION_PROMPT (generic) when
the YAML profile has no [extraction].prompt section — which is the current
state. Domain entity_types still flow into GLiNER fallback labeling.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
import ingest as I


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, capture):
        self._capture = capture

    def create(self, **kw):
        self._capture["model"] = kw.get("model")
        self._capture["prompt"] = kw["messages"][0]["content"]
        return _FakeResp('{"nodes":[],"edges":[]}')


class _FakeClient:
    def __init__(self, capture):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(capture)})()


def test_engineering_profile_uses_generic_extraction_prompt():
    """YAML engineering profile has no [extraction].prompt → generic fallback."""
    import domain_loader
    prof = domain_loader.get_domain("engineering")
    capture = {}
    import openai
    orig = openai.OpenAI
    openai.OpenAI = lambda *a, **k: _FakeClient(capture)
    try:
        I.extract_graph_llm("doc1", "Bug-204 caused outage.", profile=prof)
    finally:
        openai.OpenAI = orig
    # Generic prompt is used (no domain-specific [extraction].prompt in YAML)
    assert capture["prompt"] == C.EXTRACTION_PROMPT.replace("{doc_id}", "doc1").replace(
        "{text}", "Bug-204 caused outage."[:C.EXTRACTION_CHAR_LIMIT])


def test_medical_profile_uses_generic_extraction_prompt():
    """YAML medical profile has no [extraction].prompt → generic fallback."""
    import domain_loader
    prof = domain_loader.get_domain("medical")
    capture = {}
    import openai
    orig = openai.OpenAI
    openai.OpenAI = lambda *a, **k: _FakeClient(capture)
    try:
        I.extract_graph_llm("doc2", "Lisinopril treats hypertension.", profile=prof)
    finally:
        # Generic prompt is used (no domain-specific [extraction].prompt in YAML)
        assert capture["prompt"] == C.EXTRACTION_PROMPT.replace("{doc_id}", "doc2").replace(
            "{text}", "Lisinopril treats hypertension."[:C.EXTRACTION_CHAR_LIMIT])


def test_unknown_domain_falls_back_to_config_prompt():
    # unknown domain -> domain_loader returns None -> config.EXTRACTION_PROMPT
    import domain_loader
    prof = domain_loader.get_domain("nonexistent")
    capture = {}
    import openai
    orig = openai.OpenAI
    openai.OpenAI = lambda *a, **k: _FakeClient(capture)
    try:
        I.extract_graph_llm("doc3", "text", profile=prof)
    finally:
        openai.OpenAI = orig
    # Generic prompt is used (no domain-specific [extraction].prompt in YAML)
    assert capture["prompt"] == C.EXTRACTION_PROMPT.replace("{doc_id}", "doc3").replace(
        "{text}", "text"[:C.EXTRACTION_CHAR_LIMIT])


def test_gliner_fallback_uses_domain_entity_types():
    """REQ-4 4d: GLiNER fallback receives profile entity_types from YAML."""
    import requests
    import domain_loader
    prof = domain_loader.get_domain("medical")
    sent = {}
    orig_post = requests.post

    class _FakeCompletionsFail:
        def create(self, **kw):
            raise RuntimeError("forced LLM failure for test")

    class _FakeClientFail:
        def __init__(self, *a, **k):
            self.chat = type("Chat", (), {"completions": _FakeCompletionsFail()})()

    import openai
    orig_openai = openai.OpenAI
    openai.OpenAI = lambda *a, **k: _FakeClientFail()

    def fake_post(url, json=None, timeout=None, **kw):
        sent["url"] = url
        sent["json"] = json
        return type("R", (), {"json": lambda: {"nodes": [], "edges": []}})()

    # GLiNER fallback uses entity_types from the YAML profile (not graph_schema)
    requests.post = fake_post
    try:
        I.extract_graph_llm("doc4", "chest pain", profile=prof)
    finally:
        requests.post = orig_post
        openai.OpenAI = orig_openai
    assert sent.get("url") == C.DAEMON_EXTRACT
    assert sent["json"]["labels"] == prof["entity_types"]
    assert "Symptom" in sent["json"]["labels"]
