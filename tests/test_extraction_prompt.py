"""Unit tests for REQ-4 domain-aware extraction prompt selection.

No GPU / LLM required — we monkeypatch the OpenAI client and the GLiNER HTTP
call to assert the right prompt + labels reach the model.
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


def test_engineering_profile_uses_engineering_prompt():
    import domain_loader
    prof = domain_loader.get_domain("engineering")
    capture = {}
    # monkeypatch OpenAI used inside extract_graph_llm
    import openai
    orig = openai.OpenAI
    openai.OpenAI = lambda *a, **k: _FakeClient(capture)
    try:
        I.extract_graph_llm("doc1", "Bug-204 caused outage.", profile=prof)
    finally:
        openai.OpenAI = orig
    assert "Microservice" in capture["prompt"], capture["prompt"][:200]
    assert "engineering document" in capture["prompt"].lower()


def test_medical_profile_uses_medical_prompt():
    import domain_loader
    prof = domain_loader.get_domain("medical")
    capture = {}
    import openai
    orig = openai.OpenAI
    openai.OpenAI = lambda *a, **k: _FakeClient(capture)
    try:
        I.extract_graph_llm("doc2", "Lisinopril treats hypertension.", profile=prof)
    finally:
        openai.OpenAI = orig
    assert "Symptom" in capture["prompt"], capture["prompt"][:200]
    assert "Diagnosis" in capture["prompt"]
    assert "Medication" in capture["prompt"]


def test_unknown_domain_falls_back_to_config_prompt():
    # unknown domain -> engineering defaults -> config.EXTRACTION_PROMPT
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
    # engineering default prompt mentions Microservice
    assert "Microservice" in capture["prompt"]


def test_gliner_fallback_uses_domain_entity_types():
    """REQ-4 4d: GLiNER fallback receives profile graph_schema.entity_types."""
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

    requests.post = fake_post
    try:
        I.extract_graph_llm("doc4", "chest pain", profile=prof)
    finally:
        requests.post = orig_post
        openai.OpenAI = orig_openai
    assert sent.get("url") == C.DAEMON_EXTRACT
    assert sent["json"]["labels"] == prof["graph_schema"]["entity_types"]
    assert "Symptom" in sent["json"]["labels"]
