"""Unit tests for prompts.build_synthesis_prompt (v2.6.0 REQ-5).

No GPU / LLM. Verifies the shared prompt builder produces domain-specific
tone and the generic fallback.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
import prompts as P


def test_medical_prompt_mentions_clinical_documents():
    import domain_loader
    prof = domain_loader.get_domain("medical")
    p = P.build_synthesis_prompt("what treats hypertension?",
                                 ["Lisinopril treats hypertension."], profile=prof)
    assert "clinical documents" in p.lower(), p
    assert "[1]" in p  # numbered context injected


def test_legal_prompt_mentions_statutes():
    import domain_loader
    prof = domain_loader.get_domain("legal")
    p = P.build_synthesis_prompt("what does section 5 say?",
                                 ["Section 5 covers jurisdiction."], profile=prof)
    assert "statute" in p.lower() or "legal" in p.lower(), p


def test_engineering_unchanged_generic_fallback():
    # No profile -> generic "cite source numbers" prompt.
    p = P.build_synthesis_prompt("who reported BUG-204?", ["Bob reported BUG-204."])
    assert "cite source numbers" in p.lower(), p
    assert "[1]" in p


def test_profile_without_synthesis_falls_to_generic():
    # Engineering profile HAS synthesis, but test the no-synthesis branch by
    # passing a profile dict missing the synthesis section.
    prof = {"domain": {"name": "x"}, "chunking": {}}
    p = P.build_synthesis_prompt("q", ["c"], profile=prof)
    assert "cite source numbers" in p.lower()


def test_context_truncation_respects_budget():
    big = ["x" * 50000]
    p = P.build_synthesis_prompt("q", big)
    # Hard cap at MAX_TOKENS_CONTEXT * 4 chars.
    assert len(p) <= C.MAX_TOKENS_CONTEXT * 4 + 100
