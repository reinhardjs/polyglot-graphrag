"""Unit tests for prompts.build_synthesis_prompt.

Architecture note (v0.x): domain schemas live in domain_config.yaml (loaded by
domain_loader). The synthesis prompt is generic (config fallback) when the YAML
profile has no [synthesis] section — which is the current state. These tests
verify the generic prompt path works correctly with domain profiles.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
import prompts as P
import domain_loader


def test_medical_profile_uses_generic_synthesis_prompt():
    prof = domain_loader.get_domain("medical")
    p = P.build_synthesis_prompt("what treats hypertension?",
                                 ["Lisinopril treats hypertension."], profile=prof)
    # Generic prompt (no domain-specific [synthesis] section in YAML)
    assert "Answer the question using ONLY the context" in p
    assert "[1]" in p  # numbered context injected
    assert "Lisinopril treats hypertension." in p


def test_legal_profile_uses_generic_synthesis_prompt():
    prof = domain_loader.get_domain("legal")
    p = P.build_synthesis_prompt("what does section 5 say?",
                                 ["Section 5 covers jurisdiction."], profile=prof)
    assert "Answer the question using ONLY the context" in p
    assert "[1]" in p
    assert "Section 5 covers jurisdiction." in p


def test_none_profile_uses_generic_synthesis_prompt():
    p = P.build_synthesis_prompt("who reported BUG-204?",
                                 ["bob reported BUG-204."], profile=None)
    assert "Answer the question using ONLY the context" in p
    assert "[1]" in p
    assert "bob reported BUG-204." in p
