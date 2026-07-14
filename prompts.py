"""Shared synthesis-prompt builder (v2.6.0 REQ-5).

Single source of truth for the E4B answer-generation prompt. Used by
ask.py:synthesize() (streaming /ask) and retrieve_json.py:_synthesize_nonstream()
(Hermes plugin). Domain-aware: when a profile supplies [synthesis].prompt, the
profile's role + {context}/{query} template is used; otherwise a generic
"cite source numbers" prompt is emitted.

Token budget: contexts are truncated to config.MAX_TOKENS_CONTEXT (×4 chars).
"""
import config as C


def build_synthesis_prompt(query: str, contexts: list,
                            profile: dict = None) -> str:
    """Build the E4B synthesis prompt.

    contexts: list of retrieved text blocks (already in order).
    profile:  domain profile dict (domain_loader.get_domain result) or None.

    Returns the full prompt string for the chat completion.
    """
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    # SNOMED clinical diagnosis synthesis: frame the candidates as a ranked
    # differential. The query may be a raw JSON temporal presentation, so we
    # render it as a readable symptom timeline for the model.
    if profile and profile.get("retrieval") == "snomed_term_match":
        readable = query
        try:
            import json as _j
            pres = _j.loads(query)
            if isinstance(pres, dict):
                lines = [f"Day {k[3:] if k.startswith('day') else k}: {', '.join(v)}"
                         for k, v in sorted(pres.items(),
                                            key=lambda kv: int(''.join(filter(str.isdigit, kv[0])) or 0))]
                readable = "Patient presents with symptoms over time:\n" + "\n".join(lines)
        except Exception:
            pass
        return (
            "You are a clinical decision-support assistant using SNOMED CT "
            "terminology. Below are SNOMED concepts whose names matched the "
            "patient's reported symptom keywords (NOT a differential diagnosis). "
            "Each candidate shows the IDF (Inverse Document Frequency) weight for "
            "each matched keyword — higher IDF means the keyword is rarer and more "
            "discriminative. The total score is the sum of IDF weights across matched "
            "keywords, plus edge-signal boosts where applicable.\n\n"
            "IMPORTANT: These are FINDINGS THAT MATCH SYMPTOM KEYWORDS IN DISORDER "
            "NAMES, not definitive diagnoses. Be honest about this: the most likely "
            "explanation is based on keyword overlap, not clinical expertise. "
            "For each candidate, note which specific terms matched and with what "
            "confidence (IDF weight). If the presentation spans multiple days and "
            "several HIV-associated or systemic disorders appear, suggest that a "
            "systemic or immunodeficiency-related process may be the underlying "
            "cause — but frame this as a hypothesis, not a diagnosis.\n\n"
            "Do NOT invent findings beyond the candidates. If candidates do not "
            "clearly explain the symptoms, say so.\n\n"
            f"PATIENT SYMPTOMS:\n{readable}\n\n"
            f"CANDIDATE DIAGNOSES (from SNOMED CT term matching):\n{ctx}"
        )[: C.MAX_TOKENS_CONTEXT * 4]
    if profile and profile.get("synthesis"):
        syn = profile["synthesis"]
        role = syn.get("role", "knowledge assistant")
        tmpl = syn.get("prompt", "")
        if tmpl and "{context}" in tmpl and "{query}" in tmpl:
            # Safe substitution — profiles may contain literal {...} (JSON
            # examples) that break str.format(); replace only our two tokens.
            return (tmpl
                    .replace("{context}", ctx)
                    .replace("{query}", query))[: C.MAX_TOKENS_CONTEXT * 4]
        # Profile has a role but no full template → prefix with role.
        return (
            f"You are a {role}. Answer using ONLY the context. "
            f"Cite sources. If missing, say so.\n\n"
            f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"
        )[: C.MAX_TOKENS_CONTEXT * 4]
    # Generic fallback (no profile).
    return (
        "Answer the question using ONLY the context. "
        "Cite source numbers. If missing, say so.\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"
    )[: C.MAX_TOKENS_CONTEXT * 4]
