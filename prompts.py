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
    # Domain-aware synthesis. Driven by profile['synthesis']['kind'] so the
    # Mediator stays domain-independent (no `if retrieval == "snomed_term_match"`).
    syn_kind = (profile or {}).get("synthesis", {}).get("kind")
    if syn_kind == "clinical_dx":
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
            "terminology PLUS a clinical-prose corpus (Wikipedia medicine "
            "articles). The CANDIDATE DIAGNOSES below may include two kinds of "
            "evidence:\n"
            "  - [DIAGNOSIS CANDIDATE] ... : a SNOMED concept whose NAME matched "
            "the patient's symptom keywords (IDF-weighted; rarer keywords weigh "
            "more). These show which disorders share naming with the symptoms.\n"
            "  - [CLINICAL PROSE] <title> ... : a free-text disease description "
            "whose full symptom PICTURE is semantically similar to the query. "
            "These capture diseases whose SNOMED name lacks the symptom words.\n\n"
            "Produce a RANKED DIFFERENTIAL DIAGNOSIS. Prefer candidates where BOTH "
            "signals agree (a SNOMED name match AND a prose description match). "
            "For each, cite the candidate number and explain WHY it is supported. "
            "Note when a presentation spans multiple days and several systemic / "
            "immunodeficiency-associated entries appear — hypothesize a systemic "
            "process (e.g. HIV/AIDS) but frame it as a hypothesis, not a diagnosis. "
            "Be honest: this is keyword + semantic overlap, not clinical certainty. "
            "Do NOT invent findings beyond the candidates.\\n\\n"
            f"PATIENT SYMPTOMS:\\n{readable}\\n\\n"
            f"CANDIDATE DIAGNOSES (SNOMED CT term-match + clinical prose):\\n{ctx}"
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
