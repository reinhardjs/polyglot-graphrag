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
    if profile and profile.get("retrieval") == "snomed_traversal":
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
            "You are a clinical decision-support assistant using SNOMED CT. "
            "Below are the top diagnosis candidates retrieved from the SNOMED "
            "graph based on the patient's reported symptoms (each candidate "
            "lists the SNOMED concept and the supporting findings that matched). "
            "Produce a RANKED DIFFERENTIAL DIAGNOSIS: list the most likely "
            "disorders first, cite the candidate numbers, and explain WHY each "
            "is supported by the symptom timeline. Note if the presentation "
            "strongly suggests a systemic illness (symptoms spanning multiple "
            "days). Do NOT invent diagnoses beyond the candidates. If the "
            "candidates do not clearly explain the symptoms, say so.\n\n"
            f"PATIENT SYMPTOMS:\n{readable}\n\n"
            f"CANDIDATE DIAGNOSES (from SNOMED CT):\n{ctx}"
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
