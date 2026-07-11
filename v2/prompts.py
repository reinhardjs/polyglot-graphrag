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
    profile:  domain profile dict (load_domain_profile result) or None.

    Returns the full prompt string for the chat completion.
    """
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
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
