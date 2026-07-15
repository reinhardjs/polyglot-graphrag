"""Shared synthesis-prompt builder (v2.6.0 REQ-5, federated 2026-07).

Single source of truth for the answer-generation (synthesis) prompt. Used by
ask.py:synthesize() (streaming /ask) and retrieve_json.py:_synthesize_nonstream()
(Hermes plugin).

Domain-INDEPENDENT design: the prompt is selected by `profile['synthesis']['kind']`
and loaded from `prompts/<kind>.md` on disk — there is NO per-domain `if`
branch in this module. To add a new synthesis style you only:
  1. write prompts/<newkind>.md with {query} and {context} tokens
  2. set `synthesis: {kind: newkind}` in the domain's domain_config.yaml block
No code change in prompts.py / serve_gpu.py.

Resolution order for a given profile:
  1. profile['synthesis']['prompt'] (inline, already rendered in YAML) — if it
     contains both {query} and {context} tokens, use it verbatim.
  2. prompts/<kind>.md                — file template, {query}/{context} substituted.
  3. generic fallback                — "Answer using ONLY the context…".

Token budget: contexts are truncated to config.MAX_TOKENS_CONTEXT (×4 chars).
"""

import os

import config as C

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_template(kind: str) -> str | None:
    """Load prompts/<kind>.md if it exists, else None."""
    path = os.path.join(_PROMPTS_DIR, f"{kind}.md")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def build_synthesis_prompt(query: str, contexts: list,
                           profile: dict = None) -> str:
    """Build the synthesis (E2B) prompt.

    contexts: list of retrieved text blocks (already in order).
    profile:  domain profile dict (domain_loader.get_domain result) or None.

    Returns the full prompt string for the chat completion.
    """
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    syn = (profile or {}).get("synthesis") or {}
    kind = syn.get("kind")

    # 1. Inline profile prompt (highest precedence) — safe token substitution
    #    only for our two markers; avoids str.format() breaking on literal {...}.
    inline = syn.get("prompt")
    if inline and "{context}" in inline and "{query}" in inline:
        return inline.replace("{context}", ctx).replace("{query}", query) \
                    [: C.MAX_TOKENS_CONTEXT * 4]

    # 2. File template prompts/<kind>.md (generic, no if-branches).
    if kind:
        assert isinstance(kind, str)  # guaranteed by the `if kind` guard above
        tmpl = _load_template(kind)
        if tmpl:
            return tmpl.replace("{context}", ctx).replace("{query}", query) \
                      [: C.MAX_TOKENS_CONTEXT * 4]
        # Unknown kind → log once and fall through to generic.
        print(f"[prompts] no template for synthesis kind='{kind}'; "
              f"using generic. Add prompts/{kind}.md.", flush=True)

    # 3. Generic fallback (no profile / no kind / missing template).
    role = syn.get("role", "knowledge assistant")
    return (
        f"You are a {role}. Answer using ONLY the context. "
        f"Cite source numbers. If missing, say so.\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"
    )[: C.MAX_TOKENS_CONTEXT * 4]
