"""
retrieve_json.py — headless retrieval for the Hermes plugin.

Reuses ask.py's tested pipeline (embed -> parallel Qdrant+Neo4j -> rerank) but
returns JSON instead of streaming. Optionally synthesizes an answer (E2B).

Usage:
  python retrieve_json.py "query text"              -> {"contexts": [...]}
  python retrieve_json.py --synthesize "query text" -> {"contexts": [...], "answer": "..."}

Exit codes: 0 ok, 2 retrieval error (JSON error body still printed to stdout).
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "hf"))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

import config as C
import ask  # reuse embed_query, parallel_retrieve, condense, synthesize helpers


def _synthesize_nonstream(query: str, contexts: list, profile: dict = None) -> str:
    """Non-streaming synthesis (E2B; ask.synthesize streams to stdout).

    Uses the shared prompts.build_synthesis_prompt (v2.6.0 REQ-5) so the
    domain-aware prompt logic lives in one place.
    """
    import prompts as P
    from openai import OpenAI
    prompt = P.build_synthesis_prompt(query, contexts, profile)
    client = OpenAI(base_url=C.SYNTHESIS_LLM_BASE_URL, api_key=C.SYNTHESIS_LLM_API_KEY)
    resp = client.chat.completions.create(
        model=C.SYNTHESIS_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=C.LLM_MAX_TOKENS_OUT,
        temperature=C.SYNTH_TEMPERATURE,
    )
    msg = resp.choices[0].message
    return (getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or "").strip()


def main():
    argv = sys.argv[1:]
    do_synth = False
    domain = None
    while argv and argv[0].startswith("--"):
        if argv[0] == "--synthesize":
            do_synth = True
        elif argv[0] == "--domain" and len(argv) > 1:
            domain = argv[1]
            argv = argv[1:]
        argv = argv[1:]
    query = argv[0] if argv else ""
    if not query.strip():
        print(json.dumps({"error": "empty query"}))
        sys.exit(2)

    import domain_loader
    profile = domain_loader.get_domain(domain) if domain else None
    # Derive collection from profile when a domain is given (mirrors /ask).
    collections = [profile["collection"]] if profile else None
    entry_strategy = (profile.get("entry_strategy", "keyword")
                      if profile else "keyword")

    try:
        vec = ask.embed_query(query)
        q_res, g_res = ask.parallel_retrieve(vec, query, collections=collections,
                                            entry_strategy=entry_strategy)
        contexts = ask.condense(query, q_res, g_res)
        out = {"query": query, "contexts": contexts, "n_contexts": len(contexts)}
        if do_synth:
            out["answer"] = (_synthesize_nonstream(query, contexts, profile)
                             if contexts else "")
        if domain:
            out["domain"] = domain
        print(json.dumps(out, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "query": query}))
        sys.exit(2)


if __name__ == "__main__":
    main()
