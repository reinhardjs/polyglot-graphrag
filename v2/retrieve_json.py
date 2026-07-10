"""
retrieve_json.py — headless retrieval for the Hermes plugin.

Reuses ask.py's tested pipeline (embed -> parallel Qdrant+Neo4j -> rerank) but
returns JSON instead of streaming. Optionally synthesizes an answer with E4B.

Usage:
  python retrieve_json.py "query text"              -> {"contexts": [...]}
  python retrieve_json.py --synthesize "query text" -> {"contexts": [...], "answer": "..."}

Exit codes: 0 ok, 2 retrieval error (JSON error body still printed to stdout).
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/data-970-plus/hf_cache")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

import config as C
import ask  # reuse embed_query, parallel_retrieve, condense, synthesize helpers


def _synthesize_nonstream(query: str, contexts: list) -> str:
    """Non-streaming synthesis via E4B (ask.synthesize streams to stdout)."""
    from openai import OpenAI
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "Answer the question using ONLY the context. "
        "Cite source numbers. If missing, say so.\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"
    )[: C.MAX_TOKENS_CONTEXT * 4]
    client = OpenAI(base_url=C.SYNTHESIS_LLM_BASE_URL, api_key=C.SYNTHESIS_LLM_API_KEY)
    resp = client.chat.completions.create(
        model=C.SYNTHESIS_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=C.LLM_MAX_TOKENS_OUT,
    )
    msg = resp.choices[0].message
    return (getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or "").strip()


def main():
    argv = sys.argv[1:]
    do_synth = False
    if argv and argv[0] == "--synthesize":
        do_synth = True
        argv = argv[1:]
    query = argv[0] if argv else ""
    if not query.strip():
        print(json.dumps({"error": "empty query"}))
        sys.exit(2)

    try:
        vec = ask.embed_query(query)
        q_res, g_res = ask.parallel_retrieve(vec, query)
        contexts = ask.condense(query, q_res, g_res)
        out = {"query": query, "contexts": contexts, "n_contexts": len(contexts)}
        if do_synth:
            out["answer"] = _synthesize_nonstream(query, contexts) if contexts else ""
        print(json.dumps(out, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "query": query}))
        sys.exit(2)


if __name__ == "__main__":
    main()
