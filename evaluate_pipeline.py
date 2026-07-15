"""Phase 4 — Evaluation Harness (Ragas-compatible, local-first).

Computes the four core RAG metrics over a domain-specific "golden dataset":
    - Faithfulness      : is the answer grounded in the retrieved contexts?
    - Answer Relevancy  : does the answer address the question?
    - Context Precision : are the retrieved contexts relevant (signal vs noise)?
    - Context Recall    : do the contexts cover the ground-truth answer?

Two backends (auto-selected):
  1. ragas  — if `ragas` + `datasets` are installed AND EVAL_USE_RAGAS=1, use the
              real ragas framework wired to the LOCAL E2B LLM + Jina embeddings.
  2. local  — otherwise, built-in implementations that score with the local Jina
              embed daemon (cosine similarity) + lexical coverage. Zero cloud,
              zero heavy deps. Numbers are directional and reproducible.

Golden dataset format (JSON list):
    [
      {
        "question": "who reported BUG-204?",
        "ground_truth": "BUG-204 was reported by Alice.",
        "answer": "Alice reported BUG-204.",          # generated (optional*)
        "contexts": ["...retrieved chunk...", "..."]   # retrieved (optional*)
      },
      ...
    ]
  * If `answer`/`contexts` are missing, the harness can generate them live by
    calling the running daemon's /ask endpoint (--live).

Usage:
    python evaluate_pipeline.py sample_data/golden/engineering.json
    python evaluate_pipeline.py golden.json --live --domain medical --crag
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

import config as C


# ── Local embedding (reuse the running daemon; fall back to direct model) ──────
def _embed(texts: List[str]) -> "list":
    """Embed a list of texts via the GPU daemon /embed_query (one by one).

    Falls back to a fresh SentenceTransformer if the daemon is down.
    """
    import numpy as np
    import requests
    vecs = []
    use_daemon = True
    for t in texts:
        if use_daemon:
            try:
                r = requests.post(C.DAEMON_URL + "/embed_query",
                                  json={"text": t}, timeout=30)
                vecs.append(np.array(r.json()["vector"], dtype=float))
                continue
            except Exception:
                use_daemon = False
        # direct fallback
        from sentence_transformers import SentenceTransformer
        global _ST
        if "_ST" not in globals() or _ST is None:
            _ST = SentenceTransformer(C.EMBED_MODEL_NAME, trust_remote_code=True)
        vecs.append(np.array(_ST.encode(t), dtype=float))
    return vecs


_ST = None


def _cos(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_STOP = {"the", "a", "an", "of", "to", "is", "are", "was", "were", "and", "or",
         "in", "on", "for", "what", "who", "when", "where", "which", "how",
         "why", "does", "do", "did", "has", "have", "with", "that", "this",
         "by", "be", "it", "as", "at", "from"}


def _terms(text: str) -> set:
    return set(re.findall(r"\w+", (text or "").lower())) - _STOP


# ── Local metric implementations ──────────────────────────────────────────────
def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def faithfulness_local(answer: str, contexts: List[str]) -> float:
    """Fraction of answer sentences supported by the contexts.

    A sentence is "supported" if its content terms are substantially covered by
    the union of context terms (>= 0.5 overlap). Grounding proxy without an LLM.

    An explicit ABSTENTION (the model honestly says the context lacks the answer)
    is faithful by construction — it makes no unsupported claim. We score those
    as fully faithful so the proxy does not penalize correct "I don't know" answers.
    """
    a = (answer or "").strip()
    if not a:
        return 0.0
    low = a.lower()
    _ABSTAIN = ("does not contain", "does not describe", "does not mention",
                "not mentioned", "no information", "cannot answer",
                "cannot find", "context does not", "provided context does",
                "the context does not")
    if any(p in low for p in _ABSTAIN):
        return 1.0
    sents = _sentences(a)
    if not sents:
        return 0.0
    ctx_terms = set()
    for c in contexts:
        ctx_terms |= _terms(c)
    supported = 0
    for s in sents:
        st = _terms(s)
        if not st:
            continue
        if len(st & ctx_terms) / len(st) >= 0.5:
            supported += 1
    return supported / len(sents)


def answer_relevancy_local(question: str, answer: str) -> float:
    """Cosine similarity between the question and the generated answer."""
    if not answer or not question:
        return 0.0
    qv, av = _embed([question, answer])
    return max(0.0, _cos(qv, av))


def context_precision_local(question: str, ground_truth: str,
                            contexts: List[str]) -> float:
    """Fraction of retrieved contexts that are relevant to the answer.

    A context is relevant if its embedding cosine vs the ground_truth (or
    question, if no gt) exceeds a threshold. Precision = relevant / retrieved.
    """
    if not contexts:
        return 0.0
    ref = ground_truth or question
    ref_v = _embed([ref])[0]
    ctx_vs = _embed(contexts)
    rel = sum(1 for cv in ctx_vs if _cos(ref_v, cv) >= 0.35)
    return rel / len(contexts)


def context_recall_local(ground_truth: str, contexts: List[str]) -> float:
    """Fraction of ground-truth terms covered by the retrieved contexts."""
    gt = _terms(ground_truth)
    if not gt:
        return 0.0
    ctx_terms = set()
    for c in contexts:
        ctx_terms |= _terms(c)
    return len(gt & ctx_terms) / len(gt)


def evaluate_local(rows: List[Dict]) -> Dict:
    """Compute the four metrics (local backend) averaged over all rows."""
    f, ar, cp, cr = [], [], [], []
    for row in rows:
        q = row.get("question", "")
        gt = row.get("ground_truth", "")
        ans = row.get("answer", "")
        ctx = row.get("contexts", []) or []
        f.append(faithfulness_local(ans, ctx))
        ar.append(answer_relevancy_local(q, ans))
        cp.append(context_precision_local(q, gt, ctx))
        cr.append(context_recall_local(gt, ctx))
    n = max(1, len(rows))
    return {
        "faithfulness": round(sum(f) / n, 4),
        "answer_relevancy": round(sum(ar) / n, 4),
        "context_precision": round(sum(cp) / n, 4),
        "context_recall": round(sum(cr) / n, 4),
        "n_samples": len(rows),
        "backend": "local",
    }


# ── Optional ragas backend ────────────────────────────────────────────────────
def evaluate_ragas(rows: List[Dict]) -> Dict:
    """Use the real ragas framework wired to LOCAL models. Requires install."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (faithfulness, answer_relevancy,
                               context_precision, context_recall)
    ds = Dataset.from_list([{
        "question": r.get("question", ""),
        "answer": r.get("answer", ""),
        "contexts": r.get("contexts", []) or [],
        "ground_truth": r.get("ground_truth", ""),
    } for r in rows])
    # Wire ragas to local LLM + embeddings (OpenAI-compatible E2B + Jina daemon).
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _key = C.SYNTHESIS_LLM_API_KEY  # daemon accepts any key
    # Explicit timeout + retries: the local 2B E2B can be slow under load and
    # ragas' default request timeout is short, which caused sporadic
    # TimeoutError -> 0.0/NaN scores. 300s timeout, 2 retries absorbs that.
    llm = ChatOpenAI(model=C.SYNTHESIS_LLM_MODEL,
                     base_url=C.SYNTHESIS_LLM_BASE_URL,
                     api_key=_key, temperature=0.0,
                     timeout=300, max_retries=2)
    # ragas context_precision/recall need an embeddings model. Use the
    # daemon's OpenAI-compatible /v1/embeddings (Jina), NOT the E2B
    # llama-server (which has no embeddings endpoint without --embeddings).
    embeddings = OpenAIEmbeddings(model=C.EMBED_MODEL_NAME,
                                  base_url=f"{C.DAEMON_URL}/v1",
                                  api_key=_key, timeout=120, max_retries=2)
    # Raise ragas' own per-call timeout so a slow local LLM doesn't abort a
    # faithfulness/context job mid-run.
    try:
        from ragas import RunConfig
        # raise_exceptions=False: a 2B model often fails ragas' structured
        # context_recall/precision prompts (None -> parse error). Swallow those
        # so evaluate() returns partial scores; we override the broken context
        # metrics with the reliable local proxy in the merge below.
        _run_cfg = RunConfig(timeout=300, max_retries=2, max_wait=180,
                              raise_exceptions=False)
    except Exception:
        _run_cfg = None
    eval_kwargs = dict(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    if _run_cfg is not None:
        eval_kwargs["run_config"] = _run_cfg
    result = evaluate(**eval_kwargs)
    # ragas 0.2.x returns an EvaluationResult; convert to a plain dict of
    # metric-name -> mean score across all samples.
    if hasattr(result, "scores") and isinstance(result.scores, list):
        # list of per-row dicts -> column means
        import collections as _coll
        _acc = _coll.defaultdict(list)
        for row in result.scores:
            for k, v in row.items():
                _acc[k].append(v)
        scores = {k: sum(v) / len(v) for k, v in _acc.items()}
    elif hasattr(result, "scores"):
        scores = dict(result.scores)
    elif hasattr(result, "to_pandas"):
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
    else:  # older ragas: dict-like
        scores = dict(result)
    out = {k: round(float(v), 4) for k, v in scores.items() if v == v}  # drop NaN
    # ── Hybrid merge ────────────────────────────────────────────────────────
    # ragas' faithfulness is a true SEMANTIC grounding check — that's the metric
    # worth the heavy deps. But ragas' context_precision/context_recall require
    # the LLM to emit strict structured classifications, which a tiny 2B model
    # (E2B) frequently fails to produce (None/parse errors -> 0.0/NaN). So we
    # keep those two retrieval-tightness metrics from the reliable LOCAL proxy
    # and override ragas' (often broken) values with them.
    local = evaluate_local(rows)
    out["context_precision"] = local["context_precision"]
    out["context_recall"] = local["context_recall"]
    # If ragas faithfulness came back NaN/unusable, fall back to local proxy.
    if "faithfulness" not in out or out.get("faithfulness") != out.get("faithfulness"):
        out["faithfulness"] = local["faithfulness"]
    out["n_samples"] = len(rows)
    out["backend"] = "ragas"
    return out


# ── Live generation (optional) ────────────────────────────────────────────────
def generate_live(rows: List[Dict], domain: str = None,
                  crag: bool = False) -> List[Dict]:
    """Fill answer+contexts by calling the running daemon /ask for each row."""
    import requests
    out = []
    for r in rows:
        body = {"query": r["question"], "synthesize": True, "skip_cache": True}
        if domain:
            body["domain"] = domain
        if crag:
            body["crag"] = True
        try:
            resp = requests.post(C.DAEMON_URL + "/ask", json=body, timeout=300).json()
            r = dict(r)
            r["answer"] = resp.get("answer", "")
            r["contexts"] = resp.get("contexts", []) or []
        except Exception as e:
            r = dict(r); r["answer"] = ""; r["contexts"] = []
            print(f"  ! live gen failed for {r['question'][:40]}: {e}",
                  file=sys.stderr)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description="RAG evaluation harness (Phase 4)")
    ap.add_argument("golden", help="path to golden dataset JSON")
    ap.add_argument("--live", action="store_true",
                    help="generate answer+contexts via the running daemon /ask")
    ap.add_argument("--domain", default=None, help="domain profile for --live")
    ap.add_argument("--crag", action="store_true", help="use CRAG mode for --live")
    ap.add_argument("--backend", choices=["auto", "local", "ragas"],
                    default="auto")
    args = ap.parse_args()

    with open(args.golden) as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        print("golden dataset must be a JSON list", file=sys.stderr)
        sys.exit(1)

    if args.live:
        print(f"Generating answers live via {C.DAEMON_URL}/ask "
              f"(domain={args.domain}, crag={args.crag}) ...", file=sys.stderr)
        rows = generate_live(rows, domain=args.domain, crag=args.crag)

    backend = args.backend
    if backend == "auto":
        want_ragas = os.environ.get("EVAL_USE_RAGAS") == "1"
        try:
            import ragas, datasets  # noqa: F401
            backend = "ragas" if want_ragas else "local"
        except Exception:
            backend = "local"

    if backend == "ragas":
        metrics = evaluate_ragas(rows)
    else:
        metrics = evaluate_local(rows)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
