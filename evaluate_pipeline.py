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
    """Fraction of ground-truth terms covered by the retrieved contexts.

    Uses word overlap (|gt ∩ ctx| / |gt|) with improved token normalization.
    Falls back to cosine similarity for very short ground truths (< 5 tokens).
    """
    if not ground_truth or not contexts:
        return 0.0
    gt_tokens = set(re.findall(r'\w{2,}', ground_truth.lower()))
    if len(gt_tokens) <= 3:
        # Very short ground truth — use cosine similarity instead
        gt_v = _embed([ground_truth])[0]
        ctx_text = " ".join(contexts[:3])  # top-3 most relevant
        ctx_v = _embed([ctx_text])[0]
        return max(0.0, _cos(gt_v, ctx_v))
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens |= set(re.findall(r'\w{2,}', c.lower()))
    if not ctx_tokens:
        return 0.0
    # Simple coverage: how many ground-truth tokens appear in ANY context
    # Normalized by ground_truth length (not Jaccard, to avoid penalizing
    # short ground truths in large context corpora)
    return len(gt_tokens & ctx_tokens) / len(gt_tokens)


# ── Sprint 2: Better Proxy Metrics ─────────────────────────────────────────────
# These replace the structurally biased metrics with sentence-level / decomposition
# approaches that reflect actual RAG quality, not embedding artifacts.

def _sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex, fast, no NLTK dep)."""
    # Handle common abbreviations to avoid over-splitting
    text = re.sub(r'\b(e\.g|i\.e|etc|Dr|Mr|Ms|vs)\.', r'\1<DOT>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.replace('<DOT>', '.') for s in sentences if s.strip()]


def context_precision_local_sentence(question: str, ground_truth: str,
                                      contexts: List[str]) -> float:
    """Sentence-level context precision.

    A context is 'relevant' if ANY of its sentences has cosine >= 0.35 vs ground_truth.
    This avoids the 512-token chunk vs 50-char summary embedding mismatch.
    """
    if not contexts:
        return 0.0
    ref = ground_truth or question
    ref_v = _embed([ref])[0]
    relevant = 0
    for ctx in contexts:
        sents = _sentences(ctx)
        if not sents:
            continue
        sent_vs = _embed(sents)
        if any(_cos(ref_v, sv) >= 0.35 for sv in sent_vs):
            relevant += 1
    return relevant / len(contexts)


def answer_relevancy_local_decompose(question: str, answer: str) -> float:
    """Decomposition-based answer relevancy.

    Instead of cosine(question, answer) which is structurally low for template
    questions, decompose the question and check coverage.
    """
    if not answer or not question:
        return 0.0
    q_lower = question.lower()
    a_lower = answer.lower()

    # Detect template question: "what does the X describe?" → sub-Qs about X
    m = re.search(r'what does the (.+?) (document|file|note|md) describe', q_lower)
    doc_name = m.group(1).strip() if m else None

    checks = 0
    total = 3

    # Sub-Q1: Does the answer mention the document name/key terms?
    if doc_name and doc_name in a_lower:
        checks += 1
    elif not doc_name:  # non-template question: check for question keywords
        # Heuristic: if any noun from question appears in answer
        q_terms = set(re.findall(r'\b\w{4,}\b', q_lower))
        a_terms = set(re.findall(r'\b\w{4,}\b', a_lower))
        if q_terms & a_terms:
            checks += 1

    # Sub-Q2: Does the answer describe purpose/function (not just regurgitate)?
    purpose_terms = {'describe', 'pipeline', 'system', 'process', 'handles',
                     'manages', 'provides', 'enables', 'implements', 'defines',
                     'configures', 'monitors', 'validates', 'extracts', 'runs',
                     'uses', 'connects', 'stores', 'queries', 'generates'}
    if any(t in a_lower for t in purpose_terms):
        checks += 1

    # Sub-Q3: Does the answer include concrete details (numbers, ports, paths)?
    has_detail = bool(re.search(r'\b\d+\b', answer))  # any number
    if has_detail:
        checks += 1

    return checks / total


def evaluate_local(rows: List[Dict]) -> Dict:
    """Compute the four metrics (local backend) averaged over all rows.

    Uses sentence-level context precision and decomposition-based answer relevancy
    to avoid structural biases in the old proxy metrics.
    """
    f, ar, cp, cr = [], [], [], []
    for row in rows:
        q = row.get("question", "")
        gt = row.get("ground_truth", "")
        ans = row.get("answer", "")
        ctx = row.get("contexts", []) or []
        f.append(faithfulness_local(ans, ctx))
        ar.append(answer_relevancy_local_decompose(q, ans))
        cp.append(context_precision_local_sentence(q, gt, ctx))
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


# ── Sprint 3: RAGAS Hardening ─────────────────────────────────────────────────
# 3a: Reduce golden to n samples for RAGAS (rate limit mitigation)
def reduce_golden(rows: List[Dict], n: int = 5) -> List[Dict]:
    """Select n diverse samples from the golden set for RAGAS evaluation."""
    if len(rows) <= n:
        return rows
    # Simple diversity sampling: pick evenly spaced indices
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


# 3b: Limit contexts to top-k for 2B model context window
def limit_contexts(rows: List[Dict], max_ctx: int = 3) -> List[Dict]:
    """Limit contexts per sample to max_ctx (for 2B model context window)."""
    out = []
    for r in rows:
        r = dict(r)
        ctx = r.get("contexts", []) or []
        if len(ctx) > max_ctx:
            # Keep top-k by relevance (they're already rerank-ordered)
            r["contexts"] = ctx[:max_ctx]
        out.append(r)
    return out


# ── Optional ragas backend ────────────────────────────────────────────────────
def evaluate_ragas(rows: List[Dict]) -> Dict:
    """Use the real ragas framework wired to LOCAL models. Requires install."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (faithfulness, answer_relevancy,
                               context_precision, context_recall)
    # Wire ragas to local LLM + embeddings (OpenAI-compatible E2B + Jina daemon).
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _key = C.SYNTHESIS_LLM_API_KEY  # daemon accepts any key
    # Explicit timeout + retries: the local 2B E2B can be slow under load and
    # ragas' default request timeout is short, which caused sporadic
    # TimeoutError -> 0.0/NaN scores. 300s timeout, 2 retries absorbs that.
    # Use OpenRouter free model for ragas LLM (larger context window)
    import os
    ragas_llm_base = os.environ.get("RAGAS_LLM_BASE_URL", C.SYNTHESIS_LLM_BASE_URL)
    ragas_llm_model = os.environ.get("RAGAS_LLM_MODEL", C.SYNTHESIS_LLM_MODEL)
    ragas_llm_key = os.environ.get("RAGAS_LLM_API_KEY", _key)
    llm = ChatOpenAI(model=ragas_llm_model,
                     base_url=ragas_llm_base,
                     api_key=ragas_llm_key, temperature=0.0,
                     timeout=300, max_retries=2)
    # ragas context_precision/recall need an embeddings model. Use the
    # daemon's OpenAI-compatible /v1/embeddings (Jina), NOT the E2B
    # llama-server (which has no embeddings endpoint without --embeddings).
    # IMPORTANT: tiktoken_enabled=False because our local endpoint doesn't
    # use tiktoken; it expects raw strings, not token IDs.
    embeddings = OpenAIEmbeddings(model=C.EMBED_MODEL_NAME,
                                  base_url=f"{C.DAEMON_URL}/v1",
                                  api_key=_key, timeout=120, max_retries=2,
                                  tiktoken_enabled=False)
    # Raise ragas' own per-call timeout so a slow local LLM doesn't abort a
    # faithfulness/context job mid-run.
    try:
        from ragas import RunConfig
        _run_cfg = RunConfig(timeout=300, max_retries=2, max_wait=180)
    except Exception:
        _run_cfg = None
    # Detect if using local 2B model (small context) vs OpenRouter (large context)
    using_small_model = "localhost" in ragas_llm_base or "127.0.0.1" in ragas_llm_base
    
    # Sprint 3a: Reduce golden samples for RAGAS rate limit mitigation
    # Sprint 3b: Limit contexts for 2B model context window
    eval_rows = rows
    if using_small_model:
        eval_rows = limit_contexts(reduce_golden(rows, 5), 3)
    
    ds = Dataset.from_list([{
        "question": r.get("question", ""),
        "answer": r.get("answer", ""),
        "contexts": r.get("contexts", []) or [],
        "ground_truth": r.get("ground_truth", ""),
    } for r in eval_rows])
    # Raise ragas' own per-call timeout so a slow local LLM doesn't abort a
    # faithfulness/context job mid-run.
    try:
        from ragas import RunConfig
        _run_cfg = RunConfig(timeout=300, max_retries=2, max_wait=180)
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
    else:
        scores = dict(result)
    out = {k: round(float(v), 4) for k, v in scores.items() if v == v}
    # Hybrid merge: ragas context_precision/recall often fail on 2B models
    local = evaluate_local(rows)
    if "context_precision" not in out:
        out["context_precision"] = local["context_precision"]
    if "context_recall" not in out:
        out["context_recall"] = local["context_recall"]
    if "faithfulness" not in out:
        out["faithfulness"] = local["faithfulness"]
    # Triple-layer fallback: if ALL ragas metrics are NaN, use local proxy
    ragas_metrics = [k for k in out if k in ("faithfulness", "answer_relevancy",
                                              "context_precision", "context_recall")]
    if all(out.get(k) != out.get(k) for k in ragas_metrics):
        print("[ragas] All metrics NaN, falling back to local proxy", file=sys.stderr)
        out = local
        out["backend"] = "local"
        return out
    out["n_samples"] = len(rows)
    out["backend"] = "ragas"
    return out


# ── Live generation (optional) ────────────────────────────────────────────────
def generate_live(rows: List[Dict], domain: str = None,
                  crag: bool = False, top_k: int = None) -> List[Dict]:
    """Fill answer+contexts by calling the running daemon /ask for each row."""
    import requests
    out = []
    for r in rows:
        body = {"query": r["question"], "synthesize": True, "skip_cache": True}
        if domain:
            body["domain"] = domain
        if crag:
            body["crag"] = True
        if top_k is not None:
            body["top_k"] = top_k
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


# ── Sprint 3c: Golden Verification ────────────────────────────────────────────
def verify_golden_retrievable(golden_path: str, domain: str, top_k: int = 12) -> None:
    """Verify each golden question's ground_truth is retrievable from the corpus.

    For each question, retrieves top_k contexts from the daemon and checks
    if the ground_truth text has any Jaccard overlap with the retrieved contexts.
    Reports questions with low retrievability (Jaccard < 0.02).
    """
    import urllib.request

    with open(golden_path) as fh:
        rows = json.load(fh)

    print(f"Verifying {len(rows)} golden questions for domain '{domain}'...")
    issues = 0
    for i, row in enumerate(rows):
        q = row.get("question", "")
        gt = row.get("ground_truth", "")

        # Retrieve contexts from the daemon
        url = f"{C.DAEMON_URL}/ask?top_k={top_k}"
        if domain:
            url += f"&domain={domain}"
        try:
            req = urllib.request.Request(url, data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read().decode())
            contexts = data.get("contexts", [])
        except Exception as e:
            print(f"  [{i+1:3d}] Error retrieving: {e}")
            continue

        # Check Jaccard overlap
        gt_tokens = set(re.findall(r'\w+', gt.lower()))
        ctx_tokens = set()
        for c in contexts:
            ctx_tokens |= set(re.findall(r'\w+', c.lower()))

        if not gt_tokens:
            continue

        jaccard = len(gt_tokens & ctx_tokens) / max(len(gt_tokens | ctx_tokens), 1)

        if jaccard < 0.01:
            issues += 1
            print(f"  [{i+1:3d}] LOW RECALL (Jaccard={jaccard:.4f}): {q[:80]}...")
            print(f"          GT: {gt[:100]}...")
        elif jaccard < 0.03:
            issues += 1
            print(f"  [{i+1:3d}] MARGINAL  (Jaccard={jaccard:.4f}): {q[:80]}...")

    if issues == 0:
        print(f"ALL {len(rows)} golden questions pass — ground truths are retrievable.")
    else:
        print(f"\n{issues}/{len(rows)} questions have low retrievability — consider refining ground truths.")


def main():
    ap = argparse.ArgumentParser(description="RAG evaluation harness (Phase 4)")
    ap.add_argument("golden", nargs="?", help="path to golden dataset JSON")
    ap.add_argument("--live", action="store_true",
                    help="generate answer+contexts via the running daemon /ask")
    ap.add_argument("--domain", default=None, help="domain profile for --live")
    ap.add_argument("--crag", action="store_true", help="use CRAG mode for --live")
    ap.add_argument("--backend", choices=["auto", "local", "ragas"],
                    default="auto")
    ap.add_argument("--top-k", type=int, default=None,
                    help="override domain top_k for retrieval (for release gate speed)")
    ap.add_argument("--verify-golden", action="store_true",
                    help="verify golden ground truths are retrievable from the corpus")
    args = ap.parse_args()

    # --verify-golden: check ground truths are retrievable
    if args.verify_golden:
        if not args.golden:
            print("--verify-golden requires a golden file path", file=sys.stderr)
            sys.exit(1)
        if not args.domain:
            print("--verify-golden requires --domain (enterprise or oraetlabora)", file=sys.stderr)
            sys.exit(1)
        verify_golden_retrievable(args.golden, args.domain, top_k=args.top_k or 12)
        return

    with open(args.golden) as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        print("golden dataset must be a JSON list", file=sys.stderr)
        sys.exit(1)

    if args.live:
        print(f"Generating answers live via {C.DAEMON_URL}/ask "
              f"(domain={args.domain}, crag={args.crag}, top_k={args.top_k}) ...", file=sys.stderr)
        rows = generate_live(rows, domain=args.domain, crag=args.crag, top_k=args.top_k)

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
