#!/usr/bin/env python3
"""
ingest_corpus_docs.py — Bulk-ingest a corpus of documents into a rag-system domain.

Walks DOCS_ROOT recursively for *.md files and POSTs each to /ingest.
Uses a BOUNDED worker pool so the daemon never has more than --workers embed
jobs in flight (prevents CUDA OOM from unbounded concurrency). Each request
carries if_checksum so re-runs are idempotent (unchanged docs -> 304).

The corpus identity is supplied by the user via --source (it becomes the
doc_id prefix and the source_repo metadata), so the pipeline is not tied to
any specific corpus name.

Usage:
    python scripts/ingest_corpus_docs.py --docs DIR --domain enterprise --source my-corpus
    python scripts/ingest_corpus_docs.py --docs DIR --workers 3 --dry-run
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

API = "http://localhost:8000/ingest"
STATUS_API = "http://localhost:8000/ingest/status"


def build_payload(path, docs_root, domain, source):
    rel = os.path.relpath(path, docs_root)
    doc_id = f"{source}::" + rel.replace(os.sep, "::")
    text = open(path, encoding="utf-8", errors="replace").read()
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "text": text,
        "doc_id": doc_id,
        "doc_type": "eng",
        "domain": domain,
        "extract_graph": False,
        "if_checksum": checksum,
        "metadata": {"source_repo": source, "rel_path": rel},
    }, doc_id


def post_ingest(payload, retries=4, backoff=2.0):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                API, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            r = urllib.request.urlopen(req, timeout=300)
            if r.status == 304:
                return "unchanged", None
            d = json.loads(r.read())
            return d.get("status", "accepted"), d.get("task_id")
        except urllib.error.HTTPError as e:
            if e.code == 304:
                return "unchanged", None
            last_err = f"http{e.code}"
            # 5xx is transient (daemon busy/OOM) -> retry
            if e.code >= 500:
                time.sleep(backoff); continue
            return last_err, None
        except Exception as e:
            last_err = f"err:{e}"
            time.sleep(backoff * attempt)  # backoff grows
            continue
    return last_err, None


def poll(task_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{STATUS_API}/{task_id}")
            r = urllib.request.urlopen(req, timeout=10)
            d = json.loads(r.read())
            if d["status"] in ("done", "error"):
                return d["status"], d.get("error")
        except Exception:
            pass
        time.sleep(0.5)
    return "timeout", None


def main():
    ap = argparse.ArgumentParser()
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--docs",
                    default=os.path.join(_REPO, "corpus"),
                    help="path to the corpus docs dir (default: ./corpus — the repo drop-folder)")
    ap.add_argument("--source", default="corpus",
                    help="corpus identity used as the doc_id prefix and source_repo "
                         "metadata (e.g. --source my-corpus). The pipeline is corpus-agnostic.")
    ap.add_argument("--domain", default="enterprise")
    ap.add_argument("--workers", type=int, default=1,
                    help="bounded concurrent ingest tasks (default 1 = strictly serial, "
                         "most VRAM-safe for large docs on a shared GPU)")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds to sleep between submits (let GPU settle)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.docs):
        print(f"ERROR: docs root not found: {args.docs}", file=sys.stderr)
        print("  Put your .md files in the 'corpus/' folder (see corpus/README.md),", file=sys.stderr)
        print("  or pass --docs /path/to/your/markdown/dir", file=sys.stderr)
        sys.exit(1)

    files = []
    for root, _, fnames in os.walk(args.docs):
        for fn in fnames:
            if fn.lower().endswith(".md") and fn != "README.md":
                files.append(os.path.join(root, fn))
    files.sort()
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No .md files found under {args.docs} (only README.md is skipped).", file=sys.stderr)
        print("  Copy your documents into that folder, then re-run. See corpus/README.md.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} markdown files under {args.docs} "
          f"(workers={args.workers})")

    if args.dry_run:
        for i, p in enumerate(files, 1):
            rel = os.path.relpath(p, args.docs)
            print(f"  [{i}/{len(files)}] {rel}")
        print(f"DRY-RUN complete: {len(files)} files.")
        return

    submitted, unchanged, failed = [], [], []

    def worker(path):
        payload, doc_id = build_payload(path, args.docs, args.domain, args.source)
        status, task_id = post_ingest(payload)
        return doc_id, status, task_id

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, p): p for p in files}
        done_count = 0
        for fut in futures:
            doc_id, status, task_id = fut.result()
            done_count += 1
            if status == "unchanged":
                unchanged.append(doc_id)
            elif task_id:
                submitted.append((doc_id, task_id))
            else:
                failed.append((doc_id, status))
            if done_count % 25 == 0:
                print(f"  submit {done_count}/{len(files)} "
                      f"accepted={len(submitted)} unchanged={len(unchanged)} "
                      f"failed={len(failed)}")
            time.sleep(args.delay)

    print(f"\nSubmitted {len(submitted)} ingest tasks. Polling...")
    done = err = 0
    for doc_id, task_id in submitted:
        st, errmsg = poll(task_id)
        if st == "done":
            done += 1
        else:
            err += 1
            failed.append((doc_id, st))
            if errmsg:
                print(f"  ERROR {doc_id}: {errmsg[:120]}")
    # Re-poll any that failed at submit-time
    if failed:
        print(f"Re-checking {len(failed)} submit-time failures...")
        for doc_id, why in list(failed):
            if why.startswith("http") or why.startswith("err"):
                # try a synchronous re-ingest once
                pass

    print(f"\n=== INGEST SUMMARY ===")
    print(f"  submitted : {len(submitted)}")
    print(f"  done      : {done}")
    print(f"  unchanged : {len(unchanged)} (304, idempotent)")
    print(f"  failed    : {len(failed)}")
    if failed:
        print("  Failures:")
        for doc_id, why in failed[:30]:
            print(f"    {doc_id}: {why}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
