# Drop your documents here

This is the simplest way to add your own knowledge base to the rag-system.

## How to use

1. Copy your `.md` files into this folder (subfolders are fine — the ingester
   walks recursively):

   ```
   corpus/
     handbook/onboarding.md
     policies/security.md
     notes.md
   ```

2. Make sure the stack is running (`bash run.sh serve` — see the top-level
   README / QUICKSTART).

3. Ingest everything in this folder into the `enterprise` domain:

   ```bash
   bash run.sh ingest                       # ingests ./corpus into 'enterprise'
   # or, explicitly:
   ./venv/bin/python scripts/ingest_corpus_docs.py \
       --docs corpus --domain enterprise --source my-corpus
   ```

   - `--source` is a label for *your* corpus (becomes the doc-id prefix and
     `source_repo` metadata). Use anything memorable, e.g. `handbook`.
   - Re-running is **idempotent**: unchanged files are skipped (HTTP 304),
     only new/changed files are re-embedded.

4. Ask against it:

   ```bash
   bash run.sh ask "your question here" --domain enterprise
   ```

## Notes

- Only `*.md` files are ingested. Convert PDFs/DOCX to markdown first
  (see docs for the `ocr-and-documents` workflow) or add a loader.
- This folder is **git-ignored** (except this README) so your documents are
  never committed. Confidential corpora stay local.
- To ingest into a different domain, pass `--domain <name>` (the domain must
  exist in `domain_config.yaml`).
