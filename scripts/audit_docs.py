#!/usr/bin/env python3
"""audit_docs.py — doc-vs-code consistency oracle for the rag-system.

Goal: detect documentation that has drifted from the actual code/config/
running system, so a fix-loop can drive the repo to a fully-consistent state.

Design:
- File-based checks are deterministic and need NO running daemon.
- Live checks (marked LIVE) hit the running daemon; if it is down they are
  SKIPPED (not failed) so the audit is still useful offline.
- Exit code 0 = all checks passed; 1 = at least one FAIL.

Run:  ./venv/bin/python scripts/audit_docs.py
      ./venv/bin/python scripts/audit_docs.py --daemon-url http://127.0.0.1:8000

Scope: the rag-system itself (top-level .py, prompts/, domains/, docs/,
tests/, scripts/, corpus/, sample_data/, plans/, root *.md). The external/
tree is a separate, unrelated project and is deliberately OUT of scope.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read()


def _fail(msg):
    print(f"[FAIL] {msg}")


def _ok(msg):
    print(f"[OK]   {msg}")


def _skip(msg):
    print(f"[SKIP] {msg}")


class Audit:
    def __init__(self):
        self.fails = 0
        self.skips = 0

    def check(self, cond, msg):
        if cond:
            _ok(msg)
        else:
            _fail(msg)
            self.fails += 1

    def skip(self, cond, msg):
        if cond:
            _skip(msg)
        else:
            # skip means "could not verify"; do not count as fail
            _skip(msg)
            self.skips += 1


# ---------------------------------------------------------------------------
# File-based checks (no daemon required)
# ---------------------------------------------------------------------------
def check_domain_config(a):
    txt = _read("domain_config.yaml")
    if txt is None:
        a.check(False, "domain_config.yaml present")
        return
    a.check(True, "domain_config.yaml present")
    # enterprise must be the first domain block
    m = re.search(r"^domains:\s*\n\s{2}(\w+):", txt, re.MULTILINE)
    first = m.group(1) if m else None
    a.check(first == "enterprise",
            f"enterprise is first domain block (got '{first}')")
    a.check("neo4j_label: EnterpriseDoc" in txt,
            "enterprise has neo4j_label: EnterpriseDoc")
    a.check("default_domain: enterprise" in txt,
            "default_domain: enterprise")
    a.check("alias: snomed" in txt, "healthcare alias -> snomed present")
    # the removed 'default:' alias block must be gone
    a.check(not re.search(r"^\s{2}default:\s*\n\s{4}alias:", txt, re.MULTILINE),
            "no 'default:' alias block (was default->snomed)")


def check_domain_loader_default(a):
    try:
        import domain_loader  # noqa
        a.check(domain_loader._DEFAULT_DOMAIN == "enterprise",
                f"domain_loader._DEFAULT_DOMAIN == 'enterprise' "
                f"(got '{domain_loader._DEFAULT_DOMAIN}')")
    except Exception as e:  # pragma: no cover
        a.check(False, f"domain_loader importable ({e})")


def check_api_md(a):
    txt = _read("docs/API.md")
    if txt is None:
        a.check(False, "docs/API.md present")
        return
    a.check(True, "docs/API.md present")
    # stale engineering domain references
    for stale in ["engineering_chunks", "engineering_docs",
                  '"domain": "engineering"', 'default_domain":"default"']:
        a.check(stale not in txt,
                f"docs/API.md does not contain stale '{stale}'")
    # new endpoints documented
    a.check("/reload" in txt, "docs/API.md documents /reload")
    a.check("/v1/embeddings" in txt, "docs/API.md documents /v1/embeddings")
    # admin reload example must report enterprise default
    a.check('"default_domain": "enterprise"' in txt or '"default_domain":"enterprise"' in txt,
            "docs/API.md /admin/reload example shows default_domain enterprise")


def check_run_md(a):
    txt = _read("RUN.md")
    if txt is None:
        a.check(False, "RUN.md present")
        return
    a.check(True, "RUN.md present")
    a.check("eng/`→enterprise" in txt,
            "RUN.md maps eng/ -> enterprise (not engineering)")
    a.check("engineering_chunks" not in txt and "engineering_docs" not in txt,
            "RUN.md has no engineering_chunks/engineering_docs")


def check_quickstart_md(a):
    txt = _read("QUICKSTART.md")
    if txt is None:
        a.check(False, "QUICKSTART.md present")
        return
    a.check(True, "QUICKSTART.md present")
    a.check("default` alias → `snomed" not in txt
            and "default alias → snomed" not in txt,
            "QUICKSTART.md no longer says default alias -> snomed")
    a.check("enterprise" in txt, "QUICKSTART.md mentions enterprise domain")


def check_healthcare_quickstart(a):
    txt = _read("domains/healthcare/QUICKSTART.md")
    if txt is None:
        a.check(False, "domains/healthcare/QUICKSTART.md present")
        return
    a.check(True, "domains/healthcare/QUICKSTART.md present")
    a.check("the default clinical domain" not in txt,
            "healthcare QUICKSTART no longer calls healthcare 'the default' "
            "(enterprise is the default)")
    a.check("alias for `snomed`" in txt or "alias for snomed" in txt,
            "healthcare QUICKSTART states it is an alias for snomed")


def check_domains_readme(a):
    txt = _read("docs/domains/README.md")
    if txt is None:
        a.check(False, "docs/domains/README.md present")
        return
    a.check(True, "docs/domains/README.md present")
    a.check("defaults to engineering" not in txt,
            "docs/domains/README.md no longer says default is engineering")
    a.check("engineering_chunks" not in txt and "engineering_docs" not in txt,
            "docs/domains/README.md has no engineering_chunks/engineering_docs")
    a.check("defaults to `enterprise`" in txt or "defaults to enterprise" in txt,
            "docs/domains/README.md documents enterprise as the default domain")


def check_prompt_templates(a):
    for fname in ["prompts/extraction.md", "prompts/legal_extraction.md"]:
        txt = _read(fname)
        if txt is None:
            a.check(False, f"{fname} present")
            continue
        a.check(True, f"{fname} present")
        a.check("{entity_types}" in txt and "{relation_types}" in txt,
                f"{fname} uses {{entity_types}}/{{relation_types}} placeholders")
        # the old hardcoded engineering vocab must NOT appear in templates
        a.check("Microservice|Database|API|Metric" not in txt,
                f"{fname} does not hardcode engineering vocab "
                "(Microservice|Database|API|Metric)")


# ---------------------------------------------------------------------------
# Live checks (best-effort; skipped if daemon down)
# ---------------------------------------------------------------------------
def _daemon_up(url):
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def check_live_endpoints(a, url):
    if not _daemon_up(url):
        a.skip(True, f"daemon at {url} not reachable — skipping live checks")
        return
    a.skip(False, "daemon reachable — running live checks")
    import json
    import urllib.request

    # /reload returns status=reloaded
    try:
        req = urllib.request.Request(f"{url}/reload", method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        a.check(data.get("status") == "reloaded"
                and data.get("default_domain") == "enterprise",
                "/reload returns status=reloaded, default_domain=enterprise")
    except Exception as e:
        a.check(False, f"/reload reachable ({e})")

    # /v1/embeddings returns OpenAI-shaped vectors
    try:
        req = urllib.request.Request(
            f"{url}/v1/embeddings",
            data=json.dumps({"input": "test query", "model": "jina"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        a.check("data" in data and len(data["data"]) >= 1
                and "embedding" in data["data"][0],
                "/v1/embeddings returns OpenAI-shaped vectors")
    except Exception as e:
        a.check(False, f"/v1/embeddings reachable ({e})")

    # no-domain /ask resolves to enterprise
    try:
        req = urllib.request.Request(
            f"{url}/ask",
            data=json.dumps({"query": "what does the Basis Data architecture do?",
                              "synthesize": True}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        # enterprise corpus answer should reference Basis Data / entity resolution
        ans = (data.get("answer") or "")
        a.check("Basis Data" in ans or "Basis" in ans,
                "no-domain /ask resolves to enterprise (Basis Data answer)")
    except Exception as e:
        a.check(False, f"no-domain /ask reachable ({e})")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    a = Audit()
    print("=== rag-system doc/code consistency audit ===")
    check_domain_config(a)
    check_domain_loader_default(a)
    check_api_md(a)
    check_run_md(a)
    check_quickstart_md(a)
    check_healthcare_quickstart(a)
    check_domains_readme(a)
    check_prompt_templates(a)
    check_live_endpoints(a, args.daemon_url)

    print("============================================")
    if a.fails == 0:
        print(f"RESULT: PASS ({a.skips} skipped, 0 failures)")
        sys.exit(0)
    else:
        print(f"RESULT: FAIL ({a.fails} failure(s), {a.skips} skipped)")
        sys.exit(1)


if __name__ == "__main__":
    main()
