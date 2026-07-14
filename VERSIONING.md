# Versioning

This project uses **Semantic Versioning within the 0.x.x experimental
series**. We are explicitly **NOT at 1.0** — the system is experimental and
the configuration / API contract still changes.

> **Why 0.x.x?** Earlier git tags (`v1.0.0`, `v3.1.x`) were created before the
> project reached a stable, contract-locked 1.0. They implied maturity that did
> not exist. All those tags were **deleted** and the history was reset to a
> single `v0.1.0` baseline on 2026-07-14. Do not recreate `v1.x`/`v3.x` tags.

## Version number: `0.MINOR.PATCH`

| Bump | When |
|------|------|
| `MINOR` | A **breaking change** to a public contract: an endpoint's required field changes, a domain key is renamed, the `/ask` response shape changes, `domain_config.yaml` schema changes in a non-additive way. |
| `PATCH` | Backward-compatible change: a new domain/companion added behind the existing contract, a bug fix, a doc fix, a new optional endpoint/field. |

We do **not** use strict SemVer MAJOR in 0.x — `0.x` already signals
"anything may change." The first `1.0.0` will be cut only when the
domain/companion contract and `/ask` response shape are frozen and documented
as stable.

## Source of truth

- `VERSION` — the single file holding the current version string (`0.1.0`).
  Read it in code via `config.__version__`.
- `CHANGELOG.md` — human-readable history, grouped by version.
- Git tags — `vX.Y.Z`, created with `git tag -a vX.Y.Z -m "..."`.

## Workflow

1. Make the change. Keep config/API additive where possible.
2. Bump `VERSION` (and add a `CHANGELOG.md` entry) per the table above.
3. Commit: `git commit -m "release: vX.Y.Z" ...` or a feature commit that
   notes the version bump intent.
4. Tag and push:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z: <one-line summary>"
   git push origin main --tags
   ```
5. Never force-push tags. If a tag is wrong, cut a new PATCH and note the
   correction in `CHANGELOG.md`.

## What is frozen at 1.0 (target)

- `domain_config.yaml` schema (adding a domain = additive, safe).
- `/ask` and `/ingest` request/response contracts.
- The companion `companions:` + `_signal` + dual-evidence behavior.
- The `/embed_late` contract (`{text, doc_id, strategy, chunk_size, overlap}`).
