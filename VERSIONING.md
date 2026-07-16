# Versioning

This project reached a **stable, contract-locked `v1.0.0`** on 2026-07-16. The
domain/companion contract and the `/ask` response shape are frozen and documented
as stable, so the system is published as `1.0.0`.

> **History note.** An earlier `VERSIONING.md` stated "we are NOT at 1.0" and
> that the `v1.0.0` tag had been deleted, resetting history to a `v0.1.0`
> baseline (2026-07-14). That reset was a **false start** made before the
> domain/profile contract and `/ask` response shape were finalized. Once those
> contracts froze (multi-domain `domain_config.yaml`, federated retrieval,
> enterprise self-docs auto-seed, E2B serving extraction + synthesis, BGE rerank
> on GPU, answer-quality + doc-consistency + synthesis-latency release gates),
> `v1.0.0` became the genuine stable baseline and the tag was re-pointed to
> it. The `0.x.x` experimental framing no longer applies.

## Version number: `1.MINOR.PATCH`

| Bump | When |
|------|------|
| `MINOR` | A **breaking change** to a public contract: an endpoint's required field changes, a domain key is renamed, the `/ask` response shape changes, `domain_config.yaml` schema changes in a non-additive way. |
| `PATCH` | Backward-compatible change: a new domain/companion added behind the existing contract, a bug fix, a doc fix, a new optional endpoint/field. |

We cut `1.0.0` when the domain/companion contract and `/ask` response shape
were frozen and documented as stable. Subsequent work uses `1.0.x` (PATCH) or
`1.x.0` (MINOR) per the table above.

## Source of truth

- `VERSION` — the single file holding the current version string (`1.0.0`).
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
6. **Never re-point `v1.0.0`.** `v1.0.0` is the frozen, contract-locked stable
   baseline. Do NOT run `git tag -f v1.0.0`, do NOT `git push --force` the tag,
   and do NOT delete-and-recreate its GitHub release to absorb new work. Any
   change after 1.0.0 — even a one-line fix or a doc update — ships as a NEW
   incremental version (`v1.0.1`, `v1.1.0`, …) per the table above. Treat
   `v1.0.0` as immutable history.

> **Why this is hard:** re-pointing `v1.0.0` silently rewrites what
> downstream users pinned to `v1.0.0` resolve to, breaks reproducibility, and
> defeats the contract-lock. The cost of a new PATCH tag is one extra git tag;
> the cost of re-pointing is loss of a stable reference. Always cut forward.

## What is frozen at 1.0

- `domain_config.yaml` schema (adding a domain = additive, safe).
- `/ask` and `/ingest` request/response contracts.
- The companion `companions:` + `_signal` + dual-evidence behavior.
- The `/embed_late` contract (`{text, doc_id, strategy, chunk_size, overlap}`).
