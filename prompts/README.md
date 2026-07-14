# prompts/ — synthesis prompt templates

Each synthesis style lives in its OWN file: `prompts/<kind>.md`. The daemon
selects one via a domain's `synthesis.kind` in `domain_config.yaml` — there
is NO per-domain `if` branch in `prompts.py`. To add a new synthesis style:

1. write `prompts/<newkind>.md` containing the two substitution tokens
   `{query}` and `{context}` (verbatim; rendered with `.replace()`, so any
   other `{...}` in the text is safe — e.g. JSON examples won't break it).
2. set `synthesis: {kind: <newkind>}` in the domain's `domain_config.yaml`
   block.
3. done — `/ask` picks it up; no code change in `prompts.py` / `serve_gpu.py`.

## Resolution order (prompts.build_synthesis_prompt)
1. `profile['synthesis']['prompt']` — inline template already in YAML, used
   verbatim if it contains BOTH `{query}` and `{context}`.
2. `prompts/<kind>.md` — file template, `{query}`/`{context}` substituted.
3. generic fallback — "Answer using ONLY the context. Cite source numbers…"
   (also used if `kind` is unknown / file missing — a warning is logged).

## Files
- `clinical_dx.md` — ranked differential-diagnosis prompt for the `snomed`
  domain (SNOMED term-match + clinical-prose corpus). Selected by
  `domains.snomed.synthesis.kind: clinical_dx`.
