# prompts/ — synthesis + extraction prompt templates

## Synthesis (answer generation)
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

## Extraction (knowledge-graph extraction)
The default extraction prompt lives in `prompts/extraction.md` (editable
without touching Python — `config.EXTRACTION_PROMPT` loads it at import, with
an inline fallback if the file is missing). Substitution tokens are
`{doc_id}` and `{text}` (rendered with `.replace()`, so literal JSON `{...}`
in the prompt is safe).

A domain may override the extraction prompt two ways (in `domain_config.yaml`):
- `extraction.prompt` — inline template string (highest precedence).
- `extraction.template` — filename of a `prompts/<name>.md` template (e.g.
  `legal_extraction.md`), resolved relative to `prompts/`. Used when no inline
  `prompt` is given.

Substitution tokens in ANY extraction template (default, template, or inline):
- `{doc_id}`, `{text}` — document id / body (always substituted).
- `{entity_types}`, `{relation_types}` — the domain's vocabulary, taken from
  `domain_config.yaml` `entity_types` / `relation_types` and substituted by
  `ingest.py`. **Do NOT hardcode the vocab in the template** — the YAML is the
  single source of truth, so the LLM prompt and the structured extractor/
  validation stay in sync automatically. If a domain omits them, the
  `config.DEFAULT_ENTITY_TYPES` / `DEFAULT_RELATION_TYPES` fallbacks are used.

## Files
- `clinical_dx.md` — ranked differential-diagnosis prompt for the `snomed`
  domain (SNOMED term-match + clinical-prose corpus). Selected by
  `domains.snomed.synthesis.kind: clinical_dx`.
- `extraction.md` — default engineering knowledge-graph extraction prompt
  (loaded by `config.EXTRACTION_PROMPT`).
