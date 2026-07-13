# Archived TOML Profiles (V2.x)

These domain profiles were used by the v2.x pipeline (`serve_gpu.py` loaded
`domains/<name>.toml` per request). V3.0 replaces them with a single
`domain_config.yaml` consumed by `domain_loader.py`.

Migration notes:
- `entity_types` → `domain_config.yaml` `entity_types`
- `edge_types`   → `domain_config.yaml` `relation_types`
- `extraction.prompt` template → replaced by Index-Routing JSON schema in `index_router.py`
- `chunking.*`  → `domain_config.yaml` `chunking.*`
- `metadata_schema.fields` → `domain_config.yaml` `metadata_schema.fields`

Do NOT delete — kept for reference and rollback.
