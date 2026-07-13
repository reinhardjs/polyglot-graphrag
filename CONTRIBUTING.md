# Contributing to Polyglot GraphRAG

Thanks for your interest! This guide covers how to set up your environment,
make changes, and submit a pull request.

## Setup

```bash
# 1. Clone + build the Python environment
git clone https://github.com/reinhardjs/polyglot-graphrag.git
cd polyglot-graphrag

python3.11 -m venv /mnt/data-970-plus/rag-env
/mnt/data-970-plus/rag-env/bin/pip install --upgrade pip

# 2. Install CUDA torch (RTX 3060 / CUDA 12.1)
/mnt/data-970-plus/rag-env/bin/pip install torch==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 3. Install project deps
/mnt/data-970-plus/rag-env/bin/pip install fastapi uvicorn qdrant-client neo4j \
  openai sentence-transformers gliner einops requests numpy pytest transformers==4.49.0

# 4. Set HF cache (avoids re-downloading models)
export HF_HOME=/mnt/data-970-plus/hf_cache
mkdir -p "$HF_HOME"

# 5. Start databases (Qdrant + Neo4j)
docker compose up -d

# 6. Start the GPU daemon + LLMs
sudo systemctl start gemma-4-e2b.service gemma-4-e4b.service rag-gpu-daemon.service
#   — or:  cd v2 && bash run.sh serve
```

## Running the tests

```bash
cd v2

# Unit tests (no GPU / daemon needed — pure logic)
/mnt/data-970-plus/rag-env/bin/python -m pytest tests/test_chunking.py \
    tests/test_prompts.py tests/test_metadata.py tests/test_condense.py \
    tests/test_extraction_prompt.py tests/test_neo4j_entry.py -q

# End-to-end tests (need the daemon live on :8000 with sample data ingested)
/mnt/data-970-plus/rag-env/bin/python -m pytest tests/test_e2e_chunking.py -q

# Everything
/mnt/data-970-plus/rag-env/bin/python -m pytest tests/ -q
```

## Adding a new domain

1. Copy an existing profile from `v2/domains/` (e.g. `engineering.toml`).
2. Set `name`, `collection`, `neo4j_label` — all unique.
3. Tune `[chunking]`, `[extraction]`, `[synthesis]`, `[metadata_schema]`,
   `[neo4j_entry]` for your domain's vocabulary and retrieval needs.
4. No code changes or daemon restart needed — the profile is loaded on
   the next API call that passes `domain=<name>`.

Full schema docs: [docs/domains/README.md](docs/domains/README.md).

## Project structure

```
v2/
├── serve_gpu.py          # Primary FastAPI daemon (:8000)
├── serve_cpu.py          # CPU fallback daemon (full API parity)
├── config.py             # All constants, load_domain_profile()
├── ingest.py             # Document ingestion pipeline
├── ask.py                # Shared retrieval library + CLI client
├── chunking.py           # Pluggable chunking strategies
├── prompts.py            # Shared synthesis-prompt builder
├── retrieve_json.py      # Headless retrieval → JSON
├── bench_rag.py          # Latency benchmark
├── run.sh                # Orchestrator
├── domains/              # Domain profiles (*.toml)
├── tests/                # pytest suite
└── sample_data/          # 5 engineering docs
```

## Coding conventions

- **Imports.** Daemon endpoints import modules inside the function (lazy, avoids
  loading heavy deps at startup). Library modules (`ask.py`, `chunking.py`,
  `prompts.py`) import at the top.
- **Prompts.** Use `.replace("{token}", value)` for prompt templates — NOT
  `str.format()`. Domain profiles contain literal `{...}` JSON that breaks
  `.format()`.
- **Config.** Every model-specific behaviour is a flag in `config.py`. No
  hardcoded model names or paths outside of `config.py` and `v2/domains/`.
- **GPU safety.** Never call `.half()` on CPU models (x86 has no native fp16).
  `serve_cpu.py` enforces fp32.
- **Tests.** Unit tests must not require GPU / Neo4j / the daemon. Mock
  external dependencies. E2E tests hit the live `:8000` daemon.

## Pull request process

1. Fork the repo and create a feature branch.
2. Run `pytest tests/ -q` — ensure all 33 tests pass.
3. If you added a feature, add tests covering it.
4. Update the relevant docs if your change affects documented behaviour.
5. Open a PR against `main` with a clear description.

## Commit Message Convention (Conventional Commits)

This repo uses **release-please**, which derives the version number
**automatically from commit messages**. All commits to `main` MUST follow the
[Conventional Commits](https://www.conventionalcommits.org/) spec, or releases
will not bump correctly.

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types → version impact:**

| Prefix | Effect on version |
|--------|------------------|
| `feat:` | Minor bump (e.g. `3.1.6-beta.1` → `3.1.6-beta.2` until stable `3.2.0`) |
| `fix:` | Patch bump (e.g. `3.1.6-beta.1` → `3.1.6-beta.2`) |
| `feat!:` / `fix!:` / `BREAKING CHANGE:` | Major bump (e.g. `3.1.6-beta.1` → `4.0.0`) |
| `docs:`, `chore:`, `refactor:`, `test:`, `ci:` | No release (changelog only) |

**Examples:**
```
feat(extraction): add streaming mode to serve_gpu
fix(label-provider): evict LRU on TTL expiry
docs: refresh README repository layout
refactor(ingest): split late-chunk vector builder
feat!: change Neo4j schema labels (breaking)
```

After you merge to `main`, the release-please workflow opens/updates a release
PR. **Merge that PR** to cut the Git tag + GitHub Release. Do not create tags
by hand. See `README.md` → "Releases & Versioning" for the full flow.

## Questions?

Open an issue on the repository. Include your environment details (GPU, VRAM,
Python version) and the relevant log output if something isn't working.
