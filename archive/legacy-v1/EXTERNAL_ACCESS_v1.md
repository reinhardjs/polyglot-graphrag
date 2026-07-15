# External Access via Tailscale (LEGACY v1)

> [!WARNING]
> **Legacy v1 document.** The live RAG API is now `v2/serve_gpu.py` on `:8000`
> (`POST /ask`). The Gemma endpoints changed: **synthesis is Gemma 4 E4B on
> `:8084`** and **extraction is Gemma 4 E2B on `:8082`** (was Gemma 4 12B on
> `:8083`). `api_server.py` / `retrieve.py` no longer exist. For current docs
> see `docs/`. Kept for historical rationale.

How external agents (Hermes, Claude Code, Pi, custom scripts) can query
the RAG system from another machine on the same Tailscale network.

---

## 1. Network Setup

This machine (RAG server) is connected to Tailscale. Find its Tailscale IP:

```bash
tailscale ip -4
# Example output: 100.86.53.21
```

External machines reach the RAG system at `http://<tailscale-ip>:<port>`.

**Services exposed on Tailscale:**

| Service      | Port | Protocol | Purpose |
|-------------|------|----------|---------|
| Gemma API (E4B) | 8084 | HTTP | OpenAI-compatible chat (answer synthesis, direct) |
| Qdrant      | 6333 | HTTP     | Vector search (read chunks) |
| Qdrant gRPC | 6334 | gRPC     | Vector search (high perf) |
| Neo4j Bolt  | 7687 | Bolt     | Graph queries |
| Neo4j HTTP  | 7474 | HTTP     | Neo4j browser UI |

By default these bind to `0.0.0.0`, so they're already reachable over
Tailscale without additional config.

---

## 2. Query the RAG System (Recommended: HTTP API)

The simplest way for external agents to ask questions is via the OpenAI-
compatible endpoint on Gemma. But for proper RAG (retrieve + generate),
set up a lightweight query server on the RAG machine.

### Option A: Direct Gemma Chat (No RAG context)

Quick answer without vector/graph retrieval — Gemma uses its own knowledge.

```bash
curl http://100.86.53.21:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-QAT-Q4_0.gguf",
    "messages": [{"role": "user", "content": "what is BUG-204 about?"}],
    "max_tokens": 512
  }'
```

**Limitation:** No context from your ingested docs. Gemma only knows what
it was trained on.

### Option B: Full RAG Query Server (Recommended)

Run a persistent HTTP server on the RAG machine that exposes a `/ask`
endpoint. This loads the 3 models once and runs the full pipeline
(embed → cache → route → search → rerank → synthesize).

**File:** `<project-root>/api_server.py`

```python
#!/usr/bin/env python3
"""
Lightweight HTTP API for the GraphRAG system.
Intended for external agents over Tailscale.
Usage: python api_server.py [--port 9090]
"""
from __future__ import annotations
import os, sys, json
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ.setdefault("HF_HOME", "<hf-cache>")
sys.path.insert(0, "<project-root>")

from retrieve import QueryEmbedder, answer, Reranker
from router import SemanticRouter

# Load models once at startup
print("Loading RAG models...", flush=True)
embedder = QueryEmbedder()
router = SemanticRouter()
reranker = Reranker()
print("Ready.", flush=True)


class RAGHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        query = body.get("query", "")
        result = answer(embedder, router, reranker, query)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, indent=2).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","chunks":51,"entities":37}')
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[api] {args[0]} {args[1]} {args[2]}", flush=True)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
    server = HTTPServer(("0.0.0.0", port), RAGHandler)
    print(f"RAG API on http://0.0.0.0:{port}/ask", flush=True)
    server.serve_forever()
```

**Run it (on the RAG machine):**

```bash
cd <project-root>
<legacy-venv>/bin/python api_server.py --port 9090 &
```

**Query from external agent (any machine on Tailscale):**

```bash
curl http://100.86.53.21:9090/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "who reported BUG-204 and what severity was it?"}'
```

Response:
```json
{
  "path": "vector",
  "source": "llm",
  "answer": "BUG-204 was reported by bob and had a severity of SEV-2.",
  "ctx": ["..."],
  "path_score": 0.42
}
```

---

## 3. Agent Integration Examples

### Hermes Agent (from another machine)

```python
# In Hermes config or skill:
import requests
RAG_HOST = "http://100.86.53.21:9090"

def rag_query(query: str) -> str:
    resp = requests.post(f"{RAG_HOST}/ask",
        json={"query": query}, timeout=60)
    return resp.json()["answer"]
```

### Pi Agent (via Pi's tools/scripts)

```bash
# Pi can call the API directly with curl
QRY=$(echo "SELECT * FROM knowledge WHERE question ILIKE '%$1%';")
curl -s http://100.86.53.21:8090/ask -d "{\"query\":\"$1\"}" | jq -r '.answer'
```

### Any OpenAI-compatible client

Since Gemma speaks the OpenAI API, you can use any OpenAI client library
pointed at the Tailscale IP:

```python
from openai import OpenAI
client = OpenAI(base_url="http://100.86.53.21:8084/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="gemma-4-E4B-it-QAT-Q4_0.gguf",
    messages=[{"role": "user", "content": "what is ADR-014 about?"}],
)
print(resp.choices[0].message.content)
```

**Note:** This bypasses the RAG pipeline (no vector/graph lookup). Use
the `/ask` endpoint instead for context-augmented answers.

---

## 4. Security Notes

- Tailscale encrypts traffic end-to-end (WireGuard). No TLS needed within
  the Tailnet.
- Bind the API server to `0.0.0.0` so it's reachable over Tailscale.
- The RAG API has no auth. Tailscale ACLs control who reaches the ports.
- Default Tailscale ACL: only machines in your tailnet can connect.
- To restrict: add ACL rules in the Tailscale admin console.

---

## 5. Systemd Service (Auto-start API Server)

**File:** `/etc/systemd/system/rag-api.service`

```
[Unit]
Description=GraphRAG External Query API
After=gemma-4-e4b.service network.target
Wants=gemma-4-e4b.service

[Service]
Type=simple
User=reinhard
Environment=HF_HOME=<hf-cache>
ExecStart=<legacy-venv>/bin/python \
  <project-root>/api_server.py --port 9090
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl start rag-api.service
sudo systemctl enable rag-api.service   # auto-start on boot
```

---

## 6. Quick Test

From any machine on your Tailscale:

```bash
# Test health
curl http://100.86.53.21:9090/health

# Test query
curl http://100.86.53.21:9090/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what is ADR-014 about?"}'

# Test direct Gemma (E4B synthesis, no RAG)
curl http://100.86.53.21:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-E4B-it-QAT-Q4_0.gguf","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

---

*Generated: July 2026*
