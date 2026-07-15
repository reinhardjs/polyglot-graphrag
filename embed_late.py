"""embed_late.py — shared late-chunking embed helper (in-process).

Both the /embed_late HTTP endpoint (serve_gpu.py) and the ingest pipeline
(ingest.py::ingest_text) call this so ingestion does NOT do an HTTP loopback
to the daemon's own /embed_late endpoint (which starves the daemon's own
threadpool under large-doc load and intermittently 500s).

The embed model is supplied by the caller (serve_gpu._jina) to avoid a circular
import; callers must hold the _jina_lock while encoding.
"""
import numpy as np


def embed_late_chunks(model, text, strategy="sentence", chunk_size=512,
                      overlap=64, header_prefix="##", task=None,
                      batch=64):
    """Chunk `text` and embed in batches. Returns list of
    {"chunk_idx", "vector", "text"} dicts (vector as list[float])."""
    import chunking as CH
    chunks = CH.chunk_text(text, strategy=strategy, chunk_size=chunk_size,
                           overlap=overlap, header_prefix=header_prefix)
    if not chunks:
        chunks = [text.strip()]
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if task:
        encode_kw["task"] = task
    all_vecs = []
    for i in range(0, len(chunks), batch):
        v = model.encode(chunks[i:i + batch], **encode_kw)
        all_vecs.append(v)
    if len(all_vecs) == 1:
        vecs = all_vecs[0]
    else:
        vecs = np.concatenate(all_vecs, axis=0)
    return [{"chunk_idx": i, "vector": vecs[i].tolist(), "text": chunks[i]}
            for i in range(len(chunks))]
