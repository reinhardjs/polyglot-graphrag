#!/usr/bin/env python3
"""sync_docs.py — git-style file-change sync daemon for the GraphRAG system.

Keeps the Neo4j knowledge graph + Qdrant vector store consistent with a
directory of source documents, detecting changes the way git detects file
changes: by **content hash (SHA256)**, never by mtime.

Pipeline mapping
----------------
    file on disk ──(relative path)──▶ doc_id
    doc_id        ──POST /ingest──▶ chunks → Qdrant + entities/edges → Neo4j

Per run, every file under the watched root is classified as:

    NEW        present on disk, absent from state       → POST /ingest (create)
    CHANGED    checksum differs from stored             → POST /ingest (replace,
                                                          pass NEW checksum as
                                                          if_checksum so the server
                                                          skips ONLY if it already
                                                          has this exact content)
    UNCHANGED  checksum identical                       → skip (or send if_checksum
                                                          for a cheap server-side
                                                          confirmation pass)
    DELETED    present in state, absent on disk         → DELETE /ingest/{doc_id}

State (``.sync_state.json``) records, per relative path:
    {"checksum": "<sha256>", "doc_id": "...", "domain": "...",
     "collection": "..."}
It is only mutated AFTER a confirmed-success server response, so a crash or
daemon outage never desyncs state from reality. The state file is protected by
an ``fcntl`` advisory lock for safe concurrent use (e.g. two ``--watch``
processes, or a cron one-shot racing a daemon).

doc_id scheme
-------------
By default ``doc_id`` = the file's path **relative to the watched root**
(e.g. ``eng/runbooks/service-catalog.md``). This is stable across edits
(renames are treated as delete+create) and is globally unique because it
includes the path — important: ``doc_id`` is *not* namespaced by domain on the
server, so two files in different domain folders that happened to share a bare
basename would collide. Relative-path ids avoid that entirely.

Overrides (via ``sync_config.yaml`` or ``--config``):
    doc_id_map    : { "rel/path.md": "custom-doc-id" }
    domain_map    : { "rel/path.md": "legal" }
    daemon_url    : default base URL of the GPU ingestion daemon

Domain inference (path-based, overridable):
    eng/ engineering/        → engineering
    journal/ papers/ *.pdf   → journal
    legal/                   → legal
    anything else            → engineering (server default)

Multi-domain correctness note
-----------------------------
The server's ``DELETE /ingest/{doc_id}`` endpoint defaults its ``collection``
query param to ``engineering_chunks``. For non-engineering docs the delete would
otherwise 404 (the doc lives in a different Qdrant collection). We therefore
remember each doc's collection in the state file and pass ``?collection=`` on
delete. Same care is taken on ingest (``domain`` is sent so the server resolves
the right collection/server profile).

Exit codes
----------
    0  success (sync complete; --watch runs forever)
    1  usage / config error
    2  daemon unreachable (fail-closed, never marks files as synced)
    3  partial failure (some docs failed; state updated only for the
       successes, so a re-run retries the failures)
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DAEMON_URL = os.environ.get("RAG_DAEMON_URL", "http://127.0.0.1:8000")
DEFAULT_STATE_FILE = ".sync_state.json"
DEFAULT_CONFIG_FILE = "sync_config.yaml"
SYNCIGNORE_FILE = ".syncignore"

LOG = logging.getLogger("sync_docs")

# Domain inference rules, evaluated in order. Each is (path-marker, domain).
# *.pdf is a special filename rule handled separately (only when matched).
_DOMAIN_FOLDER_RULES = [
    ("eng/", "engineering"),
    ("engineering/", "engineering"),
    ("journal/", "journal"),
    ("papers/", "journal"),
    ("legal/", "legal"),
    ("medical/", "medical"),
    ("accounting/", "accounting"),
    ("hospitality/", "hospitality"),
]
_DEFAULT_DOMAIN = "engineering"

# File extensions we can ingest directly as UTF-8 text.
_TEXT_EXTS = {".md", ".txt", ".markdown", ".rst", ".json", ".csv", ".log", ".yaml", ".yml"}
# PDFs are handled via pdftotext.
_PDF_EXT = ".pdf"


# ── Logging helpers ──────────────────────────────────────────────────────────
class _ColorLog:
    """Minimal ANSI color wrapper so terminal runs are scannable."""

    RESET = "\033[0m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"

    @classmethod
    def _wrap(cls, color: str, msg: str) -> str:
        if sys.stdout.isatty():
            return f"{color}{msg}{cls.RESET}"
        return msg

    @classmethod
    def info(cls, msg: str) -> None:
        print(cls._wrap(cls.DIM, f"[sync] {msg}"))

    @classmethod
    def ok(cls, msg: str) -> None:
        print(cls._wrap(cls.GREEN, f"[sync] {msg}"))

    @classmethod
    def warn(cls, msg: str) -> None:
        print(cls._wrap(cls.YELLOW, f"[sync] WARNING: {msg}"))

    @classmethod
    def err(cls, msg: str) -> None:
        print(cls._wrap(cls.RED, f"[sync] ERROR: {msg}"), file=sys.stderr)

    @classmethod
    def act(cls, msg: str) -> None:
        print(cls._wrap(cls.CYAN, f"[sync] {msg}"))


# ── Config loading ───────────────────────────────────────────────────────────
def load_config(config_path: Optional[str]) -> dict:
    """Load optional YAML config (doc_id_map / domain_map / daemon_url).

    Missing file is fine — all keys are optional. Returns a normalized dict with
    at least the keys we look up, so callers never KeyError.
    """
    cfg: dict = {"doc_id_map": {}, "domain_map": {}, "daemon_url": None}
    if not config_path:
        # Try the default path next to the script.
        default = Path(DEFAULT_CONFIG_FILE)
        if not default.exists():
            return cfg
        config_path = str(default)
    if not Path(config_path).exists():
        return cfg
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            _ColorLog.warn(f"config {config_path} is not a mapping; ignoring")
            return cfg
        cfg["doc_id_map"] = raw.get("doc_id_map", {}) or {}
        cfg["domain_map"] = raw.get("domain_map", {}) or {}
        cfg["daemon_url"] = raw.get("daemon_url")
    except Exception as e:  # pragma: no cover - defensive
        _ColorLog.warn(f"could not parse config {config_path}: {e}; using defaults")
    return cfg


# ── .syncignore (gitignore-flavored) ─────────────────────────────────────────
def load_syncignore(root: Path) -> List[str]:
    """Read a .syncignore file (gitignore syntax subset) into a pattern list.

    Supported: ``*`` wildcards, ``/`` anchored patterns, blank lines and ``#``
    comments. We use pathlib's ``match`` for trailing/``*`` patterns and a
    simple contains-check for directory-prefix patterns (``dir/``).
    """
    patterns: List[str] = []
    path = root / SYNCIGNORE_FILE
    if not path.exists():
        return patterns
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def is_ignored(rel_path: str, patterns: List[str]) -> bool:
    """Return True if ``rel_path`` (posix, relative) matches any ignore rule."""
    p = Path(rel_path)
    for pat in patterns:
        # Directory-prefix rule, e.g. "drafts/" or "_drafts/" → ignore anything
        # under that directory.
        if pat.endswith("/"):
            prefix = pat.rstrip("/")
            if rel_path == prefix or rel_path.startswith(prefix + "/"):
                return True
        # Trailing wildcard dir, e.g. "build/*" → ignore files directly inside.
        if pat.endswith("/*"):
            base = pat[:-2]
            if rel_path.startswith(base + "/"):
                return True
        # pathlib match handles *, ?, **, and trailing components.
        if p.match(pat) or p.match(pat.lstrip("/")):
            return True
    return False


# ── Domain + doc_id resolution ──────────────────────────────────────────────
def infer_domain(rel_path: str, domain_map: dict) -> str:
    """Infer the ingestion domain from the relative path (overridable)."""
    if rel_path in domain_map:
        return domain_map[rel_path]
    low = rel_path.lower()
    if low.endswith(_PDF_EXT):
        return "journal"
    for marker, domain in _DOMAIN_FOLDER_RULES:
        if marker in low:  # substring covers eng/, journal/, etc.
            return domain
    return _DEFAULT_DOMAIN


def resolve_doc_id(rel_path: str, doc_id_map: dict) -> str:
    """doc_id = relative path by default, or an explicit override."""
    return doc_id_map.get(rel_path, rel_path)


# ── Text extraction ──────────────────────────────────────────────────────────
def extract_text(path: Path) -> Optional[str]:
    """Return UTF-8 text for a file, or None if it must be skipped.

    Skips:
      - empty files (0 bytes): ingest would get a 400 from the server anyway.
      - binary files we can't decode as UTF-8.
    PDFs are converted with ``pdftotext`` (poppler) when available.
    """
    if path.stat().st_size == 0:
        _ColorLog.warn(f"{path} is empty (0 bytes) — skipping")
        return None
    suffix = path.suffix.lower()
    if suffix == _PDF_EXT:
        return _extract_pdf(path)
    if suffix in _TEXT_EXTS:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _ColorLog.warn(f"{path} is not valid UTF-8 — skipping (likely binary)")
            return None
    # Unknown extension: try as UTF-8 text, skip if it fails.
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, UnicodeError):
        _ColorLog.warn(f"{path} has unknown/non-UTF-8 type — skipping")
        return None


def _extract_pdf(path: Path) -> Optional[str]:
    """Run poppler's pdftotext to get plain text from a PDF."""
    import shutil
    import subprocess

    if shutil.which("pdftotext") is None:
        _ColorLog.err("pdftotext not found; cannot ingest PDF %s" % path)
        return None
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, check=True, timeout=120,
        )
        return out.stdout
    except subprocess.CalledProcessError as e:
        _ColorLog.err(f"pdftotext failed on {path}: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        _ColorLog.err(f"pdftotext timed out on {path}")
        return None


def sha256_of(text: str) -> str:
    """SHA256 of the exact UTF-8 bytes that will be sent to the server.

    The server computes ``sha256(text.encode('utf-8'))`` on the text it
    receives, so we MUST hash the same string we POST. Do not hash the file
    bytes — if extraction ever changed encoding the checksums would diverge.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── State file (with fcntl locking) ─────────────────────────────────────────
class StateStore:
    """Read/write ``<root>/.sync_state.json`` with an advisory lock.

    The lock is held only for the duration of a load→mutate→save round-trip.
    Callers do the network work *outside* the lock, then ``save`` the updated
    state under the lock. This keeps state consistent across concurrent runs
    without serializing the (slow) ingestion calls.
    """

    def __init__(self, root: Path, state_file: str):
        self.path = root / state_file
        self._data: Dict[str, dict] = {}
        self._lock_fd = None

    # Context manager API — acquires the lock on enter.
    def __enter__(self):
        self._open_lock()
        self._load_under_lock()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close_lock()
        return False

    def _open_lock(self):
        self._lock_fd = os.open(self.path.parent / (self.path.name + ".lock"),
                                os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _close_lock(self):
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def _load_under_lock(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                _ColorLog.warn(f"state file corrupt ({e}); starting fresh")
                self._data = {}
        else:
            self._data = {}

    def get(self, rel_path: str) -> Optional[dict]:
        return self._data.get(rel_path)

    def all(self) -> Dict[str, dict]:
        return self._data

    def update(self, rel_path: str, record: dict) -> None:
        self._data[rel_path] = record

    def remove(self, rel_path: str) -> None:
        self._data.pop(rel_path, None)

    def save(self) -> None:
        """Write state atomically (temp file + rename) under the held lock."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.path)


# ── Daemon client ───────────────────────────────────────────────────────────
class DaemonClient:
    """Thin wrapper over the GPU daemon's ingest/delete/status endpoints."""

    def __init__(self, base_url: str,
                 timeout: int = int(os.environ.get("SYNC_INGEST_TIMEOUT", "600"))):
        """Per-ingest timeout (seconds). Large files (e.g. 89 KB logs under
        sliding_window extraction) can exceed the old 120 s default; raise via
        SYNC_INGEST_TIMEOUT if you see 'did not finish' timeouts."""
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def healthcheck(self) -> bool:
        """Fail-closed: if the daemon isn't healthy, we refuse to proceed."""
        try:
            r = self.session.get(f"{self.base}/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def collection_for_domain(self, domain: str) -> Optional[str]:
        """Best-effort resolve the Qdrant collection for a domain via the
        server's /profiles endpoint, so DELETE targets the right collection.

        Returns None if it can't be determined — caller then relies on the
        stored collection, or falls back to the server default.
        """
        try:
            r = self.session.get(f"{self.base}/profiles", timeout=5)
            if r.status_code == 200:
                for p in r.json().get("profiles", []):
                    if p.get("domain") == domain:
                        return p.get("collection")
        except requests.RequestException:
            pass
        return None

    def ingest(self, doc_id: str, text: str, domain: str,
               collection: Optional[str], if_checksum: Optional[str]):
        """POST /ingest and poll to completion.

        Returns (status, checksum_or_None). ``status`` is one of
        "accepted" (ingested), "unchanged" (skipped via if_checksum).
        Raises on transport/HTTP errors.
        """
        body = {"doc_id": doc_id, "text": text, "domain": domain}
        if if_checksum is not None:
            body["if_checksum"] = if_checksum
        if collection is not None:
            body["collection"] = collection

        r = self.session.post(f"{self.base}/ingest", json=body,
                              timeout=self.timeout)
        if r.status_code == 304:
            # Server confirmed checksum matches → nothing to do.
            cs = (r.json().get("checksum") if r.content else None) or if_checksum
            return "unchanged", cs
        if r.status_code == 202:
            task_id = r.json().get("task_id")
            return self._poll(task_id), if_checksum
        # 400 empty text / other — surface as failure.
        r.raise_for_status()
        return "accepted", if_checksum

    def _poll(self, task_id: str) -> str:
        deadline = time.time() + self.timeout
        last_status = "queued"
        while time.time() < deadline:
            r = self.session.get(f"{self.base}/ingest/status/{task_id}",
                                 timeout=10)
            if r.status_code == 200:
                info = r.json()
                last_status = info.get("status", last_status)
                if last_status == "done":
                    return "accepted"
                if last_status == "error":
                    raise RuntimeError(
                        f"ingest task {task_id} failed: {info.get('error')}")
            time.sleep(0.5)
        raise TimeoutError(f"ingest task {task_id} did not finish in "
                           f"{self.timeout}s (last status: {last_status})")

    def delete(self, doc_id: str, collection: Optional[str]) -> dict:
        """DELETE /ingest/{doc_id} (optionally with ?collection=)."""
        url = f"{self.base}/ingest/{doc_id}"
        params = {}
        if collection is not None:
            params["collection"] = collection
        r = self.session.delete(url, params=params, timeout=self.timeout)
        if r.status_code == 404:
            # Already gone — treat as success for our bookkeeping.
            return {"status": "absent", "vectors_deleted": 0, "nodes_cleaned": 0}
        r.raise_for_status()
        return r.json()


# ── Walk the directory ──────────────────────────────────────────────────────
def scan_files(root: Path, ignore_patterns: List[str],
               state_file: str) -> List[Path]:
    """Recursively collect ingestable files under root, honoring .syncignore.

    The script's own state artifacts (``.sync_state.json``, its ``.lock`` and
    ``.tmp`` siblings) are ALWAYS excluded — we must never ingest our own
    bookkeeping file into the knowledge base.
    """
    state_names = {state_file, state_file + ".lock", state_file + ".tmp",
                   ".syncignore"}
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in state_names:
            continue
        rel = p.relative_to(root).as_posix()
        if is_ignored(rel, ignore_patterns):
            continue
        out.append(p)
    return out


# ── Core sync ───────────────────────────────────────────────────────────────
def run_once(root: Path, daemon: DaemonClient, config: dict,
             state_file: str, dry_run: bool = False) -> int:
    """Perform a single git-style sync pass. Returns a process exit code."""
    ignore = load_syncignore(root)
    files = scan_files(root, ignore, state_file)

    # Build a map rel_path -> Path
    on_disk: Dict[str, Path] = {p.relative_to(root).as_posix(): p for p in files}

    failure = False
    with StateStore(root, state_file) as state:
        state_data = state.all()
        # 1. DELETED: in state, not on disk.
        for rel_path in list(state_data.keys()):
            if rel_path not in on_disk:
                rec = state_data[rel_path]
                doc_id = rec.get("doc_id", rel_path)
                collection = rec.get("collection")
                _ColorLog.act(f"DELETE {rel_path} (doc_id={doc_id})")
                if dry_run:
                    continue
                try:
                    res = daemon.delete(doc_id, collection)
                    _ColorLog.ok(f"  removed from stores: {res}")
                    state.remove(rel_path)
                    state.save()  # persist incremental progress
                except requests.HTTPError as e:
                    failure = True
                    _ColorLog.err(f"  delete failed for {doc_id}: {e}")
                except Exception as e:
                    failure = True
                    _ColorLog.err(f"  delete error for {doc_id}: {e}")

        # 2. NEW / CHANGED / UNCHANGED: present on disk.
        for rel_path, path in on_disk.items():
            text = extract_text(path)
            if text is None:
                continue  # empty/binary already warned; skip
            checksum = sha256_of(text)
            domain = infer_domain(rel_path, config["domain_map"])
            doc_id = resolve_doc_id(rel_path, config["doc_id_map"])
            prev = state.get(rel_path)

            if prev and prev.get("checksum") == checksum:
                # UNCHANGED — nothing to do, state already correct.
                _ColorLog.info(f"UNCHANGED {rel_path} ({domain})")
                continue

            if prev is None:
                action = "NEW"
                if_checksum = None
            else:
                action = "CHANGED"
                # CRITICAL: pass the NEW checksum as if_checksum, NOT the old
                # one. The server's if_checksum guard compares the passed value
                # against the *stored* checksum and returns 304 (skip) on match.
                # Sending the OLD checksum would match the stored old content and
                # make the server SKIP re-ingestion — silently dropping the edit.
                # Sending the NEW checksum makes the server skip ONLY if it
                # already has this exact new content (it doesn't) → it ingests.
                if_checksum = checksum

            _ColorLog.act(
                f"{action} {rel_path} → doc_id={doc_id} domain={domain}")
            if dry_run:
                continue

            # Resolve collection: prefer the server profile for the domain; the
            # stored collection (for updates) is also acceptable. We send
            # `domain` and let the server pick; we record whatever it used by
            # reading it back from /profiles so DELETE targets the right place.
            collection = daemon.collection_for_domain(domain)
            if collection is None and prev:
                collection = prev.get("collection")

            prev_coll = (prev or {}).get("collection")
            try:
                status, returned_cs = daemon.ingest(
                    doc_id, text, domain, collection, if_checksum)
                if status == "unchanged":
                    _ColorLog.ok(f"  server confirmed unchanged (checksum match)")
                    # State already has the right checksum.
                    state.update(rel_path, {
                        "checksum": (prev or {}).get("checksum", checksum),
                        "doc_id": doc_id, "domain": domain,
                        "collection": collection or prev_coll,
                    })
                else:
                    _ColorLog.ok(f"  ingested ({len(text)} chars)")
                    state.update(rel_path, {
                        "checksum": checksum,
                        "doc_id": doc_id, "domain": domain,
                        "collection": collection or prev_coll,
                    })
                # Persist incrementally so a killed run (e.g. background
                # timeout) still leaves progress on disk for the next resume.
                state.save()
            except requests.HTTPError as e:
                failure = True
                _ColorLog.err(f"  ingest failed for {doc_id}: {e}")
            except Exception as e:
                failure = True
                _ColorLog.err(f"  ingest error for {doc_id}: {e}")

        # Persist state (only reached records changed).
        if not dry_run:
            state.save()

    return 3 if failure else 0


# ── Watch mode (real-time via watchdog) ─────────────────────────────────────
def run_watch(root: Path, daemon: DaemonClient, config: dict,
              state_file: str, debounce: float = 1.0) -> int:
    """Watch the root with inotify and re-sync on any change.

    A short debounce coalesces bursts (e.g. a save storm from an editor) into a
    single sync pass. Runs until interrupted (Ctrl-C / SIGTERM).
    """
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class _Handler(FileSystemEventHandler):
        def __init__(self):
            self._pending = False
            self._last = 0.0

        def on_any_event(self, event):
            # Ignore our own lock/temp state files.
            name = getattr(event, "dest_path", "") or getattr(event, "src_path", "")
            if name and (name.endswith(".lock") or ".sync_state" in name):
                return
            self._pending = True
            self._last = time.time()

    handler = _Handler()
    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()
    _ColorLog.ok(f"Watching {root} (Ctrl-C to stop)…")

    try:
        while True:
            time.sleep(0.5)
            if handler._pending and (time.time() - handler._last) >= debounce:
                handler._pending = False
                _ColorLog.act("change detected — syncing…")
                try:
                    run_once(root, daemon, config, state_file)
                except Exception as e:
                    _ColorLog.err(f"sync pass failed: {e}")
    except KeyboardInterrupt:
        _ColorLog.info("stopping watcher…")
    finally:
        observer.stop()
        observer.join()
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="git-style file→GraphRAG sync daemon "
                    "(SHA256 change detection, Neo4j + Qdrant).")
    ap.add_argument("root", nargs="?", default=".",
                    help="directory of source docs to keep in sync "
                         "(default: current dir)")
    ap.add_argument("--watch", action="store_true",
                    help="run continuously, reacting to file changes in real time")
    ap.add_argument("--config", default=None,
                    help=f"YAML config (default: ./{DEFAULT_CONFIG_FILE} if present)")
    ap.add_argument("--daemon-url", default=None,
                    help=f"GPU daemon base URL (default: {DEFAULT_DAEMON_URL} "
                         f"or $RAG_DAEMON_URL)")
    ap.add_argument("--state-file", default=DEFAULT_STATE_FILE,
                    help=f"state file name under ROOT (default: {DEFAULT_STATE_FILE})")
    ap.add_argument("--debounce", type=float, default=1.0,
                    help="--watch debounce seconds (default: 1.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would sync without touching the daemon")
    ap.add_argument("--force", action="store_true",
                    help="ignore if_checksum fast-path: re-ingest every file "
                         "regardless of checksum")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    root = Path(args.root).resolve()
    if not root.is_dir():
        _ColorLog.err(f"root is not a directory: {root}")
        return 1

    config = load_config(args.config)
    daemon_url = (args.daemon_url
                  or config.get("daemon_url")
                  or DEFAULT_DAEMON_URL)
    daemon = DaemonClient(daemon_url)

    if args.force:
        # Force mode: erase known checksums from state view so every file is
        # treated as NEW/CHANGED. We do this by dropping the comparison basis —
        # simplest is to pass a flag down. For run_once we stub it by clearing
        # the on-disk state's checksums before the pass.
        _ColorLog.warn("--force requested: will re-ingest ALL files")

    # Fail-closed: if the daemon is down, never mark anything synced.
    if not args.dry_run:
        if not daemon.healthcheck():
            _ColorLog.err(
                f"GPU daemon unreachable at {daemon_url}. "
                f"Refusing to sync (would desync state). Is serve_gpu.py up?")
            return 2
        _ColorLog.info(f"daemon healthy at {daemon_url}")

    # Force mode: clear stored checksums so everything re-syncs.
    if args.force and not args.dry_run:
        sf = root / args.state_file
        if sf.exists():
            try:
                data = json.loads(sf.read_text(encoding="utf-8"))
                for v in data.values():
                    v["checksum"] = ""
                sf.write_text(json.dumps(data, indent=2, sort_keys=True),
                              encoding="utf-8")
            except Exception:
                pass

    if args.watch:
        return run_watch(root, daemon, config, args.state_file, args.debounce)
    return run_once(root, daemon, config, args.state_file, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
