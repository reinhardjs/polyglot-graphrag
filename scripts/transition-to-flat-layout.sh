#!/usr/bin/env bash
# =============================================================================
# transition-to-flat-layout.sh
# -----------------------------------------------------------------------------
# One-time transition script: promotes a versioned folder (e.g. `v3/`) to the
# repository ROOT, removes folder-based versioning, and prepares the repo for
# release-please + Git-tag versioning.
#
# This is the EXACT procedure used to migrate polyglot-graphrag from
# `v2/` -> `v3/` -> root. Re-run only if you have ANOTHER versioned folder to
# promote (you shouldn't — versioning is now tag-based).
#
# SAFETY:
#   * Uses `git mv` for every tracked file/dir -> full git history is PRESERVED.
#   * Untracked files (e.g. SNOMED_BENCHMARK.md) are moved with plain `mv`.
#   * Empty gitignored dirs (labels/, logs/) are recreated at root (runtime only).
#   * The live daemon keeps its cwd inode; update systemd WorkingDirectory
#     separately (see STEP 4) so the NEXT restart works.
#
# USAGE:
#   bash scripts/transition-to-flat-layout.sh <versioned-folder>
#   e.g.  bash scripts/transition-to-flat-layout.sh v3
# =============================================================================
set -euo pipefail

SRC="${1:-v3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d "$SRC" ]; then
  echo "ERROR: source folder '$SRC' not found." >&2
  exit 1
fi

echo "==> Promoting '$SRC/' to repo root (git history preserved)..."

# Move every tracked/untracked top-level entry except caches.
for item in $(ls -A "$SRC" | grep -vE '__pycache__|\.pytest_cache'); do
  case "$item" in
    README.md)
      # Replace the (temporary) root pointer README with the real one.
      [ -f README.md ] && git rm -q README.md 2>/dev/null || rm -f README.md
      git mv "$SRC/README.md" README.md 2>/dev/null || mv "$SRC/README.md" README.md
      ;;
    .gitignore)
      # Merge: append source ignores to root .gitignore, drop source copy.
      cat "$SRC/.gitignore" >> .gitignore
      git rm -q "$SRC/.gitignore" 2>/dev/null || rm -f "$SRC/.gitignore"
      ;;
    *)
      if git ls-files --error-unmatch "$SRC/$item" >/dev/null 2>&1; then
        git mv "$SRC/$item" "./$item"
      else
        mv "$SRC/$item" "./$item"   # untracked file
      fi
      ;;
  esac
done

# Recreate runtime dirs (gitignored, were empty).
mkdir -p labels logs

# Remove the now-empty versioned folder.
rm -rf "$SRC"
echo "==> Removed '$SRC/' (history retained in git log)."

echo
echo "==> STEP 4 (manual, requires sudo): point systemd at the new root."
echo "    sudo sed -i 's#WorkingDirectory=.*rag-system/$SRC#WorkingDirectory=/mnt/data-970-plus/rag-system#' \\"
echo "        /etc/systemd/system/rag-gpu-daemon.service"
echo "    sudo systemctl daemon-reload"
echo "    (Running PID keeps its cwd; change applies on next daemon restart.)"

echo
echo "==> Next: add release-please files, then run from repo root:"
echo "    git add -A && git commit -m 'refactor: flatten versioned folder to root; adopt release-please'"
echo "    git push origin main"
echo "    git tag v0.1.0-beta.1 && git push origin v0.1.0-beta.1   # or let release-please cut it"
