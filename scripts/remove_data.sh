#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

resolve_path() {
  python3 - "$1" <<'PY'
import os, sys
raw = sys.argv[1]
print(os.path.abspath(os.path.expanduser(raw)))
PY
}

load_data_dir_from_config() {
  python3 - <<'PY'
from controllers.settings import load_data_dir
print(load_data_dir())
PY
}

if [[ $# -gt 0 && -n "${1:-}" ]]; then
  RAW_DATA_DIR="$1"
else
  RAW_DATA_DIR="$(load_data_dir_from_config)"
fi

DATA_DIR="$(resolve_path "$RAW_DATA_DIR")"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory $DATA_DIR does not exist; nothing to clean."
  exit 0
fi

ANNOTATION_CSV_PATH="${ANNOTATION_CSV:-}"
if [[ -z "$ANNOTATION_CSV_PATH" ]]; then
  ANNOTATION_CSV_PATH="$DATA_DIR/sample_annotation.csv"
fi
if [[ "$ANNOTATION_CSV_PATH" != /* ]]; then
  ANNOTATION_CSV_PATH="$DATA_DIR/${ANNOTATION_CSV_PATH#/}"
fi
ANNOTATION_CSV_PATH="$(resolve_path "$ANNOTATION_CSV_PATH")"

rel_path() {
  local target="$1"
  local prefix="$DATA_DIR/"
  if [[ "$target" == "$prefix"* ]]; then
    printf '%s\n' "${target#"$prefix"}"
  else
    printf '%s\n' "$target"
  fi
}

echo "Cleaning data directory $DATA_DIR"
if [[ -f "$ANNOTATION_CSV_PATH" ]]; then
  echo "Preserving annotation CSV: $(rel_path "$ANNOTATION_CSV_PATH")"
fi

while IFS= read -r -d '' entry; do
  if [[ "$entry" = "$ANNOTATION_CSV_PATH" ]]; then
    continue
  fi

  rel="$(rel_path "$entry")"
  if [[ -d "$entry" ]]; then
    rm -rf "$entry"
    mkdir -p "$entry"
    echo "Cleared directory $rel"
  else
    rm -f "$entry"
    echo "Removed file $rel"
  fi

done < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -print0)

echo "Done."
