"""Batch helpers for computing log signatures from stored polylines."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from controllers.artifacts import resolve_stage_artifact_path
from controllers.data_paths import DataPaths
from models.signature import (
    LogSignatureResult,
    append_log_signature_csv,
    default_log_signature_csv_path,
    log_signature_from_json,
)


def analyze_polylines(
    polylines: Sequence[Path],
    *,
    depth: int,
    summary_csv: Path,
    skip_existing: bool = True,
) -> tuple[list[LogSignatureResult], list[str]]:
    """Compute log signatures for every polyline JSON in ``polylines``."""

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    results: list[LogSignatureResult] = []
    existing_polylines: set[str] = set()
    logs: list[str] = []
    if skip_existing:
        existing_polylines = _load_polylines_from_csv(summary_csv)
    for path in polylines:
        path = path.resolve()
        if not path.exists():
            logs.append(f"[skipping] {path} – file not found")
            continue
        polyline_key = str(path)
        if skip_existing and polyline_key in existing_polylines:
            logs.append(f"[cached] {path.name} -> already in CSV")
            continue
        try:
            result = log_signature_from_json(path, depth=depth)
        except ValueError as exc:
            logs.append(f"[error] {path.name}: {exc}")
            continue
        append_log_signature_csv(result, summary_csv)
        if skip_existing:
            existing_polylines.add(polyline_key)
        results.append(result)
        logs.append(f"[ok] {path.name} → CSV row ({result.dimension} dims)")
    return results, logs


def compute_signature_flow(
    *,
    data_paths: DataPaths,
    selected_files: Sequence[str] | None,
    depth_value: float | int,
    overwrite: bool,
) -> tuple[list[list[str]], list[str], str]:
    if not selected_files:
        raise ValueError("Select at least one polyline JSON before running.")

    cfg = data_paths or DataPaths.from_data_dir()
    cfg.ensure_directories()
    depth = int(depth_value)

    polylines = [_resolve_polyline_path(entry, cfg.polyline_dir) for entry in selected_files]
    summary_csv = default_log_signature_csv_path(cfg.signatures_dir)

    results, logs = analyze_polylines(
        polylines,
        depth=depth,
        summary_csv=summary_csv,
        skip_existing=not overwrite,
    )

    table, headers = _load_csv_preview(summary_csv)
    summary_line = (
        f"Wrote {len(results)} new row(s) to `{summary_csv.name}` "
        f"(depth={depth})."
    )
    if logs:
        status = summary_line + "\n\n" + "\n".join(logs)
    else:
        status = summary_line
    return table, headers, status


def _load_polylines_from_csv(csv_path: Path) -> set[str]:
    """Read polyline paths already recorded in the summary CSV."""

    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        recorded = {
            row.get("polyline_json", "").strip()
            for row in reader
            if row.get("polyline_json")
        }
    return {item for item in recorded if item}


def _load_csv_preview(csv_path: Path, limit: int = 20) -> tuple[list[list[str]], list[str]]:
    if not csv_path.exists():
        return [], []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = reader.fieldnames or []

    if not rows:
        return [], headers

    tail = rows[-limit:]
    cols = headers if headers else list(tail[0].keys())
    table = [[row.get(col, "") for col in cols] for row in tail]
    return table, cols


__all__ = [
    "analyze_polylines",
    "compute_signature_flow",
]


def _resolve_polyline_path(entry: str, polyline_dir: Path) -> Path:
    candidate = resolve_stage_artifact_path(
        entry,
        [polyline_dir, Path.cwd()],
        stage_names=("polyline",),
        default_suffix=".json",
        extra_suffixes=(".json", ""),
    )
    if candidate is None:
        raise ValueError(f"Polyline {entry} was not found in {polyline_dir}.")
    return candidate
