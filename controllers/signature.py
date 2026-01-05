"""Batch helpers for computing log signatures from stored polylines."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from controllers.artifacts import ensure_flat_stage_identifier, resolve_stage_artifact_path
from controllers.data_paths import DataPaths
from models.signature import (
    LogSignatureResult,
    append_log_signature_csv,
    log_signature_from_json,
)


def analyze_polylines(
    polylines: Sequence[Path],
    *,
    depth: int,
    summary_csv: Path,
) -> tuple[list[LogSignatureResult], list[str]]:
    """Compute log signatures for every polyline JSON in ``polylines``."""

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    results: list[LogSignatureResult] = []
    logs: list[str] = []
    for path in polylines:
        path = path.resolve()
        if not path.exists():
            logs.append(f"[skipping] {path} – file not found")
            continue
        try:
            result = log_signature_from_json(path, depth=depth)
        except ValueError as exc:
            logs.append(f"[error] {path.name}: {exc}")
            continue
        append_log_signature_csv(result, summary_csv)
        results.append(result)
        logs.append(f"[ok] {path.name} → CSV row ({result.dimension} dims)")
    return results, logs


def compute_signature_flow(
    *,
    data_paths: DataPaths,
    selected_files: Sequence[str] | None,
    depth_value: float | int,
) -> tuple[list[list[str]], list[str], str]:
    if not selected_files:
        raise ValueError("Select at least one polyline JSON before running.")

    cfg = data_paths
    cfg.ensure_signature_directories()
    depth = int(depth_value)

    polylines = [_resolve_polyline_path(entry, cfg.polyline_dir) for entry in selected_files]
    summary_csv = cfg.summary_csv_path()

    results, logs = analyze_polylines(
        polylines,
        depth=depth,
        summary_csv=summary_csv,
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
    ensure_flat_stage_identifier(entry, description="Polyline filename")
    candidate = resolve_stage_artifact_path(
        entry,
        [polyline_dir],
        stage_names=("polyline",),
        default_suffix=".json",
        extra_suffixes=(".json", ""),
    )
    if candidate is None:
        raise ValueError(f"Polyline {entry} was not found in {polyline_dir}.")
    return candidate
