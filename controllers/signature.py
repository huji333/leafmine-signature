"""Batch helpers for computing log signatures from stored polylines."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
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


@dataclass(slots=True)
class PolylineSignatureConfig:
    """Filesystem layout for batch log-signature analysis."""

    polyline_dir: Path = Path("data/polylines")
    output_dir: Path = Path("data/logsig")
    summary_csv: Path = field(default_factory=default_log_signature_csv_path)

    def ensure(self) -> None:
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_data_paths(cls, data_paths: DataPaths) -> PolylineSignatureConfig:
        summary_csv = default_log_signature_csv_path(data_paths.signatures_dir)
        return cls(
            polyline_dir=data_paths.polyline_dir,
            output_dir=data_paths.signatures_dir,
            summary_csv=summary_csv,
        )

    def resolve_polyline_path(self, entry: str) -> Path:
        candidate = resolve_stage_artifact_path(
            entry,
            [self.polyline_dir, Path.cwd()],
            stage_names=("polyline",),
            default_suffix=".json",
            extra_suffixes=(".json", ""),
        )
        if candidate is None:
            raise ValueError(f"Polyline {entry} was not found in {self.polyline_dir}.")
        return candidate


def analyze_polylines(
    polylines: Sequence[Path],
    *,
    depth: int,
    config: PolylineSignatureConfig | None = None,
    skip_existing: bool = True,
    summary: bool = True,
) -> tuple[list[LogSignatureResult], list[str]]:
    """Compute log signatures for every polyline JSON in ``polylines``."""

    cfg = config or PolylineSignatureConfig()
    cfg.ensure()
    results: list[LogSignatureResult] = []
    existing_polylines: set[str] = set()
    logs: list[str] = []
    if skip_existing and summary:
        existing_polylines = _load_polylines_from_csv(cfg.summary_csv)
    for path in polylines:
        path = path.resolve()
        if not path.exists():
            logs.append(f"[skipping] {path} – file not found")
            continue
        polyline_key = str(path)
        if skip_existing and summary and polyline_key in existing_polylines:
            logs.append(f"[cached] {path.name} -> already in CSV")
            continue
        try:
            result = log_signature_from_json(path, depth=depth)
        except ValueError as exc:
            logs.append(f"[error] {path.name}: {exc}")
            continue
        if summary:
            append_log_signature_csv(result, cfg.summary_csv)
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

    signature_config = PolylineSignatureConfig.from_data_paths(data_paths)
    signature_config.ensure()
    depth = int(depth_value)

    paths = [signature_config.resolve_polyline_path(entry) for entry in selected_files]

    results, logs = analyze_polylines(
        paths,
        depth=depth,
        config=signature_config,
        skip_existing=not overwrite,
        summary=True,
    )

    table, headers = _load_csv_preview(signature_config.summary_csv)
    summary_line = (
        f"Wrote {len(results)} new row(s) to `{signature_config.summary_csv.name}` "
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
    "PolylineSignatureConfig",
    "analyze_polylines",
    "compute_signature_flow",
]
