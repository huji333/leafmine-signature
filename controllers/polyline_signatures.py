"""Batch helpers for computing log signatures from stored polylines."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

from models.signature import (
    LogSignatureResult,
    append_log_signature_csv,
    default_log_signature_csv_path,
    log_signature_from_json,
)


@dataclass(slots=True)
class PolylineSignatureConfig:
    """Filesystem layout for batch log-signature analysis."""

    data_dir: Path = Path("data")
    polyline_dir: Path = Path("data/polylines")
    output_dir: Path = Path("data/logsig")
    summary_csv: Path = field(default_factory=default_log_signature_csv_path)

    def ensure(self) -> None:
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)


def analyze_polylines(
    polylines: Sequence[Path],
    *,
    depth: int,
    config: PolylineSignatureConfig | None = None,
    skip_existing: bool = True,
    summary: bool = True,
) -> list[LogSignatureResult]:
    """Compute log signatures for every polyline JSON in ``polylines``."""

    cfg = config or PolylineSignatureConfig()
    cfg.ensure()
    results: list[LogSignatureResult] = []
    existing_polylines: set[str] = set()
    if skip_existing and summary:
        existing_polylines = _load_polylines_from_csv(cfg.summary_csv)
    for path in polylines:
        path = path.resolve()
        if not path.exists():
            print(f"[skipping] {path} – file not found", file=sys.stderr)
            continue
        polyline_key = str(path)
        if skip_existing and summary and polyline_key in existing_polylines:
            print(f"[cached] {path.name} -> already in CSV")
            continue
        try:
            result = log_signature_from_json(path, depth=depth)
        except ValueError as exc:
            print(f"[error] {path.name}: {exc}", file=sys.stderr)
            continue
        if summary:
            append_log_signature_csv(result, cfg.summary_csv)
            if skip_existing:
                existing_polylines.add(polyline_key)
        results.append(result)
        print(f"[ok] {path.name} → CSV row ({result.dimension} dims)")
    return results


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


__all__ = [
    "PolylineSignatureConfig",
    "analyze_polylines",
]
