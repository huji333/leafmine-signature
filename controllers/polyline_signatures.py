"""Batch helpers for computing log signatures from stored polylines."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute log signatures for polyline JSON files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional polyline JSON paths. If omitted, scans <data-dir>/polylines/*.json.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Project data directory containing polylines/ and logsig/ subfolders.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Log-signature depth (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute log signatures even if they already exist in the CSV.",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    polyline_dir = (data_dir / "polylines").resolve()
    output_dir = (data_dir / "logsig").resolve()
    summary_csv = default_log_signature_csv_path(output_dir)

    cfg = PolylineSignatureConfig(
        data_dir=data_dir,
        polyline_dir=polyline_dir,
        output_dir=output_dir,
        summary_csv=summary_csv,
    )

    if args.paths:
        candidates = [_resolve_polyline_path(arg, polyline_dir) for arg in args.paths]
        candidates = [path for path in candidates if path is not None]
    else:
        candidates = sorted(polyline_dir.glob("*.json"))

    if not candidates:
        print("No polyline JSON files found; nothing to do.", file=sys.stderr)
        return 1

    analyze_polylines(
        candidates,
        depth=args.depth,
        config=cfg,
        skip_existing=not args.overwrite,
        summary=True,
    )
    return 0


def _resolve_polyline_path(arg: Path, polyline_dir: Path) -> Path | None:
    raw = Path(arg)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((polyline_dir / raw).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    print(f"[missing] {arg} – skipping", file=sys.stderr)
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "PolylineSignatureConfig",
    "analyze_polylines",
]
