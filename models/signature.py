"""Log-signature helpers built on top of :mod:`iisignature`."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from functools import lru_cache

from .utils import canonical_sample_name, prefixed_name

try:  # pragma: no cover - allow importing without optional dependency
    import iisignature
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "iisignature is required for models.signature; install it via `uv pip install iisignature`."
    ) from exc


def default_log_signature_csv_path(
    output_dir: Path | str = Path("data/logsig"),
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Return a timestamped CSV path inside ``output_dir``.

    Each call emits a new ``logsignatures_<timestamp>.csv`` file so downstream
    tooling can keep per-run context unless a specific file override is passed.
    """

    base_dir = Path(output_dir)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return base_dir / f"logsignatures_{stamp}.csv"


@dataclass(slots=True)
class LogSignatureResult:
    """Container storing the computed log signature and metadata."""

    vector: np.ndarray
    depth: int
    num_samples: int
    path_length: float
    resample_points: int
    polyline_path: Path
    npz_path: Path | None = None
    sample_filename: str | None = None

    @property
    def dimension(self) -> int:
        return int(self.vector.size)


@lru_cache(maxsize=None)
def _prepared_logsigs(dimension: int, depth: int):
    """Cache iisignature.prepare results per (dimension, depth)."""

    return iisignature.prepare(int(dimension), int(depth))


def log_signature_from_json(
    polyline_json: Path,
    *,
    depth: int = 4,
) -> LogSignatureResult:
    """Load a resampled polyline JSON payload and compute its log signature."""

    payload = _load_polyline_payload(polyline_json)
    resampled = np.asarray(payload["resampled_polyline"], dtype=np.float64)
    if resampled.ndim != 2 or resampled.shape[1] != 2:
        raise ValueError(f"{polyline_json} has malformed resampled coordinates.")
    vector = _compute_log_signature(resampled, depth)
    num_samples = int(resampled.shape[0])
    path_length = float(payload.get("path_length", float("nan")))
    resample_points = int(payload.get("resample_points", num_samples))
    sample_filename = _extract_sample_filename(payload, polyline_json)
    return LogSignatureResult(
        vector=vector,
        depth=int(depth),
        num_samples=num_samples,
        path_length=path_length,
        resample_points=resample_points,
        polyline_path=polyline_json,
        sample_filename=sample_filename,
    )


def save_log_signature_npz(
    result: LogSignatureResult,
    destination: Path,
) -> Path:
    """Persist the vector + metadata to an NPZ file and store its path on ``result``."""

    destination = _resolve_npz_destination(result, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        vector=result.vector,
        depth=result.depth,
        num_samples=result.num_samples,
        path_length=result.path_length,
        resample_points=result.resample_points,
        polyline=str(result.polyline_path),
    )
    result.npz_path = destination
    return destination


def append_log_signature_csv(result: LogSignatureResult, csv_path: Path) -> Path:
    """Append a summary row for the result to ``csv_path``."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "filename",
        "polyline_json",
        "depth",
        "dimension",
        "num_samples",
        "resample_points",
        "path_length",
        "log_signature",
    ]
    needs_header = not csv_path.exists()
    sample_filename = result.sample_filename or prefixed_name(
        "segmented",
        canonical_sample_name(result.polyline_path),
        ".png",
    )
    vector_values = [float(value) for value in result.vector.ravel().tolist()]
    signature_blob = json.dumps(vector_values, ensure_ascii=False)
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(header)
        writer.writerow(
            [
                sample_filename,
                str(result.polyline_path),
                result.depth,
                result.dimension,
                result.num_samples,
                result.resample_points,
                f"{result.path_length:.6f}",
                signature_blob,
            ]
        )
    return csv_path


def _extract_sample_filename(payload: dict[str, object], polyline_json: Path) -> str:
    candidate = payload.get("segmented_filename")
    if isinstance(candidate, str):
        cleaned = candidate.strip()
        if cleaned:
            return cleaned
    sample_base = payload.get("sample_base")
    if isinstance(sample_base, str):
        cleaned = sample_base.strip()
        if cleaned:
            return prefixed_name("segmented", cleaned, ".png")
    return prefixed_name("segmented", canonical_sample_name(polyline_json), ".png")


def _compute_log_signature(points: np.ndarray, depth: int) -> np.ndarray:
    if depth < 1:
        raise ValueError("Log-signature depth must be >= 1.")
    dimension = int(points.shape[1])
    if dimension <= 0:
        raise ValueError("Polyline points must have a positive dimension.")
    if dimension < 2:
        raise ValueError("iisignature requires polyline dimension >= 2.")
    double_points = points.astype(np.float64, copy=False)
    prepared = _prepared_logsigs(dimension, depth)
    vector = iisignature.logsig(double_points, prepared)
    return np.asarray(vector, dtype=np.float64)


def _resolve_npz_destination(result: LogSignatureResult, destination: Path) -> Path:
    destination = destination.expanduser()
    if destination.suffix:
        return destination
    filename = f"{result.polyline_path.stem}_logsig_d{result.depth}.npz"
    return destination / filename


def _load_polyline_payload(polyline_json: Path) -> dict[str, object]:
    payload = json.loads(polyline_json.read_text())
    if "resampled_polyline" not in payload:
        raise ValueError(
            f"{polyline_json} is missing resampled polyline data; rerun the polyline generation step."
        )
    return payload


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute log signature for a polyline JSON.")
    parser.add_argument("polyline", type=Path, help="Path to *_route.json produced earlier.")
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Log-signature depth (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/logsig"),
        help="Output NPZ path or directory (default: data/logsig).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=default_log_signature_csv_path(),
        help="Summary CSV destination (default: data/logsig/logsignatures_<timestamp>.csv).",
    )
    args = parser.parse_args(argv)

    result = log_signature_from_json(args.polyline, depth=args.depth)
    npz_path = save_log_signature_npz(result, args.output)
    append_log_signature_csv(result, args.summary_csv)
    print(f"Saved log signature (dim={result.dimension}) to {npz_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "LogSignatureResult",
    "append_log_signature_csv",
    "log_signature_from_json",
    "save_log_signature_npz",
]
