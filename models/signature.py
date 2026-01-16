"""Log-signature helpers built on top of :mod:`roughpy`."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import numpy as np

from .utils.naming import canonical_sample_name, prefixed_name

try:  # pragma: no cover - allow importing without optional dependency
    import roughpy as rp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "roughpy is required for models.signature; install it via `uv pip install roughpy`."
    ) from exc


def default_log_signature_csv_path(
    output_dir: Path | str,
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
    path_length: float
    resample_points: int
    polyline_path: Path
    sample_filename: str | None = None

    @property
    def dimension(self) -> int:
        return int(self.vector.size)


@lru_cache(maxsize=None)
def _logsig_context(dimension: int, depth: int):
    """Cache roughpy contexts per (dimension, depth)."""

    return rp.get_context(width=int(dimension), depth=int(depth), coeffs=rp.DPReal)


def log_signature_from_json(
    polyline_json: Path,
    *,
    depth: int = 3,
) -> LogSignatureResult:
    """Load a resampled polyline JSON payload and compute its log signature."""

    payload = _load_polyline_payload(polyline_json)
    resampled = np.asarray(payload["resampled_polyline"], dtype=np.float64)
    if resampled.ndim != 2 or resampled.shape[1] != 2:
        raise ValueError(f"{polyline_json} has malformed resampled coordinates.")
    vector = _compute_log_signature(resampled, depth)
    path_length = float(payload.get("path_length", float("nan")))
    resample_points = int(payload.get("resample_points", int(resampled.shape[0])))
    sample_filename = _extract_sample_filename(payload, polyline_json)
    return LogSignatureResult(
        vector=vector,
        depth=int(depth),
        path_length=path_length,
        resample_points=resample_points,
        polyline_path=polyline_json,
        sample_filename=sample_filename,
    )


def append_log_signature_csv(result: LogSignatureResult, csv_path: Path) -> Path:
    """Append a summary row for the result to ``csv_path``."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "filename",
        "polyline_json",
        "depth",
        "dimension",
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
        raise ValueError("Log-signature requires polyline dimension >= 2.")
    if points.shape[0] < 2:
        raise ValueError("Polyline must contain at least two points.")

    increments = np.diff(points.astype(np.float64, copy=False), axis=0)
    ctx = _logsig_context(dimension, depth)
    interval = rp.RealInterval(0.0, float(increments.shape[0]))
    stream = rp.LieIncrementStream.from_increments(increments, ctx=ctx, resolution=0)
    vector = stream.log_signature(interval)
    return np.asarray(vector, dtype=np.float64)


def _load_polyline_payload(polyline_json: Path) -> dict[str, object]:
    payload = json.loads(polyline_json.read_text())
    if "resampled_polyline" not in payload:
        raise ValueError(
            f"{polyline_json} is missing resampled polyline data; rerun the polyline generation step."
        )
    return payload


__all__ = [
    "LogSignatureResult",
    "append_log_signature_csv",
    "log_signature_from_json",
]
