"""Signature helpers built on top of :mod:`iisignature` for quick experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

try:  # pragma: no cover - import guard keeps module importable sans dependency
    import iisignature
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "iisignature is required for models.signature; add it to your environment."
    ) from exc

Direction = Literal["forward", "reverse"]
DIRECTION_CHOICES: tuple[Direction, ...] = ("forward", "reverse")


@dataclass(slots=True)
class SignatureResult:
    """Container with signature vector and bookkeeping metadata."""

    signature: np.ndarray
    depth: int
    num_samples: int
    direction: Direction
    source: Path
    path_points: int
    path_length: float
    start_xy: tuple[float, float]

    @property
    def dimension(self) -> int:
        return int(self.signature.size)


def signature_from_json(
    polyline_json: Path,
    *,
    num_samples: int = 256,
    depth: int = 4,
    direction: Direction = "forward",
) -> SignatureResult:
    """Load a polyline JSON file, normalize it, and compute its signature."""

    points = _load_polyline(polyline_json)
    resampled, path_length = _resample_polyline(points, num_samples)
    ordered = resampled if direction == "forward" else resampled[::-1]
    start_xy = tuple(map(float, ordered[0]))
    centered = ordered - ordered[0]
    signature = _compute_signature(centered, depth)
    return SignatureResult(
        signature=signature,
        depth=depth,
        num_samples=num_samples,
        direction=direction,
        source=polyline_json,
        path_points=int(points.shape[0]),
        path_length=path_length,
        start_xy=start_xy,
    )


def write_signature_csv(result: SignatureResult, output_path: Path) -> Path:
    """Append the signature vector to a CSV file with human-friendly metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "polyline",
        "depth",
        "num_samples",
        "signature_dim",
        "path_points",
        "path_length",
        "start_x",
        "start_y",
    ] + [f"sig_{i}" for i in range(result.dimension)]

    write_header = not output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        row = [
            str(result.source),
            result.depth,
            result.num_samples,
            result.dimension,
            result.path_points,
            f"{result.path_length:.3f}",
            f"{result.start_xy[0]:.3f}",
            f"{result.start_xy[1]:.3f}",
        ] + [f"{value:.10f}" for value in result.signature]
        writer.writerow(row)
    return output_path


def _load_polyline(polyline_json: Path) -> np.ndarray:
    payload = json.loads(polyline_json.read_text())
    if "polyline" not in payload:
        raise ValueError(f"{polyline_json} is missing a 'polyline' key")
    coords = np.asarray(payload["polyline"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Polyline must be shaped (N, 2).")
    return coords


def _resample_polyline(points: np.ndarray, num_samples: int) -> tuple[np.ndarray, float]:
    if num_samples < 2:
        raise ValueError("Need at least two samples for a path.")
    if points.shape[0] < 2:
        raise ValueError("Polyline must contain at least two points.")

    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length == 0.0:
        tiled = np.repeat(points[:1], num_samples, axis=0)
        return tiled, total_length

    target = np.linspace(0.0, total_length, num_samples)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    resampled = np.stack((x, y), axis=1)
    return resampled, total_length


def _compute_signature(path: np.ndarray, depth: int) -> np.ndarray:
    if depth < 1:
        raise ValueError("Signature depth must be >= 1.")
    return iisignature.sig(path.astype(np.float64, copy=False), depth)


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute iisignature for a polyline JSON")
    parser.add_argument("polyline", type=Path, help="Path to *_longest.json produced earlier")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Resample the path to N equally spaced points (default: 256)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="iisignature truncation depth (default: 4)",
    )
    parser.add_argument(
        "--direction",
        choices=list(DIRECTION_CHOICES),
        default="forward",
        help="Process polyline in original ('forward') order or reversed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/signatures/signatures.csv"),
        help="CSV destination (default: data/signatures/signatures.csv).",
    )
    args = parser.parse_args(argv)

    result = signature_from_json(
        args.polyline,
        num_samples=args.num_samples,
        depth=args.depth,
        direction=args.direction,
    )

    csv_path = write_signature_csv(result, args.output)
    print(f"Appended signature (dim={result.dimension}) to {csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "SignatureResult",
    "signature_from_json",
    "write_signature_csv",
]
