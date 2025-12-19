"""Controller helpers for the Line Extension / longest-path workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from models.longest_component import export_longest_path
from models.signature import (
    Direction,
    SignatureResult,
    signature_from_json,
    write_signature_csv,
)


@dataclass(slots=True)
class LineExtensionConfig:
    """Filesystem layout and options for the longest-path workflow."""

    skeleton_dir: Path = Path("data/skeletonized")
    polyline_dir: Path = Path("data/tmp")
    signatures_dir: Path = Path("data/signatures")
    signature_csv: Path = Path("data/signatures/signatures.csv")
    directions: Sequence[Direction] = ("forward", "reverse")

    def ensure_directories(self) -> None:
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.signature_csv.parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class LineExtensionResult:
    highlight_image: Image.Image
    polyline_payload: dict[str, object]
    signature_summary: list[dict[str, object]] | None
    signature_message: str
    highlight_path: Path
    polyline_path: Path
    signature_results: list[SignatureResult]
    signature_csv_path: Path | None


def run_longest_path_flow(
    skeleton_filename: str | Path,
    *,
    compute_signature: bool,
    num_samples: int,
    depth: int = 4,
    config: LineExtensionConfig | None = None,
) -> LineExtensionResult:
    """Extract the longest path and optionally append signatures.

    Args:
        skeleton_filename: Absolute path or basename under ``config.skeleton_dir``.
        compute_signature: Whether to append signatures for each configured
            direction.
        num_samples: Resampling count for signature computation.
        depth: Signature depth.
        config: Optional layout override.
    """

    cfg = config or LineExtensionConfig()
    cfg.ensure_directories()

    skeleton_path = Path(skeleton_filename)
    if not skeleton_path.is_absolute():
        skeleton_path = cfg.skeleton_dir / skeleton_path.name
    if not skeleton_path.exists():
        raise FileNotFoundError(f"Could not find {skeleton_path}")

    artifacts = export_longest_path(skeleton_path, cfg.polyline_dir)
    highlight_path = artifacts["highlight"]
    polyline_path = artifacts["polyline"]

    with Image.open(highlight_path) as highlight_image:
        highlight = highlight_image.copy()

    payload = json.loads(polyline_path.read_text())

    signature_results: list[SignatureResult] = []
    summary: list[dict[str, object]] | None = None
    message = "Signature computation skipped for this run."
    csv_path: Path | None = None

    if compute_signature:
        for direction in cfg.directions:
            result = signature_from_json(
                polyline_path,
                num_samples=num_samples,
                depth=depth,
                direction=direction,
            )
            csv_path = write_signature_csv(result, cfg.signature_csv)
            signature_results.append(result)

        summary = [
            {
                "direction": result.direction,
                "depth": result.depth,
                "num_samples": result.num_samples,
                "signature_dim": result.dimension,
                "path_points": result.path_points,
                "path_length": round(result.path_length, 3),
                "start_xy": [round(result.start_xy[0], 3), round(result.start_xy[1], 3)],
                "end_xy": [round(result.end_xy[0], 3), round(result.end_xy[1], 3)],
                "csv_path": str(csv_path) if csv_path else str(cfg.signature_csv),
            }
            for result in signature_results
        ]
        message = (
            f"Appended {len(signature_results)} signature rows to {csv_path}. "
            f"Directions: {', '.join(cfg.directions)}."
        )

    return LineExtensionResult(
        highlight_image=highlight,
        polyline_payload=payload,
        signature_summary=summary,
        signature_message=message,
        highlight_path=highlight_path,
        polyline_path=polyline_path,
        signature_results=signature_results,
        signature_csv_path=csv_path,
    )


__all__ = [
    "LineExtensionConfig",
    "LineExtensionResult",
    "run_longest_path_flow",
]
