"""Segmentation->signature orchestration helpers.

This module keeps the end-to-end pipeline (segmented mask -> skeleton ->
longest path -> signature) out of the UI layer so both Gradio tabs and
batch/CLI tooling can reuse the exact same logic. Today we assume the
caller already produced a binary mask (e.g., manual tracing), but the
functions are structured so a future ML segmenter can plug in ahead of
`process_segmented_mask` without touching the downstream flow.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
import re
from pathlib import Path
from typing import Protocol, Sequence

from PIL import Image

from models.longest_component import export_longest_path
from models.signature import (
    DIRECTION_CHOICES,
    Direction,
    SignatureResult,
    signature_from_json,
    write_signature_csv,
)
from models.skeletonization import run_skeletonization


@dataclass(slots=True)
class PipelineConfig:
    """File-system destinations for every stage of the pipeline."""

    segmented_dir: Path = Path("data/segmented")
    skeleton_dir: Path = Path("data/skeletonized")
    polyline_dir: Path = Path("data/tmp")
    signatures_dir: Path = Path("data/signatures")
    signature_csv: Path = Path("data/signatures/signatures.csv")

    def ensure_directories(self) -> None:
        """Create output folders so downstream saves never fail."""

        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.signature_csv.parent.mkdir(parents=True, exist_ok=True)

    def with_timestamped_signature_csv(self, *, prefix: str = "signatures") -> "PipelineConfig":
        """Return a new config whose signature CSV includes a timestamp.

        Useful for batch/Make runs where each invocation should write to a
        fresh CSV like ``data/signatures/<prefix>_20251219-153000.csv``.
        """

        safe_prefix = _slugify(prefix) or "signatures"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{safe_prefix}_{timestamp}.csv"
        return replace(self, signature_csv=self.signatures_dir / filename)


@dataclass(slots=True)
class PipelineResult:
    """Artifacts and metadata emitted by a successful pipeline run."""

    mask_path: Path
    skeleton_path: Path
    highlight_path: Path
    polyline_path: Path
    signature_csv_path: Path
    signature_results: list[SignatureResult]

    @property
    def signature(self) -> SignatureResult:
        """Convenience accessor returning the first signature result."""

        return self.signature_results[0]


class Segmenter(Protocol):
    """Callable interface for future ML segmentation models."""

    def __call__(self, image: Image.Image) -> Image.Image:  # pragma: no cover - protocol
        ...


def process_segmented_mask(
    segmented_mask: Image.Image | str | Path,
    *,
    base_name: str | None = None,
    config: PipelineConfig | None = None,
    num_samples: int = 256,
    depth: int = 4,
    direction: Direction = "forward",
    directions: Sequence[Direction] | None = None,
) -> PipelineResult:
    """Persist a segmented mask and push it through the remaining stages.

    Args:
        segmented_mask: Pillow image or path to a binary mask (white mine,
            black background).
        base_name: Optional override for artifact filenames (without
            extension). Defaults to the source stem or a timestamp if the
            mask is an in-memory image.
        config: Optional :class:`PipelineConfig` that controls output
            folders. The defaults match the repo's ``data/*`` layout.
        num_samples: Resampling count before signature calculation.
        depth: Truncation depth for the path signature.
        direction: Whether the signature should follow the forward or
            reverse traversal of the polyline.

    Returns:
        PipelineResult containing file paths for every artifact plus the
        in-memory :class:`SignatureResult`.
    """

    cfg = config or PipelineConfig()
    cfg.ensure_directories()

    normalized_name = _ensure_mask_prefix(_resolve_base_name(segmented_mask, base_name))
    mask_image = _load_image(segmented_mask, mode="L")
    mask_path = cfg.segmented_dir / f"{normalized_name}.png"
    mask_image.save(mask_path)

    skeleton_artifacts = run_skeletonization(mask_path)
    skeleton_path = cfg.skeleton_dir / f"{normalized_name}_skeleton.png"
    skeleton_artifacts["skeleton_mask"].save(skeleton_path)

    longest = export_longest_path(skeleton_path, cfg.polyline_dir)
    directions_to_use: tuple[Direction, ...]
    if directions:
        directions_to_use = tuple(directions)
    else:
        directions_to_use = (direction,)

    signature_results: list[SignatureResult] = []
    signature_csv_path: Path | None = None

    for dir_choice in directions_to_use:
        if dir_choice not in DIRECTION_CHOICES:
            raise ValueError(f"Unsupported direction: {dir_choice}")
        signature = signature_from_json(
            longest["polyline"],
            num_samples=num_samples,
            depth=depth,
            direction=dir_choice,
        )
        signature_csv_path = write_signature_csv(signature, cfg.signature_csv)
        signature_results.append(signature)

    if not signature_results or signature_csv_path is None:
        raise RuntimeError("No signatures were generated; ensure directions are provided.")

    return PipelineResult(
        mask_path=mask_path,
        skeleton_path=skeleton_path,
        highlight_path=longest["highlight"],
        polyline_path=longest["polyline"],
        signature_csv_path=signature_csv_path,
        signature_results=signature_results,
    )


def process_with_segmenter(
    image: Image.Image | str | Path,
    segmenter: Segmenter,
    *,
    base_name: str | None = None,
    config: PipelineConfig | None = None,
    num_samples: int = 256,
    depth: int = 4,
    direction: Direction = "forward",
    directions: Sequence[Direction] | None = None,
) -> PipelineResult:
    """Run an ML/heuristic segmenter first, then finish the pipeline.

    This helper is future-facing: provide any callable that consumes an
    RGB Pillow image and returns a binary mask image, and we will persist
    the mask before delegating to :func:`process_segmented_mask`.
    """

    rgb_image = _load_image(image).convert("RGB")
    mask_image = segmenter(rgb_image)
    if not isinstance(mask_image, Image.Image):
        raise TypeError("segmenter must return a Pillow Image")

    return process_segmented_mask(
        mask_image,
        base_name=base_name or _resolve_base_name(image, None),
        config=config,
        num_samples=num_samples,
        depth=depth,
        direction=direction,
        directions=directions,
    )


_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _ensure_mask_prefix(base_name: str) -> str:
    if not base_name:
        return "mask"
    return base_name if base_name.startswith("mask_") else f"mask_{base_name}"


def _resolve_base_name(source: object, override: str | None) -> str:
    if override:
        slug = _slugify(override)
        if slug:
            return slug
        raise ValueError("base_name must contain at least one alphanumeric character")

    if isinstance(source, (str, Path)):
        slug = _slugify(Path(source).stem)
        if slug:
            return slug

    timestamp = datetime.now().strftime("mask_%Y%m%d-%H%M%S")
    return timestamp


def _slugify(candidate: str) -> str:
    slug = _SLUG_PATTERN.sub("_", candidate).strip("._-")
    return slug


def _load_image(source: Image.Image | str | Path, *, mode: str | None = None) -> Image.Image:
    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        path = Path(source)
        with Image.open(path) as opened:
            image = opened.copy()
    if mode is not None:
        image = image.convert(mode)
    return image


def _default_data_dir() -> Path:
    env_dir = os.environ.get("LEAFMINE_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.cwd() / "data"


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch process every segmented PNG sitting under data/segmented/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Root folder containing segmented/, skeletonized/, tmp/, signatures/ (default: ./data or $LEAFMINE_DATA_DIR).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Resample each polyline to N points before computing signatures (default: 256).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="iisignature truncation depth (default: 4).",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=list(DIRECTION_CHOICES),
        default=list(DIRECTION_CHOICES),
        help="Signature directions to compute per mask (default: forward reverse).",
    )
    parser.add_argument(
        "--csv-prefix",
        type=str,
        default="batch",
        help="Prefix for the timestamped CSV saved under data/signatures/ (default: batch).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap how many masks to process from the segmented directory.",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser()
    segmented_dir = data_dir / "segmented"
    if not segmented_dir.exists():
        parser.error(f"{segmented_dir} does not exist.")

    mask_paths = sorted(segmented_dir.glob("*.png"))
    if args.limit is not None:
        mask_paths = mask_paths[: args.limit]

    if not mask_paths:
        print(f"No PNG masks found under {segmented_dir}", file=sys.stderr)
        return 1

    cfg = PipelineConfig(
        segmented_dir=segmented_dir,
        skeleton_dir=data_dir / "skeletonized",
        polyline_dir=data_dir / "tmp",
        signatures_dir=data_dir / "signatures",
        signature_csv=data_dir / "signatures" / "signatures.csv",
    ).with_timestamped_signature_csv(prefix=args.csv_prefix)

    total = len(mask_paths)
    last_csv: Path | None = None

    for idx, mask_path in enumerate(mask_paths, start=1):
        try:
            result = process_segmented_mask(
                mask_path,
                config=cfg,
                num_samples=args.num_samples,
                depth=args.depth,
                direction=args.directions[0],
                directions=tuple(args.directions),
            )
        except Exception as exc:  # pragma: no cover - CLI convenience
            print(f"[{idx}/{total}] Failed {mask_path.name}: {exc}", file=sys.stderr)
            return 1

        last_csv = result.signature_csv_path
        print(f"[{idx}/{total}] {mask_path.name} -> {result.signature_csv_path.name}")

    destination = last_csv or cfg.signature_csv
    print(f"Processed {total} mask(s). Signatures appended to {destination}.")
    return 0


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "Segmenter",
    "process_segmented_mask",
    "process_with_segmenter",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
