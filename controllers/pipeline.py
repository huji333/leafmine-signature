"""Segmentation->signature orchestration helpers.

This module implements the end-to-end pipeline (segmented mask -> skeleton ->
longest path -> signature.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

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


@dataclass(slots=True)
class PipelineUIResult:
    """UI-friendly result format for Gradio tabs."""

    highlight_image: Image.Image
    signature_summary: list[dict[str, object]]
    status_message: str


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

    mask_path, original_base = _resolve_mask_artifact(segmented_mask, cfg, base_name)

    skeleton_artifacts = run_skeletonization(mask_path)
    skeleton_stem = _ensure_prefix("skeleton_", original_base)
    skeleton_path = cfg.skeleton_dir / f"{skeleton_stem}.png"
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


def run_pipeline_for_ui(
    mask_file: str | Path,
    *,
    base_name: str | None = None,
    num_samples: float | int,
    depth: float | int,
    directions: Sequence[str] | None = None,
    config: PipelineConfig | None = None,
) -> PipelineUIResult:
    """Run the pipeline with UI-friendly input validation and output formatting.

    This function validates inputs, converts types, runs the pipeline, and formats
    results for Gradio UI consumption. It keeps UI-specific logic out of the views layer.

    Args:
        mask_file: Path to the segmented mask file.
        base_name: Optional override for artifact filenames.
        num_samples: Resampling count (will be converted to int).
        depth: Signature depth (will be converted to int).
        directions: List of direction strings to compute signatures for.
        config: Optional pipeline configuration.

    Returns:
        PipelineUIResult with formatted image, signature summary, and status message.

    Raises:
        ValueError: If inputs are invalid (mask_file is None, directions are empty,
            or num_samples/depth cannot be converted to integers).
    """
    if mask_file is None:
        raise ValueError("Upload a segmented mask first.")

    base = base_name.strip() if base_name else None
    selected_dirs = [d for d in (directions or []) if d in DIRECTION_CHOICES]
    if not selected_dirs:
        raise ValueError("Select at least one signature direction.")

    if num_samples is None or depth is None:
        raise ValueError("Enter numeric values for samples and depth.")

    try:
        samples = int(num_samples)
        depth_value = int(depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("Samples and depth must be integers.") from exc

    result = process_segmented_mask(
        mask_file,
        base_name=base,
        config=config,
        num_samples=samples,
        depth=depth_value,
        direction=DIRECTION_CHOICES[0],
        directions=selected_dirs,
    )

    with Image.open(result.highlight_path) as highlight_image:
        overlay = highlight_image.copy()

    signature_summary = [
        {
            "direction": signature.direction,
            "depth": signature.depth,
            "num_samples": signature.num_samples,
            "signature_dim": signature.dimension,
            "path_points": signature.path_points,
            "path_length": round(signature.path_length, 3),
            "start_xy": [round(signature.start_xy[0], 3), round(signature.start_xy[1], 3)],
            "csv_path": str(result.signature_csv_path),
            "polyline_json": str(result.polyline_path),
            "highlight_png": str(result.highlight_path),
            "skeleton_png": str(result.skeleton_path),
            "mask_png": str(result.mask_path),
        }
        for signature in result.signature_results
    ]

    status = (
        f"Saved mask `{result.mask_path.name}` → skeleton `{result.skeleton_path.name}` "
        f"→ signature CSV `{result.signature_csv_path.name}` "
        f"({len(result.signature_results)} direction(s))."
    )

    return PipelineUIResult(
        highlight_image=overlay,
        signature_summary=signature_summary,
        status_message=status,
    )


def process_with_segmenter(
    image: Image.Image | str | Path,
    segmenter: Callable[[Image.Image], Image.Image],
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

    cfg = config or PipelineConfig()
    cfg.ensure_directories()

    rgb_image = _load_image(image).convert("RGB")
    mask_image = segmenter(rgb_image)
    if not isinstance(mask_image, Image.Image):
        raise TypeError("segmenter must return a Pillow Image")

    mask_path = _save_mask_image(
        mask_image,
        cfg=cfg,
        source=image,
        override=base_name,
    )

    return process_segmented_mask(
        mask_path,
        base_name=base_name,
        config=cfg,
        num_samples=num_samples,
        depth=depth,
        direction=direction,
        directions=directions,
    )


_KNOWN_PREFIXES = ("mask_", "segmented_", "skeleton_", "skeletonized_")


def _resolve_mask_artifact(
    segmented_mask: Image.Image | str | Path,
    cfg: PipelineConfig,
    base_name: str | None,
) -> tuple[Path, str]:
    if isinstance(segmented_mask, Image.Image):
        raise ValueError("Provide a saved mask path when calling process_segmented_mask")

    source_path = Path(segmented_mask)
    if not source_path.exists():
        raise FileNotFoundError(f"Mask source {source_path} does not exist")
    resolved = source_path.resolve()
    segmented_root = cfg.segmented_dir.resolve()
    try:
        resolved.relative_to(segmented_root)
    except ValueError:
        mask_image = _load_image(source_path, mode="L")
        copied = _save_mask_image(
            mask_image,
            cfg=cfg,
            source=source_path,
            override=base_name,
        )
        base = _derive_base_name(copied, None)
        return copied, base

    base = _derive_base_name(source_path, base_name)
    return source_path, base


def _derive_base_name(source: object | None, override: str | None) -> str:
    if override:
        cleaned = Path(str(override)).stem.strip()
        if cleaned:
            return _strip_prefix(cleaned)
        raise ValueError("base_name must contain at least one visible character")

    if isinstance(source, (str, Path)):
        stem = Path(source).stem.strip()
        if stem:
            stripped = _strip_prefix(stem)
            if stripped:
                return stripped

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_prefix(prefix: str, base: str) -> str:
    if base.startswith(prefix):
        return base
    return f"{prefix}{base}"


def _strip_prefix(value: str) -> str:
    for prefix in _KNOWN_PREFIXES:
        if value.startswith(prefix) and len(value) > len(prefix):
            return value[len(prefix) :]
    return value


def _slugify(candidate: str) -> str:
    return candidate.replace(" ", "_").strip(" ._")


def _save_mask_image(
    mask_image: Image.Image,
    *,
    cfg: PipelineConfig,
    source: object | None,
    override: str | None,
) -> Path:
    base = _derive_base_name(source, override)
    mask_stem = _ensure_prefix("mask_", base)
    destination = cfg.segmented_dir / f"{mask_stem}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_image.convert("L").save(destination)
    return destination


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
        return 0

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
    "PipelineUIResult",
    "process_segmented_mask",
    "process_with_segmenter",
    "run_pipeline_for_ui",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
