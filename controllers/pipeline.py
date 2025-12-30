"""Segmentation->log-signature orchestration helpers.

This module implements the end-to-end pipeline (segmented mask -> skeleton ->
route -> log signature).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from .polyline import PolylineTabConfig, auto_route_polyline
from models.utils import apply_stage_prefix, load_image, strip_prefix
from models.signature import (
    LogSignatureResult,
    append_log_signature_csv,
    default_log_signature_csv_path,
    log_signature_from_json,
)
from models.skeletonization import SkeletonizationConfig, run_skeletonization


DEFAULT_SKELETON_CONFIG = SkeletonizationConfig()


@dataclass(slots=True)
class PipelineConfig:
    """File-system destinations for every stage of the pipeline."""

    segmented_dir: Path = Path("data/segmented")
    skeleton_dir: Path = Path("data/skeletonized")
    polyline_dir: Path = Path("data/polylines")
    signatures_dir: Path = Path("data/logsig")
    signature_csv: Path = field(default_factory=default_log_signature_csv_path)
    polyline_branch_threshold: float = 0.0

    def ensure_directories(self) -> None:
        """Create output folders so downstream saves never fail."""

        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.signature_csv.parent.mkdir(parents=True, exist_ok=True)



@dataclass(slots=True)
class PipelineResult:
    """Artifacts and metadata emitted by a successful pipeline run."""

    mask_path: Path
    skeleton_path: Path
    highlight_path: Path
    polyline_path: Path
    signature_csv_path: Path
    signature_results: list[LogSignatureResult]

    @property
    def signature(self) -> LogSignatureResult:
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
    skeleton_config: SkeletonizationConfig | None = None,
    num_samples: int = 256,
    depth: int = 4,
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
        skeleton_config: Optional :class:`SkeletonizationConfig` that tunes
            mask cleanup before skeletonization.
        num_samples: Resampling count for the route polyline (passed through
            to the controllers.polyline helpers).
        depth: Truncation depth for the log signature.

    Returns:
        PipelineResult containing file paths for every artifact plus the
        in-memory :class:`LogSignatureResult`.
    """

    cfg = config or PipelineConfig()
    cfg.ensure_directories()
    skel_cfg = skeleton_config or DEFAULT_SKELETON_CONFIG

    mask_path, sample_base = _resolve_mask_artifact(segmented_mask, cfg, base_name)

    skeleton_artifacts = run_skeletonization(mask_path, config=skel_cfg)
    skeleton_stem = apply_stage_prefix("skeletonized", sample_base)
    skeleton_path = cfg.skeleton_dir / f"{skeleton_stem}.png"
    skeleton_artifacts["skeleton_mask"].save(skeleton_path)

    poly_cfg = PolylineTabConfig(
        skeleton_dir=cfg.skeleton_dir,
        tmp_dir=cfg.polyline_dir,
        polyline_dir=cfg.polyline_dir,
    )
    route_artifacts = auto_route_polyline(
        skeleton_path,
        resample_points=num_samples,
        branch_threshold=cfg.polyline_branch_threshold,
        config=poly_cfg,
        highlight_dir=cfg.polyline_dir,
    )

    log_signature = log_signature_from_json(route_artifacts.polyline_path, depth=depth)
    _ensure_sample_metadata(log_signature, mask_path)
    signature_csv_path = append_log_signature_csv(log_signature, cfg.signature_csv)

    return PipelineResult(
        mask_path=mask_path,
        skeleton_path=skeleton_path,
        highlight_path=route_artifacts.highlight_path,
        polyline_path=route_artifacts.polyline_path,
        signature_csv_path=signature_csv_path,
        signature_results=[log_signature],
    )


def run_pipeline_for_ui(
    mask_file: str | Path,
    *,
    base_name: str | None = None,
    num_samples: float | int,
    depth: float | int,
    config: PipelineConfig | None = None,
    skeleton_config: SkeletonizationConfig | None = None,
) -> PipelineUIResult:
    """Run the pipeline with UI-friendly input validation and output formatting.

    Args:
        mask_file: Path to the segmented mask file.
        base_name: Optional override for artifact filenames.
        num_samples: Resampling count (converted to int).
        depth: Log-signature depth (converted to int).
        config: Optional pipeline configuration.
        skeleton_config: Optional skeletonization config passed through to the models layer.

    Returns:
        PipelineUIResult with formatted image, signature summary, and status message.

    Raises:
        ValueError: If inputs are invalid or the numeric parameters cannot be converted.
    """

    if mask_file is None:
        raise ValueError("Upload a segmented mask first.")
    if num_samples is None or depth is None:
        raise ValueError("Enter numeric values for samples and depth.")

    base = base_name.strip() if base_name else None
    try:
        samples = int(num_samples)
        depth_value = int(depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("Samples and depth must be integers.") from exc

    result = process_segmented_mask(
        mask_file,
        base_name=base,
        config=config,
        skeleton_config=skeleton_config,
        num_samples=samples,
        depth=depth_value,
    )

    with Image.open(result.highlight_path) as highlight_image:
        overlay = highlight_image.copy()

    signature_summary = [
        {
            "sample_filename": signature.sample_filename or result.mask_path.name,
            "depth": signature.depth,
            "dimension": signature.dimension,
            "resample_points": signature.resample_points,
            "path_length": round(signature.path_length, 3),
            "csv_path": str(result.signature_csv_path),
            "polyline_json": str(result.polyline_path),
            "highlight_png": str(result.highlight_path),
            "skeleton_png": str(result.skeleton_path),
            "mask_png": str(result.mask_path),
        }
        for signature in result.signature_results
    ]

    primary = result.signature_results[0]
    status = (
        f"Saved mask `{result.mask_path.name}` → skeleton `{result.skeleton_path.name}` "
        f"→ appended log-signature row in `{result.signature_csv_path.name}`."
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
    skeleton_config: SkeletonizationConfig | None = None,
    num_samples: int = 256,
    depth: int = 4,
) -> PipelineResult:
    """Run an ML/heuristic segmenter first, then finish the pipeline.

    This helper is future-facing: provide any callable that consumes an
    RGB Pillow image and returns a binary mask image, and we will persist
    the mask before delegating to :func:`process_segmented_mask`.
    """

    cfg = config or PipelineConfig()
    cfg.ensure_directories()

    rgb_image = load_image(image, mode="RGB")
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
        skeleton_config=skeleton_config,
        num_samples=num_samples,
        depth=depth,
    )


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
        mask_image = load_image(source_path, mode="L")
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


def _ensure_sample_metadata(result: LogSignatureResult, mask_path: Path) -> None:
    """Populate sample filename metadata on the log-signature result if missing."""

    if not result.sample_filename:
        result.sample_filename = Path(mask_path).name


def _derive_base_name(source: object | None, override: str | None) -> str:
    if override:
        cleaned = Path(str(override)).stem.strip()
        if cleaned:
            return strip_prefix(cleaned)
        raise ValueError("base_name must contain at least one visible character")

    if isinstance(source, (str, Path)):
        stem = Path(source).stem.strip()
        if stem:
            stripped = strip_prefix(stem)
            if stripped:
                return stripped

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _save_mask_image(
    mask_image: Image.Image,
    *,
    cfg: PipelineConfig,
    source: object | None,
    override: str | None,
) -> Path:
    base = _derive_base_name(source, override)
    mask_stem = apply_stage_prefix("segmented", base)
    destination = cfg.segmented_dir / f"{mask_stem}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_image.convert("L").save(destination)
    return destination


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
        help="Root folder containing segmented/, skeletonized/, tmp/, logsig/ (default: ./data or $LEAFMINE_DATA_DIR).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Resample each polyline to N points before computing log signatures (default: 256).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Log-signature truncation depth (default: 4).",
    )
    parser.add_argument(
        "--polyline-branch-threshold",
        type=float,
        default=0.0,
        help="Length threshold (in px) for pruning skeleton branches before routing (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap how many masks to process from the segmented directory.",
    )
    parser.add_argument(
        "--skeleton-white-threshold",
        type=int,
        default=DEFAULT_SKELETON_CONFIG.white_threshold,
        help="RGB cutoff (0-255) for mask foreground pixels before skeletonization (default: 200).",
    )
    parser.add_argument(
        "--skeleton-smooth-radius",
        type=int,
        default=DEFAULT_SKELETON_CONFIG.smooth_radius,
        help="Radius of the closing structuring element before skeletonization (default: 3).",
    )
    parser.add_argument(
        "--skeleton-hole-area",
        type=int,
        default=DEFAULT_SKELETON_CONFIG.hole_area_threshold,
        help="Maximum hole area (px^2) to fill inside the mask before skeletonization (default: 100).",
    )
    parser.add_argument(
        "--skeleton-erode-radius",
        type=int,
        default=DEFAULT_SKELETON_CONFIG.erode_radius,
        help="Erosion radius applied before closing to separate touching regions (default: 4).",
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
        polyline_dir=data_dir / "polylines",
        signatures_dir=data_dir / "logsig",
        signature_csv=default_log_signature_csv_path(data_dir / "logsig"),
        polyline_branch_threshold=args.polyline_branch_threshold,
    )
    print(f"Appending log-signature rows to {cfg.signature_csv}")

    skeleton_cfg = SkeletonizationConfig(
        white_threshold=args.skeleton_white_threshold,
        smooth_radius=args.skeleton_smooth_radius,
        hole_area_threshold=args.skeleton_hole_area,
        erode_radius=args.skeleton_erode_radius,
    )

    total = len(mask_paths)
    last_csv: Path | None = None

    for idx, mask_path in enumerate(mask_paths, start=1):
        try:
            result = process_segmented_mask(
                mask_path,
                config=cfg,
                skeleton_config=skeleton_cfg,
                num_samples=args.num_samples,
                depth=args.depth,
            )
        except Exception as exc:  # pragma: no cover - CLI convenience
            print(f"[{idx}/{total}] Failed {mask_path.name}: {exc}", file=sys.stderr)
            return 1

        last_csv = result.signature_csv_path
        sample_name = result.signature.sample_filename or mask_path.name
        print(f"[{idx}/{total}] {sample_name} -> CSV row {result.signature_csv_path.name}")

    destination = last_csv or cfg.signature_csv
    print(f"Processed {total} mask(s). Log-signature summary appended to {destination}.")
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
