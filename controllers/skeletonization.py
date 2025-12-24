"""Controller helpers for the standalone skeletonization tab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .pipeline import PipelineConfig
from models.skeletonization import (
    SkeletonizationConfig,
    run_skeletonization,
)


@dataclass(slots=True)
class SkeletonizationResult:
    """Artifacts emitted by a skeletonization run."""

    mask_image: Image.Image
    preprocessed_image: Image.Image
    skeleton_image: Image.Image
    mask_path: Path
    preprocessed_path: Path
    skeleton_path: Path


def process_mask(
    mask: Image.Image,
    *,
    original_name: str,
    pipeline_config: PipelineConfig | None = None,
    config: SkeletonizationConfig | None = None,
) -> SkeletonizationResult:
    """Persist a segmented mask and run the preprocessing + skeletonization pipeline."""

    cfg = pipeline_config or PipelineConfig()
    cfg.ensure_directories()

    mask_gray = mask.convert("L")
    mask_path = _save_image(mask_gray, cfg.segmented_dir, "mask", original_name)
    suffix = _mask_suffix(mask_path)

    artifacts = run_skeletonization(mask_gray, config=config)
    preprocessed = artifacts["preprocessed_mask"]
    skeleton = artifacts["skeleton_mask"]

    preprocessed_path = _save_image(
        preprocessed,
        cfg.segmented_dir,
        "preprocessed",
        f"{suffix}.png",
    )
    skeleton_path = _save_image(
        skeleton,
        cfg.skeleton_dir,
        "skeleton",
        f"{suffix}.png",
    )

    return SkeletonizationResult(
        mask_image=artifacts["mask"],
        preprocessed_image=preprocessed,
        skeleton_image=skeleton,
        mask_path=mask_path,
        preprocessed_path=preprocessed_path,
        skeleton_path=skeleton_path,
    )


def _save_image(
    image: Image.Image,
    directory: Path,
    prefix: str,
    original_name: str,
) -> Path:
    base_name = Path(original_name or "upload.png")
    if base_name.suffix == "":
        base_name = base_name.with_suffix(".png")
    filename = f"{prefix}_{base_name.name}"
    destination = directory / filename
    image.save(destination)
    return destination


def _mask_suffix(mask_path: Path) -> str:
    stem = mask_path.stem
    return stem[len("mask_") :] if stem.startswith("mask_") else stem


__all__ = ["SkeletonizationResult", "process_mask"]
