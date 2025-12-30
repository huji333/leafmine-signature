"""Controller helpers for the standalone skeletonization tab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .pipeline import PipelineConfig
from models.utils import apply_stage_prefix, save_png, strip_prefix
from models.skeletonization import (
    SkeletonizationConfig,
    run_skeletonization,
)


@dataclass(slots=True)
class SkeletonizationResult:
    """Artifacts emitted by a skeletonization run."""

    mask_image: Image.Image
    skeleton_image: Image.Image
    mask_path: Path
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
    sample_base = _derive_sample_base(original_name)
    mask_path = _save_stage_image(mask_gray, cfg.segmented_dir, "segmented", sample_base)

    artifacts = run_skeletonization(mask_gray, config=config)
    skeleton = artifacts["skeleton_mask"]

    skeleton_path = _save_stage_image(
        skeleton,
        cfg.skeleton_dir,
        "skeletonized",
        sample_base,
    )

    return SkeletonizationResult(
        mask_image=artifacts["mask"],
        skeleton_image=skeleton,
        mask_path=mask_path,
        skeleton_path=skeleton_path,
    )


def _save_stage_image(
    image: Image.Image,
    directory: Path,
    stage: str,
    sample_base: str,
) -> Path:
    stem = apply_stage_prefix(stage, sample_base)
    destination = directory / f"{stem}.png"
    save_png(image, destination, mode="L")
    return destination


def _derive_sample_base(original_name: str) -> str:
    stem = Path(original_name or "").stem.strip()
    if not stem:
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    return strip_prefix(stem)


__all__ = ["SkeletonizationResult", "process_mask"]
