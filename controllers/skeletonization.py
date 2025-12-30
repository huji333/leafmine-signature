"""Controller helpers for the standalone skeletonization tab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from controllers.artifacts import resolve_segmented_mask_path
from controllers.data_paths import DataPaths
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
    data_paths: DataPaths | None = None,
    config: SkeletonizationConfig | None = None,
) -> SkeletonizationResult:
    """Persist a segmented mask and run the preprocessing + skeletonization pipeline."""

    paths = data_paths or DataPaths.from_data_dir()
    paths.ensure_directories()

    mask_gray = mask.convert("L")
    sample_base = _derive_sample_base(original_name)
    mask_path = _save_stage_image(mask_gray, paths.segmented_dir, "segmented", sample_base)

    artifacts = run_skeletonization(mask_gray, config=config)
    skeleton = artifacts["skeleton_mask"]

    skeleton_path = _save_stage_image(
        skeleton,
        paths.skeleton_dir,
        "skeletonized",
        sample_base,
    )

    return SkeletonizationResult(
        mask_image=artifacts["mask"],
        skeleton_image=skeleton,
        mask_path=mask_path,
        skeleton_path=skeleton_path,
    )


def resolve_mask_source(
    data_paths: DataPaths,
    uploaded_file: str | Path | None,
    selected_filename: str | None,
) -> tuple[Image.Image, str]:
    """Return the grayscale mask image to process plus its source name."""

    if uploaded_file:
        upload_path = Path(uploaded_file)
        return _load_grayscale_image(upload_path), upload_path.name

    if selected_filename:
        candidate = resolve_segmented_mask_path(
            selected_filename,
            [data_paths.segmented_dir],
        )
        if candidate is None:
            raise ValueError(
                f"Could not find `{selected_filename}` in {data_paths.segmented_dir}."
            )
        return _load_grayscale_image(candidate), candidate.name

    raise ValueError("Upload a segmented mask or choose an existing filename first.")


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


def _load_grayscale_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as src:
            image = src.convert("L")
            image.load()
            return image
    except FileNotFoundError as exc:
        raise ValueError(f"Mask source {path} was not found.") from exc
    except UnidentifiedImageError as exc:
        raise ValueError(f"{path} is not a valid image file.") from exc


__all__ = [
    "SkeletonizationResult",
    "process_mask",
    "resolve_mask_source",
]
