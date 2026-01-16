"""Controller helpers for the standalone skeletonization tab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from controllers.artifacts import ensure_flat_stage_identifier, resolve_segmented_mask_path
from controllers.data_paths import DataPaths
from models.utils.image_io import save_png
from models.utils.naming import apply_stage_prefix, strip_prefix
from models.skeletonization import SkeletonizationConfig, run_skeletonization

DEFAULT_SKELETON_CONFIG = SkeletonizationConfig()


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
    persist_mask: bool = True,
    existing_mask_path: Path | None = None,
) -> SkeletonizationResult:
    """Persist a segmented mask and run the preprocessing + skeletonization pipeline."""

    paths = data_paths or DataPaths.from_data_dir()
    paths.ensure_directories()

    mask_gray = mask.convert("L")
    sample_base = _derive_sample_base(original_name)
    if persist_mask or existing_mask_path is None:
        mask_path = _save_stage_image(mask_gray, paths.segmented_dir, "segmented", sample_base)
    else:
        mask_path = existing_mask_path

    skeleton_config = config or DEFAULT_SKELETON_CONFIG
    artifacts = run_skeletonization(mask_gray, config=skeleton_config)
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
) -> tuple[Image.Image, str, Path | None]:
    """Return the grayscale mask image to process plus its source name."""

    if uploaded_file:
        upload_path = Path(uploaded_file)
        return _load_grayscale_image(upload_path), upload_path.name, None

    if selected_filename:
        ensure_flat_stage_identifier(
            selected_filename,
            description="Segmented filename",
        )
        candidate = resolve_segmented_mask_path(
            selected_filename,
            [data_paths.segmented_dir],
        )
        if candidate is None:
            raise ValueError(
                f"Could not find `{selected_filename}` in {data_paths.segmented_dir}."
            )
        return _load_grayscale_image(candidate), candidate.name, candidate

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


def render_skeleton_overlay(mask_image: Image.Image, skeleton_image: Image.Image) -> Image.Image:
    """Overlay a thickened skeleton on the mask for quick QA."""

    from PIL import ImageFilter, ImageOps

    base = Image.new("RGBA", mask_image.size, (0, 0, 0, 255))

    mask_l = ImageOps.autocontrast(mask_image)
    mask_alpha = mask_l.point(lambda value: int(180 if value > 0 else 0))
    mask_overlay = Image.new("RGBA", mask_image.size, (180, 180, 180, 0))
    mask_overlay.putalpha(mask_alpha)
    combined = Image.alpha_composite(base, mask_overlay)

    thick = skeleton_image.filter(ImageFilter.MaxFilter(size=5))
    skeleton_alpha = thick.point(lambda value: 255 if value > 0 else 0)
    skeleton_overlay = Image.new("RGBA", mask_image.size, (255, 64, 64, 0))
    skeleton_overlay.putalpha(skeleton_alpha)
    combined = Image.alpha_composite(combined, skeleton_overlay)

    return combined.convert("RGB")


def summarize_components(skeleton_image: Image.Image) -> str:
    """Return a markdown summary of connected components in a skeleton image."""

    import numpy as np
    from skimage.measure import label

    binary = np.asarray(skeleton_image, dtype=np.uint8) > 0
    if not binary.any():
        return "#### Connected components: 0  \n(no skeleton pixels detected)"

    labeled = label(binary, connectivity=2)
    components = int(labeled.max())
    if components <= 1:
        return "#### Connected components: 1  \n(single continuous skeleton)"
    return (
        f"### ⚠️ Connected components: {components}\n"
        "Multiple disjoint branches detected — consider easing preprocessing."
    )


__all__ = [
    "SkeletonizationResult",
    "SkeletonizationConfig",
    "DEFAULT_SKELETON_CONFIG",
    "process_mask",
    "resolve_mask_source",
    "render_skeleton_overlay",
    "summarize_components",
]
