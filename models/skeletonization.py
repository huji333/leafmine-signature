"""Skeletonization helpers that wrap preprocessing + pruning passes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "scikit-image is required for skeletonization."
    ) from exc

from .skeletonization_utils import preprocess_mask


@dataclass(slots=True)
class SkeletonizationConfig:
    """Tunable parameters for the skeletonization pipeline."""

    white_threshold: int = 200
    smooth_radius: int = 2
    erode_radius: int = 1
    hole_area_threshold: int = 400


def run_skeletonization(
    mask: Image.Image | str | Path,
    *,
    config: SkeletonizationConfig | None = None,
) -> dict[str, Image.Image]:
    """
    Load a binary mask, run preprocess -> skeletonize, and return images.

    Args:
        mask: a Pillow image or path to the PNG mask (white mine, black background).
        config: optional overrides for the preprocessing/pruning parameters.

    Returns:
        A dictionary containing:
            - ``mask``: the original binary mask as a Pillow image (mode "L").
            - ``preprocessed_mask``: result after smoothing + hole filling.
            - ``skeleton_mask``: the skeletonized mask rendered as a Pillow image.
    """

    cfg = config or SkeletonizationConfig()
    mask_bool = _load_binary_mask(mask, threshold=max(0, min(255, int(cfg.white_threshold))))
    clean_mask = preprocess_mask(
        mask_bool,
        smooth_radius=max(0, int(cfg.smooth_radius)),
        erode_radius=max(0, int(cfg.erode_radius)),
        hole_area_threshold=max(0, int(cfg.hole_area_threshold)),
    )
    skeleton_bool = skeletonize(clean_mask)

    return {
        "mask": _to_image(mask_bool),
        "preprocessed_mask": _to_image(clean_mask),
        "skeleton_mask": _to_image(skeleton_bool),
    }


def _load_binary_mask(
    source: Image.Image | str | Path,
    *,
    threshold: int = 200,
) -> np.ndarray:
    """Load a mask (file path or Pillow image) into a boolean numpy array."""

    if not isinstance(source, Image.Image):
        with Image.open(source) as opened:
            source = opened.convert("RGB")
    if source.mode != "RGB":
        source = source.convert("RGB")

    rgb = np.asarray(source, dtype=np.uint8)
    thresh = np.clip(threshold, 0, 255)
    return np.all(rgb >= thresh, axis=-1)


def _to_image(mask: np.ndarray) -> Image.Image:
    """Convert a boolean mask into a Pillow grayscale image."""

    return Image.fromarray(np.uint8(mask) * 255, mode="L")


__all__ = ["run_skeletonization", "SkeletonizationConfig"]
