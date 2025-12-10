"""
Minimal helpers for turning a binary mask PNG into skeleton images.

The current scope is intentionally small: read an image, skeletonize it, and
return the original mask image plus the thin skeleton mask so the Gradio tab has
something to display. Polyline extraction and post-processing will live in
follow-up modules once this foundation works end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

try:
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "scikit-image is required for skeletonization."
    ) from exc


def run_skeletonization(image_path: str | Path) -> Dict[str, Image.Image]:
    """
    Load a binary mask, skeletonize it, and return image-friendly artifacts.

    Args:
        image_path: path to the PNG mask (white mine, black background).

    Returns:
        A dictionary containing:
            - ``mask``: the original binary mask as a Pillow image (mode "L").
            - ``skeleton_mask``: the skeletonized mask rendered as a Pillow image.
    """

    mask_bool = _load_binary_mask(image_path)
    skeleton_bool = skeletonize(mask_bool)

    return {
        "mask": _to_image(mask_bool),
        "skeleton_mask": _to_image(skeleton_bool),
    }


def _load_binary_mask(image_path: str | Path) -> np.ndarray:
    """Load a PNG mask into a boolean numpy array."""

    with Image.open(image_path) as image:
        array = np.asarray(image.convert("L"))
    return array > 0


def _to_image(mask: np.ndarray) -> Image.Image:
    """Convert a boolean mask into a Pillow grayscale image."""

    array = np.uint8(mask) * 255
    return Image.fromarray(array, mode="L")


__all__ = ["run_skeletonization"]
