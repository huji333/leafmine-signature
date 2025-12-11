"""
Minimal helpers for turning a binary mask PNG into skeleton images.

The current scope is intentionally small: read an image, skeletonize it, and
return the original mask image plus the thin skeleton mask so the Gradio tab has
something to display. Polyline extraction and post-processing will live in
follow-up modules once this foundation works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "scikit-image is required for skeletonization."
    ) from exc


def run_skeletonization(mask: Image.Image | str | Path) -> dict[str, Image.Image]:
    """
    Load a binary mask, skeletonize it, and return image-friendly artifacts.

    Args:
        mask: a Pillow image or path to the PNG mask (white mine, black background).

    Returns:
        A dictionary containing:
            - ``mask``: the original binary mask as a Pillow image (mode "L").
            - ``skeleton_mask``: the skeletonized mask rendered as a Pillow image.
    """

    mask_bool = _load_binary_mask(mask)
    skeleton_bool = skeletonize(mask_bool)

    return {
        "mask": _to_image(mask_bool),
        "skeleton_mask": _to_image(skeleton_bool),
    }


def _load_binary_mask(source: Image.Image | str | Path) -> np.ndarray:
    """Load a mask (file path or Pillow image) into a boolean numpy array."""

    if not isinstance(source, Image.Image):
        with Image.open(source) as opened:
            source = opened.convert("L")
    if source.mode != "L":
        source = source.convert("L")

    return np.asarray(source) > 0


def _to_image(mask: np.ndarray) -> Image.Image:
    """Convert a boolean mask into a Pillow grayscale image."""

    return Image.fromarray(np.uint8(mask) * 255, mode="L")


__all__ = ["run_skeletonization"]
