"""Utility helpers for mask cleanup and skeleton pruning."""

from __future__ import annotations

import numpy as np
from skimage.morphology import (
    binary_closing,
    binary_erosion,
    disk,
    remove_small_holes,
)


def preprocess_mask(
    mask: np.ndarray,
    *,
    smooth_radius: int = 2,
    erode_radius: int = 1,
    hole_area_threshold: int = 0,
) -> np.ndarray:
    """Remove small artifacts and gently slim a binary mask before skeletonizing.

    Args:
        mask: Input mask where non-zero pixels represent the foreground.
        smooth_radius: Radius of the disk structuring element used for closing.
        erode_radius: Radius for erosion to pull apart touching regions.
        hole_area_threshold: Fill dark holes smaller than this many pixels.

    Returns:
        A boolean numpy array ready for skeletonization.
    """

    mask_bool = np.asarray(mask).astype(bool, copy=False)
    processed = mask_bool.copy()

    if erode_radius > 0:
        processed = binary_erosion(processed, disk(erode_radius))

    if smooth_radius > 0:
        processed = binary_closing(processed, disk(smooth_radius))

    if hole_area_threshold > 0:
        processed = remove_small_holes(processed, area_threshold=int(hole_area_threshold))

    return processed.astype(bool, copy=False)


__all__ = ["preprocess_mask"]
