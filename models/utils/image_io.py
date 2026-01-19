"""Thin wrappers around Pillow loading utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TypeVar

import numpy as np
from PIL import Image

ImageSource = TypeVar("ImageSource", Image.Image, str, Path)


def load_image(source: Image.Image | str | Path, *, mode: str | None = None) -> Image.Image:
    """Load an image from ``source`` (path or Image) and optionally convert modes."""

    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        path = Path(source)
        with Image.open(path) as opened:
            image = opened.copy()
    if mode is not None:
        image = image.convert(mode)
    return image


def encode_png(image: Image.Image, *, mode: str | None = None) -> bytes:
    """Return PNG-encoded bytes for the provided Pillow image."""

    buffer = io.BytesIO()
    target = image.convert(mode) if mode is not None else image
    target.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(data: bytes, *, mode: str | None = None) -> Image.Image:
    """Decode PNG bytes into a Pillow image (optionally forcing a mode)."""

    with Image.open(io.BytesIO(data)) as src:
        image = src.convert(mode) if mode is not None else src.copy()
        image.load()
    return image


def save_png(image: Image.Image, destination: Path, *, mode: str | None = None) -> Path:
    """Persist ``image`` as PNG at ``destination``."""

    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = encode_png(image, mode=mode)
    destination.write_bytes(data)
    return destination


def crop_to_foreground(
    image: Image.Image,
    *,
    padding: int = 8,
    min_size: int = 32,
) -> tuple[Image.Image, tuple[int, int]]:
    """Crop image to non-zero pixels and return (cropped, offset)."""

    array = np.asarray(image)
    if array.ndim == 3:
        mask = array.any(axis=2)
    else:
        mask = array > 0
    height, width = mask.shape[:2]
    if not mask.any():
        return image.copy(), (0, 0)

    ys, xs = np.where(mask)
    left = max(0, int(xs.min()) - padding)
    right = min(width - 1, int(xs.max()) + padding)
    top = max(0, int(ys.min()) - padding)
    bottom = min(height - 1, int(ys.max()) + padding)

    crop_width = right - left + 1
    crop_height = bottom - top + 1

    if crop_width < min_size:
        extra = min_size - crop_width
        left = max(0, left - extra // 2)
        right = min(width - 1, left + min_size - 1)
        left = max(0, right - min_size + 1)
    if crop_height < min_size:
        extra = min_size - crop_height
        top = max(0, top - extra // 2)
        bottom = min(height - 1, top + min_size - 1)
        top = max(0, bottom - min_size + 1)

    cropped = image.crop((left, top, right + 1, bottom + 1))
    return cropped, (left, top)


__all__ = [
    "load_image",
    "encode_png",
    "decode_png",
    "save_png",
    "crop_to_foreground",
]
