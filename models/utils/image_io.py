"""Thin wrappers around Pillow loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

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


__all__ = ["load_image"]

