"""Thin wrappers around Pillow loading utilities."""

from __future__ import annotations

import io
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


__all__ = ["load_image", "encode_png", "decode_png", "save_png"]
