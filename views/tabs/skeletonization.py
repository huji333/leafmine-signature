from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from models.skeletonization import run_skeletonization

DATA_DIR = Path(os.environ.get("LEAFMINE_DATA_DIR", Path.cwd() / "data"))
SEGMENTED_DIR = DATA_DIR / "segmented"
SKELETONIZED_DIR = DATA_DIR / "skeletonized"


def render() -> None:
    """Render the skeletonization tab inside a gr.Tab context."""

    gr.Markdown(
        "Upload a **segmentation mask** (white mine on black background). "
        "Running the step will store the uploaded mask under `data/segmented/` "
        "and write the skeletonized result to `data/skeletonized/`."
    )
    segmented_input = gr.Image(label="Segmentation Mask", type="pil")
    run_button = gr.Button("Save & Skeletonize", variant="primary")

    with gr.Row():
        segmented_preview = gr.Image(label="Stored Mask Preview")
        skeleton_preview = gr.Image(label="Skeleton Preview")

    path_info = gr.JSON(label="Saved Paths", value={})

    run_button.click(
        fn=_handle_skeletonization,
        inputs=segmented_input,
        outputs=[segmented_preview, skeleton_preview, path_info],
        show_progress=True,
    )


def _handle_skeletonization(image: Image.Image | None):
    """Gradio callback to persist inputs, run skeletonize, and persist outputs."""

    if image is None:
        raise gr.Error("Please upload a mask image before running skeletonization.")

    _ensure_directories()

    segmented_image = image.convert("L")
    segmented_path = _save_image(segmented_image, SEGMENTED_DIR, prefix="segmented")

    result = run_skeletonization(segmented_path)
    skeleton_image: Image.Image = result["skeleton_mask"]
    skeleton_path = _save_image(skeleton_image, SKELETONIZED_DIR, prefix="skeleton")

    return segmented_image, skeleton_image, {
        "segmented_path": str(segmented_path),
        "skeleton_path": str(skeleton_path),
    }


def _ensure_directories() -> None:
    SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    SKELETONIZED_DIR.mkdir(parents=True, exist_ok=True)


def _save_image(image: Image.Image, directory: Path, prefix: str) -> Path:
    """Persist the given grayscale image and return the resulting path."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{prefix}_{timestamp}.png"
    path = directory / filename
    image.save(path)
    return path


__all__ = ["render"]
