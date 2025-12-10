from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image
from gradio.data_classes import FileData

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
    segmented_input = gr.File(
        label="Segmentation Mask",
        file_types=["image"],
        file_count="single",
    )
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


def _handle_skeletonization(file_data: FileData | None):
    """Gradio callback to persist inputs, run skeletonize, and persist outputs."""

    if file_data is None:
        raise gr.Error("Please upload a mask image before running skeletonization.")

    _ensure_directories()

    file_path_str = getattr(file_data, "path", None) or getattr(file_data, "name", None)
    if not file_path_str:
        raise gr.Error("Unable to read the uploaded file.")

    file_path = Path(file_path_str)
    upload_name = getattr(file_data, "orig_name", None) or file_path.name

    with Image.open(file_path) as uploaded_image:
        segmented_image = uploaded_image.convert("L")
    segmented_path = _save_image(
        segmented_image,
        SEGMENTED_DIR,
        prefix="segmented",
        original_name=upload_name,
    )

    result = run_skeletonization(segmented_path)
    skeleton_image: Image.Image = result["skeleton_mask"]
    skeleton_path = _save_image(
        skeleton_image,
        SKELETONIZED_DIR,
        prefix="skeleton",
        original_name=upload_name,
    )

    return segmented_image, skeleton_image, {
        "segmented_path": str(segmented_path),
        "skeleton_path": str(skeleton_path),
    }


def _ensure_directories() -> None:
    SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    SKELETONIZED_DIR.mkdir(parents=True, exist_ok=True)


def _save_image(
    image: Image.Image,
    directory: Path,
    prefix: str,
    original_name: str,
) -> Path:
    """Persist the given grayscale image and return the resulting path."""

    base_name = Path(original_name or "upload.png").name
    if Path(base_name).suffix == "":
        base_name = f"{base_name}.png"

    filename = f"{prefix}_{base_name}"
    path = directory / filename

    image.save(path)
    return path


__all__ = ["render"]
