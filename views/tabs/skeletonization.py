from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from controllers.pipeline import PipelineConfig
from controllers.skeletonization import process_mask
from models.skeletonization import SkeletonizationConfig

DATA_DIR = Path(os.environ.get("LEAFMINE_DATA_DIR", Path.cwd() / "data"))
PIPELINE_CONFIG = PipelineConfig(
    segmented_dir=DATA_DIR / "segmented",
    skeleton_dir=DATA_DIR / "skeletonized",
    polyline_dir=DATA_DIR / "tmp",
    signatures_dir=DATA_DIR / "signatures",
    signature_csv=DATA_DIR / "signatures" / "signatures.csv",
)
DEFAULT_SKELETON_CONFIG = SkeletonizationConfig()


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
    with gr.Accordion("Skeletonization Settings", open=False):
        smooth_radius_input = gr.Slider(
            minimum=0,
            maximum=10,
            value=DEFAULT_SKELETON_CONFIG.smooth_radius,
            step=1,
            label="Closing Radius (smooth rough edges)",
        )
        hole_area_input = gr.Slider(
            minimum=0,
            maximum=2000,
            value=DEFAULT_SKELETON_CONFIG.hole_area_threshold,
            step=10,
            label="Hole Fill Area (px^2) - fill black holes up to this size",
        )
        erode_radius_input = gr.Slider(
            minimum=0,
            maximum=5,
            value=DEFAULT_SKELETON_CONFIG.erode_radius,
            step=1,
            label="Erosion Radius (separate touching parts)",
        )
    run_button = gr.Button("Skeletonize", variant="primary")

    with gr.Row():
        segmented_preview = gr.Image(label="Stored Mask Preview")
        preprocessed_preview = gr.Image(label="Preprocessed Mask")
        skeleton_preview = gr.Image(label="Skeleton Preview")

    run_button.click(
        fn=_handle_skeletonization,
        inputs=[
            segmented_input,
            smooth_radius_input,
            hole_area_input,
            erode_radius_input,
        ],
        outputs=[
            segmented_preview,
            preprocessed_preview,
            skeleton_preview,
        ],
        show_progress=True,
    )


def _handle_skeletonization(
    file_data: str | None,
    smooth_radius: float | int,
    hole_area: float | int,
    erode_radius: float | int,
):
    """Gradio callback to persist inputs, run skeletonize, and persist outputs."""

    if file_data is None:
        raise gr.Error("Please upload a mask image before running skeletonization.")

    PIPELINE_CONFIG.ensure_directories()

    file_path = Path(file_data)
    upload_name = file_path.name

    with Image.open(file_path) as uploaded_image:
        segmented_image = uploaded_image.convert("L")
    config = SkeletonizationConfig(
        smooth_radius=int(smooth_radius),
        hole_area_threshold=int(hole_area),
        erode_radius=int(erode_radius),
    )

    try:
        result = process_mask(
            segmented_image,
            original_name=upload_name,
            pipeline_config=PIPELINE_CONFIG,
            config=config,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    return (
        result.mask_image,
        result.preprocessed_image,
        result.skeleton_image,
    )


__all__ = ["render"]
