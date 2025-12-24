from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from controllers.pipeline import PipelineConfig, run_pipeline_for_ui
from models.signature import DIRECTION_CHOICES
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
    """Render the single-mask pipeline: segmented binary mask -> signature."""

    gr.Markdown(
        "Upload a pre-segmented binary mask (white mine on black background). "
        "The pipeline skeletonizes it, extracts the longest path, overlays the "
        "yellow polyline, and appends a path signature to the CSV."
    )

    with gr.Row():
        with gr.Column(scale=1):
            mask_input = gr.File(
                label="Segmented Mask (binary)",
                file_types=["image"],
                file_count="single",
            )
            base_name_input = gr.Textbox(
                label="Artifact basename (optional)",
                placeholder="auto-generated if left blank",
            )
            num_samples_input = gr.Number(
                label="Signature samples (N)",
                value=256,
                precision=0,
                minimum=32,
                maximum=4096,
            )
            depth_input = gr.Slider(
                label="Signature depth",
                value=4,
                minimum=1,
                maximum=6,
                step=1,
            )
            direction_input = gr.CheckboxGroup(
                choices=list(DIRECTION_CHOICES),
                value=list(DIRECTION_CHOICES),
                label="Signature directions",
            )
            run_button = gr.Button("Run Pipeline", variant="primary")
        with gr.Column(scale=1):
            highlight_preview = gr.Image(label="Longest Path Overlay")
            signature_preview = gr.JSON(label="Signature Metadata")
            status_output = gr.Markdown(value="")

    run_button.click(
        fn=_run_pipeline,
        inputs=[
            mask_input,
            base_name_input,
            num_samples_input,
            depth_input,
            direction_input,
        ],
        outputs=[highlight_preview, signature_preview, status_output],
        show_progress=True,
    )


def _run_pipeline(
    mask_file: str | None,
    base_name: str,
    num_samples: float,
    depth: float,
    directions: list[str] | None,
):
    """Gradio callback that delegates to the controller layer."""
    try:
        result = run_pipeline_for_ui(
            mask_file,
            base_name=base_name,
            num_samples=num_samples,
            depth=depth,
            directions=directions,
            config=PIPELINE_CONFIG,
            skeleton_config=DEFAULT_SKELETON_CONFIG,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    return result.highlight_image, result.signature_summary, result.status_message


__all__ = ["render"]
