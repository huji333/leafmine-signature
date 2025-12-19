from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image
from controllers.pipeline import PipelineConfig, process_segmented_mask
from models.signature import DIRECTION_CHOICES

DATA_DIR = Path(os.environ.get("LEAFMINE_DATA_DIR", Path.cwd() / "data"))
PIPELINE_CONFIG = PipelineConfig(
    segmented_dir=DATA_DIR / "segmented",
    skeleton_dir=DATA_DIR / "skeletonized",
    polyline_dir=DATA_DIR / "tmp",
    signatures_dir=DATA_DIR / "signatures",
    signature_csv=DATA_DIR / "signatures" / "signatures.csv",
)


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
    if mask_file is None:
        raise gr.Error("Upload a segmented mask first.")

    base = base_name.strip() or None
    selected_dirs = [d for d in (directions or []) if d in DIRECTION_CHOICES]
    if not selected_dirs:
        raise gr.Error("Select at least one signature direction.")

    try:
        result = process_segmented_mask(
            mask_file,
            base_name=base,
            config=PIPELINE_CONFIG,
            num_samples=int(num_samples),
            depth=int(depth),
            direction=DIRECTION_CHOICES[0],
            directions=selected_dirs,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    with Image.open(result.highlight_path) as highlight_image:
        overlay = highlight_image.copy()

    signature_summary = [
        {
            "direction": signature.direction,
            "depth": signature.depth,
            "num_samples": signature.num_samples,
            "signature_dim": signature.dimension,
            "path_points": signature.path_points,
            "path_length": round(signature.path_length, 3),
            "start_xy": [round(signature.start_xy[0], 3), round(signature.start_xy[1], 3)],
            "end_xy": [round(signature.end_xy[0], 3), round(signature.end_xy[1], 3)],
            "csv_path": str(result.signature_csv_path),
            "polyline_json": str(result.polyline_path),
            "highlight_png": str(result.highlight_path),
            "skeleton_png": str(result.skeleton_path),
            "mask_png": str(result.mask_path),
        }
        for signature in result.signature_results
    ]

    status = (
        f"Saved mask `{result.mask_path.name}` → skeleton `{result.skeleton_path.name}` "
        f"→ signature CSV `{result.signature_csv_path.name}` "
        f"({len(result.signature_results)} direction(s))."
    )

    return overlay, signature_summary, status


__all__ = ["render"]
