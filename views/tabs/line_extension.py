from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from controllers.line_extension import (
    LineExtensionConfig,
    run_longest_path_flow,
)

DATA_DIR = Path(os.environ.get("LEAFMINE_DATA_DIR", Path.cwd() / "data"))
SKELETONIZED_DIR = DATA_DIR / "skeletonized"
TMP_DIR = DATA_DIR / "tmp"
SIGNATURES_DIR = DATA_DIR / "signatures"
SIGNATURES_CSV = SIGNATURES_DIR / "signatures.csv"

LINE_EXTENSION_CONFIG = LineExtensionConfig(
    skeleton_dir=SKELETONIZED_DIR,
    polyline_dir=TMP_DIR,
    signatures_dir=SIGNATURES_DIR,
    signature_csv=SIGNATURES_CSV,
)


def render() -> None:
    """Render the Line Extension tab that highlights the longest skeleton path."""

    gr.Markdown(
        "Pick a skeleton file saved under `data/skeletonized/` and extract the "
        "longest skeleton segment. The result overlays the segment in red and "
        "stores the polyline metadata for signature analysis."
    )
    filename_input = gr.Textbox(
        label="Skeleton filename",
        placeholder="skeleton_YYYYMMDD-HHMMSS.png",
        info="Enter just the filename (the app looks under data/skeletonized/).",
    )
    run_button = gr.Button("Extract Longest Path", variant="primary")
    signature_checkbox = gr.Checkbox(
        label="Compute and append signatures (forward + reverse)",
        value=True,
    )
    num_samples_input = gr.Number(
        label="Resampled points (N)",
        value=256,
        minimum=32,
        maximum=4096,
        step=1,
    )

    with gr.Row():
        highlight_preview = gr.Image(label="Longest Path Preview")
        polyline_preview = gr.JSON(label="Polyline JSON")
        signature_preview = gr.JSON(label="Signature Metadata (latest run)")
    signature_status = gr.Markdown(value="")

    gr.Examples(
        examples=_example_file_list(),
        inputs=filename_input,
        label="Recent skeletons",
    )

    run_button.click(
        fn=_handle_longest_path,
        inputs=[filename_input, signature_checkbox, num_samples_input],
        outputs=[highlight_preview, polyline_preview, signature_preview, signature_status],
        show_progress=True,
    )


def _handle_longest_path(
    filename: str | None,
    compute_signature: bool,
    num_samples: float,
):
    if not filename:
        raise gr.Error("Enter the skeleton filename first.")

    try:
        result = run_longest_path_flow(
            filename,
            compute_signature=compute_signature,
            num_samples=int(num_samples),
            depth=4,
            config=LINE_EXTENSION_CONFIG,
        )
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    return (
        result.highlight_image,
        result.polyline_payload,
        result.signature_summary,
        result.signature_message,
    )


def _example_file_list(limit: int = 6) -> list[str]:
    if not SKELETONIZED_DIR.exists():
        return []
    files = sorted(SKELETONIZED_DIR.glob("*.png"), reverse=True)
    return [file.name for file in files[:limit]]


__all__ = ["render"]
