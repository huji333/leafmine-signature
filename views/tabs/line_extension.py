from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from models.longest_component import export_longest_path
from models.signature import signature_from_json, write_signature_csv

DATA_DIR = Path(os.environ.get("LEAFMINE_DATA_DIR", Path.cwd() / "data"))
SKELETONIZED_DIR = DATA_DIR / "skeletonized"
TMP_DIR = DATA_DIR / "tmp"
SIGNATURES_DIR = DATA_DIR / "signatures"
SIGNATURES_CSV = SIGNATURES_DIR / "signatures.csv"
SIGNATURE_DIRECTIONS: tuple[str, ...] = ("forward", "reverse")


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
    with gr.Accordion("Signature Options", open=False):
        num_samples_slider = gr.Slider(
            label="Resampled points (N)",
            minimum=64,
            maximum=512,
            step=32,
            value=256,
        )
        depth_slider = gr.Slider(
            label="Signature depth",
            minimum=2,
            maximum=6,
            step=1,
            value=4,
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
        inputs=[filename_input, signature_checkbox, num_samples_slider, depth_slider],
        outputs=[highlight_preview, polyline_preview, signature_preview, signature_status],
        show_progress=True,
    )


def _handle_longest_path(
    filename: str | None,
    compute_signature: bool,
    num_samples: int,
    depth: int,
):
    if not filename:
        raise gr.Error("Enter the skeleton filename first.")

    _ensure_directories()

    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = SKELETONIZED_DIR / candidate.name

    if not candidate.exists():
        raise gr.Error(f"Could not find {candidate} (did you skeletonize first?).")

    artifacts = export_longest_path(candidate, TMP_DIR)

    with Image.open(artifacts["highlight"]) as highlight_image:
        preview = highlight_image.copy()

    payload = json.loads(artifacts["polyline"].read_text())
    signature_data: list[dict[str, object]] | None = None
    signature_message = ""
    if compute_signature:
        signature_data, signature_message = _compute_signatures(
            artifacts["polyline"], num_samples, depth
        )
    else:
        signature_message = "Signature computation skipped for this run."

    return preview, payload, signature_data, signature_message


def _compute_signatures(
    polyline_path: Path, num_samples: int, depth: int
) -> tuple[list[dict[str, object]], str]:
    results = []
    csv_path: Path | None = None
    for direction in SIGNATURE_DIRECTIONS:
        result = signature_from_json(
            polyline_path,
            num_samples=num_samples,
            depth=depth,
            direction=direction,
        )
        csv_path = write_signature_csv(result, SIGNATURES_CSV)
        results.append(result)

    assert csv_path is not None
    summary = [
        {
            "direction": result.direction,
            "depth": result.depth,
            "num_samples": result.num_samples,
            "signature_dim": result.dimension,
            "path_points": result.path_points,
            "path_length": round(result.path_length, 3),
            "start_xy": [round(result.start_xy[0], 3), round(result.start_xy[1], 3)],
            "end_xy": [round(result.end_xy[0], 3), round(result.end_xy[1], 3)],
            "csv_path": str(csv_path),
        }
        for result in results
    ]
    message = (
        f"Appended {len(results)} signature rows to {csv_path}. "
        "Both directions are included."
    )
    return summary, message


def _ensure_directories() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    SKELETONIZED_DIR.mkdir(parents=True, exist_ok=True)
    SIGNATURES_DIR.mkdir(parents=True, exist_ok=True)


def _example_file_list(limit: int = 6) -> list[str]:
    if not SKELETONIZED_DIR.exists():
        return []
    files = sorted(SKELETONIZED_DIR.glob("*.png"), reverse=True)
    return [file.name for file in files[:limit]]


__all__ = ["render"]
