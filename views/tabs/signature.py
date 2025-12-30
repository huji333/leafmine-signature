from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr

from controllers.polyline_signatures import (
    PolylineSignatureConfig,
    analyze_polylines,
)
from views.config import DATA_DIR, build_pipeline_config, list_polylines

PIPELINE_CONFIG = build_pipeline_config()
DATA_BASE = DATA_DIR.expanduser().resolve()
SIGNATURE_CONFIG = PolylineSignatureConfig(
    data_dir=DATA_BASE,
    polyline_dir=PIPELINE_CONFIG.polyline_dir,
    output_dir=PIPELINE_CONFIG.signatures_dir,
    summary_csv=PIPELINE_CONFIG.signature_csv,
)


def render() -> None:
    SIGNATURE_CONFIG.ensure()

    gr.Markdown(
        "Scan stored polylines under `data/polylines/` and append log-signature rows "
        "to the shared CSV (same as `make analyze_polylines`). All polylines are "
        "selected by default—uncheck any you want to skip."
    )

    initial_choices = list_polylines()
    polyline_selector = gr.CheckboxGroup(
        label="Polyline JSON files",
        choices=initial_choices,
        value=initial_choices,
    )
    refresh_button = gr.Button("Refresh polyline list")

    depth_slider = gr.Slider(
        label="Log-signature depth",
        minimum=1,
        maximum=6,
        step=1,
        value=4,
    )
    overwrite_checkbox = gr.Checkbox(
        label="Overwrite existing CSV rows (recompute even if cached)",
        value=False,
    )

    run_button = gr.Button("Compute Signatures", variant="primary")
    csv_preview = gr.Dataframe(
        headers=None,
        label="Signature CSV preview (latest rows)",
        interactive=False,
    )
    status_output = gr.Markdown("")

    refresh_button.click(
        fn=_refresh_polylines,
        inputs=[polyline_selector],
        outputs=[polyline_selector],
    )

    run_button.click(
        fn=_handle_signatures,
        inputs=[polyline_selector, depth_slider, overwrite_checkbox],
        outputs=[csv_preview, status_output],
        show_progress=True,
    )


def _refresh_polylines(current_selection: list[str] | None) -> gr.CheckboxGroup:
    choices = list_polylines()
    if not choices:
        return gr.update(choices=[], value=[])

    current = current_selection or []
    value = [item for item in current if item in choices]
    if not value:
        value = choices
    return gr.update(choices=choices, value=value)


def _handle_signatures(
    selected_files: list[str] | None,
    depth_value: float | int,
    overwrite: bool,
):
    if not selected_files:
        raise gr.Error("Select at least one polyline JSON before running.")

    SIGNATURE_CONFIG.ensure()
    depth = int(depth_value)
    paths = [_resolve_polyline_path(entry) for entry in selected_files]

    results = analyze_polylines(
        paths,
        depth=depth,
        config=SIGNATURE_CONFIG,
        skip_existing=not overwrite,
        summary=True,
    )

    table, headers = _load_csv_preview(SIGNATURE_CONFIG.summary_csv)
    table_update = gr.update(value=table, headers=headers)
    status = (
        f"Wrote {len(results)} new row(s) to `{SIGNATURE_CONFIG.summary_csv.name}` "
        f"(depth={depth})."
    )
    return table_update, status


def _resolve_polyline_path(entry: str) -> Path:
    raw = Path(entry)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((SIGNATURE_CONFIG.polyline_dir / raw.name).resolve())
        candidates.append((Path.cwd() / raw).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise gr.Error(f"Polyline {entry} was not found in {SIGNATURE_CONFIG.polyline_dir}.")


def _load_csv_preview(csv_path: Path, limit: int = 20) -> tuple[list[list[str]], list[str]]:
    if not csv_path.exists():
        return [], []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = reader.fieldnames or []

    if not rows:
        return [], headers

    tail = rows[-limit:]
    cols = headers if headers else list(tail[0].keys())
    table = [[row.get(col, "") for col in cols] for row in tail]
    return table, cols


__all__ = ["render"]
