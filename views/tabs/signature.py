from __future__ import annotations

import csv
from pathlib import Path
from functools import partial

import gradio as gr

from controllers.polyline_signatures import (
    PolylineSignatureConfig,
    analyze_polylines,
)
from data_paths import DataPaths
from models.signature import default_log_signature_csv_path
from views.config import DataBrowser


def render(
    *,
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> None:
    cfg = data_paths or DataPaths.from_data_dir()
    browser = data_browser or DataBrowser(cfg)

    gr.Markdown(
        "Scan stored polylines under `data/polylines/` and append log-signature rows "
        "to the shared CSV (same as `make analyze_polylines`). All polylines are "
        "selected by default—uncheck any you want to skip."
    )

    initial_choices = browser.polylines()
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
        fn=partial(_refresh_polylines, browser),
        inputs=[polyline_selector],
        outputs=[polyline_selector],
    )

    run_button.click(
        fn=partial(_handle_signatures, cfg),
        inputs=[polyline_selector, depth_slider, overwrite_checkbox],
        outputs=[csv_preview, status_output],
        show_progress=True,
    )


def _refresh_polylines(
    data_browser: DataBrowser,
    current_selection: list[str] | None,
) -> gr.CheckboxGroup:
    choices = data_browser.polylines()
    if not choices:
        return gr.update(choices=[], value=[])

    current = current_selection or []
    value = [item for item in current if item in choices]
    if not value:
        value = choices
    return gr.update(choices=choices, value=value)


def _handle_signatures(
    data_paths: DataPaths,
    selected_files: list[str] | None,
    depth_value: float | int,
    overwrite: bool,
):
    if not selected_files:
        raise gr.Error("Select at least one polyline JSON before running.")

    signature_config = _create_signature_config(data_paths)
    signature_config.ensure()
    depth = int(depth_value)
    paths = [_resolve_polyline_path(signature_config, entry) for entry in selected_files]

    results = analyze_polylines(
        paths,
        depth=depth,
        config=signature_config,
        skip_existing=not overwrite,
        summary=True,
    )

    table, headers = _load_csv_preview(signature_config.summary_csv)
    table_update = gr.update(value=table, headers=headers)
    status = (
        f"Wrote {len(results)} new row(s) to `{signature_config.summary_csv.name}` "
        f"(depth={depth})."
    )
    return table_update, status


def _resolve_polyline_path(
    config: PolylineSignatureConfig,
    entry: str,
) -> Path:
    raw = Path(entry)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((config.polyline_dir / raw.name).resolve())
        candidates.append((Path.cwd() / raw).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise gr.Error(f"Polyline {entry} was not found in {config.polyline_dir}.")


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


def _create_signature_config(data_paths: DataPaths) -> PolylineSignatureConfig:
    summary_csv = default_log_signature_csv_path(data_paths.signatures_dir)
    data_dir = data_paths.segmented_dir.parent
    return PolylineSignatureConfig(
        data_dir=data_dir,
        polyline_dir=data_paths.polyline_dir,
        output_dir=data_paths.signatures_dir,
        summary_csv=summary_csv,
    )


__all__ = ["render"]
