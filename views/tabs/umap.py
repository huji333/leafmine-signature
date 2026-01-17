from __future__ import annotations

from functools import partial
from pathlib import Path

import gradio as gr

from controllers.data_paths import DataPaths
from controllers.umap import UmapConfig, compute_umap_flow, load_csv_columns
from views.components import file_selector
from views.config import DataBrowser, resolve_runtime_paths


def render(
    *,
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> None:
    cfg, browser = resolve_runtime_paths(data_paths, data_browser)

    gr.Markdown(
        "Load a log-signature CSV from `data/logsig/` and an annotation CSV from "
        "`data/`. Pick a column (e.g., `family`) to color the UMAP plot. "
        "This uses a Random Forest proximity distance before UMAP."
    )

    with gr.Row():
        logsig_selector = file_selector(
            label="Log-signature CSV",
            choices_provider=browser.logsignatures,
            refresh_label="Refresh logsig list",
        )
        annotation_selector = file_selector(
            label="Annotation CSV",
            choices_provider=browser.annotations,
            refresh_label="Refresh annotation list",
        )

    load_columns_button = gr.Button("Load annotation columns")
    columns_status = gr.Markdown("")

    color_column = gr.Dropdown(
        label="Color by (annotation column)",
        choices=[],
        value=None,
    )

    canonicalize = gr.Checkbox(
        label="Match by canonical sample id (strip prefixes/suffix)",
        value=True,
    )

    with gr.Accordion("UMAP settings", open=False):
        with gr.Row():
            n_neighbors = gr.Slider(
                label="UMAP neighbors",
                minimum=5,
                maximum=200,
                step=1,
                value=30,
            )
            min_dist = gr.Slider(
                label="UMAP min_dist",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.05,
            )
            random_state = gr.Number(
                label="Random seed (blank = random)",
                value=0,
                precision=0,
            )

    with gr.Accordion("Random Forest proximity settings", open=False):
        with gr.Row():
            n_estimators = gr.Slider(
                label="RF trees",
                minimum=100,
                maximum=2000,
                step=50,
                value=800,
            )
            min_samples_leaf = gr.Slider(
                label="RF min samples leaf",
                minimum=1,
                maximum=50,
                step=1,
                value=5,
            )
            max_depth = gr.Number(
                label="RF max depth (0 = unlimited)",
                value=0,
                precision=0,
            )
        max_samples = gr.Slider(
            label="Max samples (for speed)",
            minimum=50,
            maximum=5000,
            step=50,
            value=500,
        )

    run_button = gr.Button("Compute UMAP", variant="primary")
    umap_plot = gr.Plot(label="UMAP projection")
    umap_table = gr.Dataframe(
        headers=["sample_key", "label", "umap_x", "umap_y"],
        label="UMAP coordinates (preview)",
        interactive=False,
    )
    status_output = gr.Markdown("")

    load_inputs = [
        logsig_selector.dropdown,
        annotation_selector.dropdown,
        color_column,
    ]
    load_outputs = [color_column, columns_status]

    load_columns_button.click(
        fn=partial(_load_annotation_columns, cfg),
        inputs=load_inputs,
        outputs=load_outputs,
        show_progress=True,
    )
    annotation_selector.dropdown.change(
        fn=partial(_load_annotation_columns, cfg),
        inputs=load_inputs,
        outputs=load_outputs,
        show_progress=True,
    )
    logsig_selector.dropdown.change(
        fn=partial(_load_annotation_columns, cfg),
        inputs=load_inputs,
        outputs=load_outputs,
        show_progress=True,
    )

    run_button.click(
        fn=partial(_handle_umap, cfg),
        inputs=[
            logsig_selector.dropdown,
            annotation_selector.dropdown,
            color_column,
            canonicalize,
            n_neighbors,
            min_dist,
            random_state,
            n_estimators,
            min_samples_leaf,
            max_depth,
            max_samples,
        ],
        outputs=[umap_plot, umap_table, status_output],
        show_progress=True,
    )


def _load_annotation_columns(
    data_paths: DataPaths,
    logsig_entry: str | None,
    annotation_entry: str | None,
    current_color: str | None,
) -> tuple[gr.Dropdown, str]:
    if not logsig_entry or not annotation_entry:
        raise gr.Error("Select both CSV files before loading columns.")

    logsig_path = _resolve_csv_path(logsig_entry, [data_paths.signatures_dir])
    annotation_path = _resolve_csv_path(annotation_entry, [data_paths.annotations_dir])

    if not logsig_path.exists():
        raise gr.Error(f"Log-signature CSV not found: {logsig_path}")
    if not annotation_path.exists():
        raise gr.Error(f"Annotation CSV not found: {annotation_path}")

    try:
        logsig_columns = load_csv_columns(logsig_path)
        annotation_columns = load_csv_columns(annotation_path)
    except FileNotFoundError as exc:
        raise gr.Error(f"CSV not found: {exc.filename}") from exc

    if "filename" not in logsig_columns:
        raise gr.Error("Log-signature CSV must include a 'filename' column.")
    if "sample_id" not in annotation_columns:
        raise gr.Error("Annotation CSV must include a 'sample_id' column.")
    color_value = _pick_color(annotation_columns, current_color, "sample_id")
    color_update = gr.update(choices=annotation_columns, value=color_value)

    status = (
        f"Loaded {len(logsig_columns)} columns from `{logsig_path.name}` "
        f"and {len(annotation_columns)} columns from `{annotation_path.name}`."
    )
    return color_update, status


def _handle_umap(
    data_paths: DataPaths,
    logsig_entry: str | None,
    annotation_entry: str | None,
    color_column: str | None,
    canonicalize: bool,
    n_neighbors: float,
    min_dist: float,
    random_state: float | None,
    n_estimators: float,
    min_samples_leaf: float,
    max_depth: float,
    max_samples: float,
) -> tuple[object, list[list[object]], str]:
    if not color_column:
        raise gr.Error("Select a color column first.")

    config = UmapConfig(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_depth=_parse_max_depth(max_depth),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=_parse_random_state(random_state),
        max_samples=int(max_samples) if max_samples else None,
        canonicalize_keys=bool(canonicalize),
    )

    try:
        result = compute_umap_flow(
            data_paths=data_paths,
            logsig_entry=logsig_entry,
            annotation_entry=annotation_entry,
            color_column=color_column,
            config=config,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    table_update = gr.update(value=result.table_rows, headers=result.table_headers)
    return result.figure, table_update, result.status


def _parse_max_depth(value: float | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return None if parsed <= 0 else parsed


def _parse_random_state(value: float | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return None if parsed < 0 else parsed


def _pick_color(columns: list[str], current: str | None, annotation_key: str | None) -> str | None:
    if not columns:
        return None
    if current in columns:
        return current
    for column in columns:
        if annotation_key and column == annotation_key:
            continue
        return column
    return columns[0]


def _resolve_csv_path(entry: str, directories: list[Path]) -> Path:
    path = Path(entry).expanduser()
    if path.is_absolute():
        return path
    for directory in directories:
        candidate = directory / path
        if candidate.exists():
            return candidate
    return directories[0] / path


__all__ = ["render"]
