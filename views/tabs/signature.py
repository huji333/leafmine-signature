from __future__ import annotations

from functools import partial

import gradio as gr

from controllers.signature import (
    compute_signature_flow,
)
from controllers.data_paths import DataPaths
from views.config import DataBrowser, reconcile_selection, resolve_runtime_paths


def render(
    *,
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> None:
    cfg, browser = resolve_runtime_paths(data_paths, data_browser)

    gr.Markdown(
        "Scan stored polylines under `data/polylines/` and append log-signature rows "
        "to the shared CSV. All polylines are selected by default—uncheck any you want to skip."
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
        value=3,
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
        inputs=[polyline_selector, depth_slider],
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
    value = reconcile_selection(choices, current_selection)
    return gr.update(choices=choices, value=value)


def _handle_signatures(
    data_paths: DataPaths,
    selected_files: list[str] | None,
    depth_value: float | int,
):
    try:
        rows, headers, status = compute_signature_flow(
            data_paths=data_paths,
            selected_files=selected_files,
            depth_value=depth_value,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    table_update = gr.update(value=rows, headers=headers)
    return table_update, status


__all__ = ["render"]
