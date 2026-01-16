from __future__ import annotations

from collections.abc import Callable
from functools import partial

import gradio as gr

from controllers.artifact_status import ActionType, ProcessingStatusService
from controllers.bulk_skeletonization import run_bulk_skeletonization
from controllers.skeletonization import (
    DEFAULT_SKELETON_CONFIG,
    SkeletonizationConfig,
    process_mask,
    render_skeleton_overlay,
    resolve_mask_source,
    summarize_components,
)
from controllers.data_paths import DataPaths
from models.utils.naming import canonical_sample_name, prefixed_name, stage_spec
from views.components import file_selector
from views.config import DataBrowser, reconcile_selection, resolve_runtime_paths

def render(
    *,
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> None:
    """Upload or reuse segmented masks and inspect the resulting skeleton."""

    cfg, browser = resolve_runtime_paths(data_paths, data_browser)
    status_service = ProcessingStatusService(cfg)

    segmented_glob = stage_spec("segmented").glob
    gr.Markdown(
        "Upload a **segmented binary mask** or point to an existing "
        f"`data/segmented/{segmented_glob}`. Uploaded files are persisted with the "
        "canonical `segmented_` prefix automatically. Use the controls below to tune "
        "preprocessing before moving on to routing/log-signatures."
    )

    upload_input = gr.File(
        label="Upload segmented mask",
        file_types=["image"],
        file_count="single",
    )
    with gr.Row():
        segmented_selector = file_selector(
            label="Or pick an existing segmented filename",
            choices_provider=browser.segmented,
            refresh_label="Refresh segmented list",
            status_service=status_service,
            action_type=ActionType.SKELETON,
            status_badge="✅ skeleton",
        )
        existing_selector = segmented_selector.dropdown

    gr.Markdown("### Skeletonization Settings")
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

    run_button = gr.Button("Run skeletonization", variant="primary")

    components_output = gr.Markdown("")

    with gr.Row():
        mask_preview = gr.Image(label="Stored Mask", type="pil")
        skeleton_overlay_preview = gr.Image(
            label="Skeleton Overlay",
            type="pil",
        )

    status_output = gr.Markdown("")

    run_inputs = [
        existing_selector,
        upload_input,
        smooth_radius_input,
        hole_area_input,
        erode_radius_input,
    ]
    run_outputs = [
        components_output,
        mask_preview,
        skeleton_overlay_preview,
        status_output,
        existing_selector,
    ]

    run_button.click(
        fn=partial(
            _handle_skeletonization,
            cfg,
            segmented_selector.choices_provider,
        ),
        inputs=run_inputs,
        outputs=run_outputs,
        show_progress=True,
    )
    # file_selector already wires refresh behavior; no extra click handler needed.

    with gr.Accordion("Bulk Skeletonization", open=False):
        gr.Markdown(
            "Run skeletonization over multiple segmented masks at once. "
            "This uses the same preprocessing settings above."
        )

        bulk_upload = gr.File(
            label="Upload segmented masks (batch)",
            file_types=["image"],
            file_count="multiple",
            type="filepath",
        )

        initial_choices = browser.segmented()
        show_unprocessed_default = True
        if show_unprocessed_default:
            initial_choices = _filter_unprocessed(cfg, initial_choices)
        show_unprocessed = gr.Checkbox(
            label="Show only unprocessed",
            value=show_unprocessed_default,
        )
        bulk_selector = gr.CheckboxGroup(
            label="Segmented mask files",
            choices=initial_choices,
            value=initial_choices,
        )
        bulk_refresh = gr.Button("Refresh segmented list")

        bulk_run = gr.Button("Run Bulk Skeletonization", variant="primary")
        bulk_table = gr.Dataframe(
            headers=None,
            label="Batch results",
            interactive=False,
        )
        bulk_status = gr.Markdown("")

        bulk_refresh.click(
            fn=partial(_refresh_bulk_segmented, cfg, browser),
            inputs=[bulk_selector, show_unprocessed],
            outputs=[bulk_selector],
        )
        show_unprocessed.change(
            fn=partial(_refresh_bulk_segmented, cfg, browser),
            inputs=[bulk_selector, show_unprocessed],
            outputs=[bulk_selector],
        )

        bulk_run.click(
            fn=partial(_handle_bulk, cfg),
            inputs=[
                bulk_selector,
                bulk_upload,
                smooth_radius_input,
                hole_area_input,
                erode_radius_input,
            ],
            outputs=[bulk_table, bulk_status],
            show_progress=True,
        )


def _handle_skeletonization(
    data_paths: DataPaths,
    choices_provider: Callable[[], list[str | tuple[str, str]]],
    selected_filename: str | None,
    uploaded_file: str | None,
    smooth_radius: float | int,
    hole_area: float | int,
    erode_radius: float | int,
):
    """Persist the mask if needed, run skeletonization, and surface diagnostics."""

    data_paths.ensure_directories()
    try:
        mask_image, source_name, existing_mask_path = resolve_mask_source(
            data_paths,
            uploaded_file,
            selected_filename,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    config = SkeletonizationConfig(
        smooth_radius=int(smooth_radius),
        hole_area_threshold=int(hole_area),
        erode_radius=int(erode_radius),
    )

    persist_mask = uploaded_file is not None
    try:
        result = process_mask(
            mask_image,
            original_name=source_name,
            data_paths=data_paths,
            config=config,
            persist_mask=persist_mask,
            existing_mask_path=existing_mask_path,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    overlay = render_skeleton_overlay(result.mask_image, result.skeleton_image)
    component_msg = summarize_components(result.skeleton_image)
    status = (
        f"Saved `{result.mask_path.name}` -> `{result.skeleton_path.name}` "
        f"(closing={config.smooth_radius}, hole<={config.hole_area_threshold}, "
        f"erode={config.erode_radius})."
    )
    dropdown_update = gr.update(
        choices=choices_provider(),
        value=result.mask_path.name,
    )

    return (
        component_msg,
        result.mask_image,
        overlay,
        status,
        dropdown_update,
    )


def _refresh_bulk_segmented(
    data_paths: DataPaths,
    data_browser: DataBrowser,
    current_selection: list[str] | None,
    show_unprocessed: bool,
) -> gr.CheckboxGroup:
    choices = data_browser.segmented()
    if show_unprocessed:
        choices = _filter_unprocessed(data_paths, choices)
    if not choices:
        return gr.update(choices=[], value=[])
    value = reconcile_selection(choices, current_selection)
    return gr.update(choices=choices, value=value)


def _filter_unprocessed(data_paths: DataPaths, choices: list[str]) -> list[str]:
    filtered: list[str] = []
    for entry in choices:
        sample_id = canonical_sample_name(entry)
        skeleton_path = data_paths.skeleton_dir / prefixed_name(
            "skeletonized", sample_id, ".png"
        )
        if not skeleton_path.exists():
            filtered.append(entry)
    return filtered


def _handle_bulk(
    data_paths: DataPaths,
    selected_files: list[str] | None,
    uploaded_files: list[str] | None,
    smooth_radius: float | int,
    hole_area: float | int,
    erode_radius: float | int,
):
    try:
        rows, headers, status = run_bulk_skeletonization(
            data_paths=data_paths,
            selected_files=selected_files,
            uploaded_files=uploaded_files,
            smooth_radius=smooth_radius,
            hole_area=hole_area,
            erode_radius=erode_radius,
            skip_existing_skeleton=True,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    table_update = gr.update(value=rows, headers=headers)
    return table_update, status


__all__ = ["render"]
