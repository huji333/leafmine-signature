from __future__ import annotations

from collections.abc import Callable
from functools import partial

import gradio as gr

from controllers.artifact_status import ActionType, ProcessingStatusService
from controllers.polyline_graph import GraphSession, prepare_graph
from controllers.polyline_route import compute_route_flow
from controllers.data_paths import DataPaths
from views.components import file_selector
from views.config import DataBrowser, resolve_runtime_paths


def render(
    *,
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> None:
    gr.Markdown(
        "Load a skeleton PNG from `data/skeletonized/`, prune tiny branches, "
        "inspect the resulting graph, and compute an edge traversal order. "
        "You can also type an absolute path if the file lives elsewhere."
    )

    cfg, browser = resolve_runtime_paths(data_paths, data_browser)
    status_service = ProcessingStatusService(cfg)

    with gr.Row():
        skeleton_selector = file_selector(
            label="Skeleton filename",
            choices_provider=browser.skeletonized,
            refresh_label="Refresh skeleton list",
            status_service=status_service,
            action_type=ActionType.ROUTE,
            status_badge="✅ route",
        )
        skeleton_input = skeleton_selector.dropdown
    branch_threshold = gr.Slider(
        label="Branch pruning threshold (px)",
        value=10.0,
        minimum=0.0,
        maximum=300.0,
        step=1.0,
    )
    loop_threshold = gr.Slider(
        label="Loop removal threshold (px)",
        value=10.0,
        minimum=0.0,
        maximum=300.0,
        step=1.0,
    )
    build_button = gr.Button("Build Skeleton Graph", variant="primary")

    graph_state = gr.State()

    status_markdown = gr.Markdown("")
    with gr.Row():
        skeleton_preview = gr.Image(label="Input Preview (segmented or skeleton)")
        pruned_preview = gr.Image(label="Pruned Overlay")

    with gr.Accordion("Graph JSON", open=False):
        graph_json = gr.JSON(label="Graph JSON")
    short_edge_table = gr.Dataframe(
        headers=["edge_id", "u", "v", "length_px"],
        datatype=["number", "number", "number", "number"],
        label="Leaf-connected short edges",
        interactive=False,
    )

    start_node = gr.Number(
        label="Start node id",
        value=None,
        precision=0,
    )
    goal_node = gr.Number(
        label="Goal node id",
        value=None,
        precision=0,
    )
    resample_points = gr.Slider(
        label="Resampled points (polyline)",
        value=256,
        minimum=32,
        maximum=2048,
        step=32,
    )

    route_button = gr.Button("Compute Route", variant="primary")
    route_status = gr.Markdown("")
    route_preview = gr.Image(label="Route Preview (blue→red gradient)")
    route_json = gr.JSON(label="Route summary")
    node_path_json = gr.JSON(label="Node sequence")
    polyline_path_box = gr.Textbox(
        label="Saved polyline JSON",
        interactive=False,
    )

    build_button.click(
        fn=lambda filename, branch, loop: _handle_build_graph(cfg, filename, branch, loop),
        inputs=[skeleton_input, branch_threshold, loop_threshold],
        outputs=[
            skeleton_preview,
            pruned_preview,
            graph_json,
            short_edge_table,
            start_node,
            goal_node,
            graph_state,
            status_markdown,
        ],
        show_progress=True,
    )

    route_button.click(
        fn=partial(
            _handle_compute_route,
            skeleton_selector.choices_provider,
        ),
        inputs=[graph_state, start_node, goal_node, resample_points],
        outputs=[
            route_json,
            node_path_json,
            route_status,
            route_preview,
            polyline_path_box,
            skeleton_input,
        ],
        show_progress=True,
    )


def _handle_build_graph(
    data_paths: DataPaths,
    filename: str | None,
    branch_threshold: float | None,
    loop_threshold: float | None,
) -> tuple:
    if not filename:
        raise gr.Error("Enter the skeleton filename first.")
    branch_value = float(branch_threshold or 0.0)
    loop_value = float(loop_threshold or 0.0)
    try:
        result = prepare_graph(
            filename,
            branch_value,
            loop_threshold=loop_value,
            data_paths=data_paths,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    return _graph_view_payload(result)


def _handle_compute_route(
    choices_provider: Callable[[], list[str | tuple[str, str]]],
    session: GraphSession | None,
    start_value: float | None,
    goal_value: float | None,
    resample_points: float,
) -> tuple:
    if session is None:
        raise gr.Error("Build the graph first.")
    try:
        start = int(start_value)
        goal = int(goal_value) if goal_value is not None else start
        resample = int(resample_points)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Provide integer start/goal node ids.") from exc

    try:
        result = compute_route_flow(
            session,
            start,
            goal,
            resample_points=resample,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    dropdown_update = gr.update(
        choices=choices_provider(),
        value=session.source.skeleton_path.name if session.source else None,
    )
    return (*_route_view_payload(result), dropdown_update)


def _graph_view_payload(result):
    preview = result.segmented_image or result.skeleton_image
    short_edge_rows = [
        [edge["edge_id"], edge["u"], edge["v"], edge["length"]]
        for edge in result.short_edges
    ]
    start_value = result.default_start if result.default_start is not None else None
    goal_value = result.default_goal if result.default_goal is not None else None

    return (
        preview,
        result.pruned_overlay,
        result.graph_payload,
        short_edge_rows,
        start_value,
        goal_value,
        result.session,
        result.status_message,
    )


def _route_view_payload(result):
    return (
        result.route_payload,
        result.node_path,
        result.message,
        result.route_preview,
        str(result.polyline_path) if result.polyline_path else "",
    )


__all__ = ["render"]
