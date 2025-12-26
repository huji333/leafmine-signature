from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from controllers.polyline import (
    GraphPrepResult,
    GraphSession,
    PolylineTabConfig,
    RouteFlowResult,
    compute_route_flow,
    prepare_graph,
)

DATA_DIR = Path(os.environ.get("LEAFMINE_DATA_DIR", Path.cwd() / "data"))
SKELETON_DIR = DATA_DIR / "skeletonized"
TMP_DIR = DATA_DIR / "tmp"

TAB_CONFIG = PolylineTabConfig(skeleton_dir=SKELETON_DIR, tmp_dir=TMP_DIR)


def render() -> None:
    gr.Markdown(
        "Load a skeleton PNG, prune tiny branches, inspect the resulting graph, "
        "and compute an edge traversal order by specifying start / goal nodes."
    )

    skeleton_input = gr.Textbox(
        label="Skeleton filename",
        placeholder="skeleton_YYYYMMDD-HHMMSS.png",
        info="Path under data/skeletonized/ (absolute paths also accepted).",
    )
    branch_threshold = gr.Slider(
        label="Branch/loop pruning threshold (px)",
        value=100.0,
        minimum=0.0,
        maximum=200.0,
        step=1.0,
        info="Remove degree-1 branches or micro loops shorter than this length (0 disables pruning).",
    )
    build_button = gr.Button("Build Skeleton Graph", variant="primary")

    graph_state = gr.State()

    status_markdown = gr.Markdown("")
    with gr.Row():
        skeleton_preview = gr.Image(label="Original Skeleton")
        pruned_preview = gr.Image(label="Pruned Overlay")

    graph_json = gr.JSON(label="Graph JSON")
    leaf_table = gr.Dataframe(
        headers=["node_id", "x", "y"],
        datatype=["number", "number", "number"],
        label="Degree-1 nodes",
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
    route_json = gr.JSON(label="Route summary")
    node_path_json = gr.JSON(label="Node sequence")
    route_status = gr.Markdown("")
    route_preview = gr.Image(label="Route Preview")
    polyline_path_box = gr.Textbox(
        label="Saved polyline JSON",
        interactive=False,
    )

    build_button.click(
        fn=_handle_build_graph,
        inputs=[skeleton_input, branch_threshold],
        outputs=[
            skeleton_preview,
            pruned_preview,
            graph_json,
            leaf_table,
            start_node,
            goal_node,
            graph_state,
            status_markdown,
        ],
        show_progress=True,
    )

    route_button.click(
        fn=_handle_compute_route,
        inputs=[graph_state, start_node, goal_node, resample_points],
        outputs=[route_json, node_path_json, route_status, route_preview, polyline_path_box],
        show_progress=True,
    )


def _handle_build_graph(
    filename: str | None,
    threshold: float | None,
) -> tuple[Any, ...]:
    if not filename:
        raise gr.Error("Enter the skeleton filename first.")
    branch_threshold = float(threshold or 0.0)

    try:
        result = prepare_graph(filename, branch_threshold, config=TAB_CONFIG)
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    leaf_rows = [[leaf["id"], leaf["x"], leaf["y"]] for leaf in result.leaf_nodes]

    start_value = result.default_start if result.default_start is not None else None
    goal_value = result.default_goal if result.default_goal is not None else None

    return (
        result.skeleton_image,
        result.pruned_overlay,
        result.graph_payload,
        leaf_rows,
        start_value,
        goal_value,
        result.session,
        result.status_message,
    )


def _handle_compute_route(
    session: GraphSession | None,
    start_value: float | None,
    goal_value: float | None,
    resample_points: float,
) -> tuple[Any, Any, Any, Any, Any]:
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

    return (
        result.route_payload,
        result.node_path,
        result.message,
        result.route_preview,
        str(result.polyline_path) if result.polyline_path else "",
    )


__all__ = ["render"]
