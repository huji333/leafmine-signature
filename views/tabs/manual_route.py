from __future__ import annotations

import gradio as gr

from controllers.artifact_status import ActionType, ProcessingStatusService
from controllers.data_paths import DataPaths
from controllers.polyline_graph import GraphSession, prepare_graph
from controllers.polyline_route import (
    advance_manual_route,
    finalize_manual_route,
    manual_route_candidates,
    manual_route_node_path,
    render_manual_preview,
    start_manual_route,
    undo_manual_route,
)
from views.components import file_selector
from views.config import DataBrowser, resolve_runtime_paths
from models.utils.image_io import crop_to_foreground


def render(
    *,
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> None:
    gr.Markdown(
        "Build a skeleton graph and step through a manual route by picking the next node. "
        "The output polyline/JSON matches the automatic route builder."
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
    manual_state = gr.State()
    manual_base_image = gr.State()
    manual_crop_offset = gr.State()

    status_markdown = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=3):
            manual_preview = gr.Image(label="Manual Route Preview", height=720)
        with gr.Column(scale=2):
            start_node = gr.Number(
                label="Start node id",
                value=None,
                precision=0,
            )
            manual_start_button = gr.Button("Start Manual Route")
            manual_status = gr.Markdown("Manual route not started.")
            manual_candidates = gr.Dataframe(
                headers=["next_node", "edge_id", "length_px", "edge_visits"],
                datatype=["number", "number", "number", "number"],
                label="Candidate neighbors",
                interactive=False,
            )
            manual_next = gr.Dropdown(label="Next node", choices=[], value=None)
            with gr.Row():
                manual_step_button = gr.Button("Add Step")
                manual_undo_button = gr.Button("Undo")
                manual_reset_button = gr.Button("Reset")
            resample_points = gr.Slider(
                label="Resampled points (polyline)",
                value=256,
                minimum=32,
                maximum=2048,
                step=32,
            )
            manual_finish_button = gr.Button("Finish & Save Polyline", variant="primary")

    manual_node_path = gr.JSON(label="Manual node sequence")

    with gr.Accordion("Graph Details", open=False):
        with gr.Row():
            pruned_preview = gr.Image(label="Pruned Overlay")
            graph_json = gr.JSON(label="Graph JSON")
        short_edge_table = gr.Dataframe(
            headers=["edge_id", "u", "v", "length_px"],
            datatype=["number", "number", "number", "number"],
            label="Leaf-connected short edges",
            interactive=False,
        )

    with gr.Accordion("Saved Route Output", open=False):
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
            manual_preview,
            pruned_preview,
            graph_json,
            short_edge_table,
            start_node,
            graph_state,
            status_markdown,
            manual_base_image,
            manual_crop_offset,
            manual_state,
            manual_candidates,
            manual_next,
            manual_status,
            manual_node_path,
            route_json,
            node_path_json,
            route_status,
            route_preview,
            polyline_path_box,
        ],
        show_progress=True,
    )

    manual_start_button.click(
        fn=_handle_manual_start,
        inputs=[graph_state, start_node, manual_base_image, manual_crop_offset],
        outputs=[
            manual_state,
            manual_candidates,
            manual_next,
            manual_status,
            manual_preview,
            manual_node_path,
        ],
        show_progress=True,
    )

    manual_step_button.click(
        fn=_handle_manual_step,
        inputs=[graph_state, manual_state, manual_next, manual_base_image, manual_crop_offset],
        outputs=[
            manual_state,
            manual_candidates,
            manual_next,
            manual_status,
            manual_preview,
            manual_node_path,
        ],
        show_progress=True,
    )

    manual_undo_button.click(
        fn=_handle_manual_undo,
        inputs=[graph_state, manual_state, manual_base_image, manual_crop_offset],
        outputs=[
            manual_state,
            manual_candidates,
            manual_next,
            manual_status,
            manual_preview,
            manual_node_path,
        ],
        show_progress=True,
    )

    manual_reset_button.click(
        fn=_handle_manual_reset,
        inputs=[graph_state, manual_base_image, manual_crop_offset],
        outputs=[
            manual_state,
            manual_candidates,
            manual_next,
            manual_status,
            manual_preview,
            manual_node_path,
        ],
        show_progress=True,
    )

    manual_finish_button.click(
        fn=_handle_manual_finish,
        inputs=[graph_state, manual_state, resample_points],
        outputs=[
            route_json,
            node_path_json,
            route_status,
            route_preview,
            polyline_path_box,
            manual_status,
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


def _handle_manual_start(
    session: GraphSession | None,
    start_value: float | None,
    base_image,
    offset,
) -> tuple:
    if session is None:
        raise gr.Error("Build the graph first.")
    try:
        start = int(start_value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Provide an integer start node id.") from exc
    try:
        state = start_manual_route(session, start)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return _manual_view_payload(session, state, base_image, offset, "Manual route started.")


def _handle_manual_step(
    session: GraphSession | None,
    state,
    next_value,
    base_image,
    offset,
) -> tuple:
    if session is None:
        raise gr.Error("Build the graph first.")
    if next_value is None:
        raise gr.Error("Pick a next node from the candidate list.")
    try:
        next_edge = int(next_value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Provide a valid next node selection.") from exc
    try:
        updated = advance_manual_route(session, state, next_edge)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    message = f"Moved to node {updated.current} (steps: {len(updated.visits)})."
    return _manual_view_payload(session, updated, base_image, offset, message)


def _handle_manual_undo(
    session: GraphSession | None,
    state,
    base_image,
    offset,
) -> tuple:
    if session is None:
        raise gr.Error("Build the graph first.")
    updated = undo_manual_route(state)
    if updated is None:
        return _handle_manual_reset(session, base_image, offset)
    return _manual_view_payload(session, updated, base_image, offset, "Undid last step.")


def _handle_manual_reset(
    session: GraphSession | None,
    base_image,
    offset,
) -> tuple:
    manual_dropdown = gr.update(choices=[], value=None)
    manual_preview = base_image
    if session is not None and base_image is not None:
        leaf_ids = {
            node_id
            for node_id in session.pruned_graph.nodes
            if session.pruned_graph.degree(node_id) == 1
        }
        manual_preview = render_manual_preview(
            session,
            None,
            base_image,
            offset=offset,
            highlight_ids=leaf_ids,
        )
    return (None, [], manual_dropdown, "Manual route not started.", manual_preview, [])


def _handle_manual_finish(
    session: GraphSession | None,
    state,
    resample_points: float,
) -> tuple:
    if session is None:
        raise gr.Error("Build the graph first.")
    try:
        resample = int(resample_points)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Provide a valid resample point count.") from exc
    try:
        result = finalize_manual_route(
            session,
            state,
            resample_points=resample,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    manual_message = "Manual route saved."
    return (*_route_view_payload(result), manual_message)


def _graph_view_payload(result):
    preview = result.segmented_image or result.skeleton_image
    cropped, offset = crop_to_foreground(preview)
    short_edge_rows = [
        [edge["edge_id"], edge["u"], edge["v"], edge["length"]]
        for edge in result.short_edges
    ]
    start_value = result.default_start if result.default_start is not None else None
    leaf_ids = {
        node_id
        for node_id in result.session.pruned_graph.nodes
        if result.session.pruned_graph.degree(node_id) == 1
    }
    manual_dropdown = gr.update(choices=[], value=None)
    manual_status = "Manual route not started."

    manual_preview = render_manual_preview(
        result.session,
        None,
        cropped,
        offset=offset,
        highlight_ids=leaf_ids,
    )

    return (
        manual_preview,
        result.pruned_overlay,
        result.graph_payload,
        short_edge_rows,
        start_value,
        result.session,
        result.status_message,
        cropped,
        offset,
        None,
        [],
        manual_dropdown,
        manual_status,
        [],
        {},
        [],
        "",
        None,
        "",
    )


def _route_view_payload(result):
    return (
        result.route_payload,
        result.node_path,
        result.message,
        result.route_preview,
        str(result.polyline_path) if result.polyline_path else "",
    )


def _manual_view_payload(
    session: GraphSession,
    state,
    base_image,
    offset,
    status_message: str,
) -> tuple:
    candidates = manual_route_candidates(session, state)
    rows = [
        [row["node_id"], row["edge_id"], row["length"], row["edge_visits"]]
        for row in candidates
    ]
    node_counts: dict[int, int] = {}
    for row in candidates:
        node_id = int(row["node_id"])
        node_counts[node_id] = node_counts.get(node_id, 0) + 1

    choices = []
    for row in candidates:
        node_id = int(row["node_id"])
        suffix = ""
        if node_counts.get(node_id, 0) > 1:
            suffix = f" · edge {row['edge_id']}"
        label = (
            f"{node_id} ({row['length']}px{suffix}"
            f"{' · visited ' + str(row['node_visits']) if row.get('node_visits', 0) else ''})"
        )
        choices.append((label, row["edge_id"]))
    dropdown = gr.update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )
    preview = render_manual_preview(session, state, base_image, offset=offset)
    node_path = manual_route_node_path(session, state)
    return (state, rows, dropdown, status_message, preview, node_path)


__all__ = ["render"]
