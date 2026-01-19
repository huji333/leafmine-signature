"""Route computation helpers for skeleton polylines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from models.graph_render import render_graph_overlay, render_route_overlay
from models.polyline_utils import compute_polyline_artifacts, render_route_preview, route_to_polyline
from models.route import RouteEdgeVisit, RouteResult, compute_route
from models.utils.naming import prefixed_name

from .polyline_graph import GraphSession


@dataclass(slots=True)
class RouteFlowResult:
    route_payload: dict[str, object]
    node_path: list[dict[str, int]]
    message: str
    result: RouteResult
    polyline_payload: dict[str, object]
    polyline_path: Path | None
    route_preview: Image.Image | None


@dataclass(slots=True)
class ManualRouteState:
    start: int
    current: int
    visits: list[RouteEdgeVisit]
    node_path: list[int]


def compute_route_flow(
    session: GraphSession | None,
    start_node: int,
    goal_node: int,
    *,
    resample_points: int,
) -> RouteFlowResult:
    if session is None:
        raise ValueError("Run the graph preparation step first.")

    graph = session.pruned_graph

    if start_node not in graph.nodes:
        raise ValueError(f"Start node {start_node} is not present in the graph.")
    if goal_node not in graph.nodes:
        raise ValueError(f"Goal node {goal_node} is not present in the graph.")

    comp_map = session.component_map
    start_comp = comp_map.get(start_node)
    goal_comp = comp_map.get(goal_node)
    if start_comp is None or goal_comp is None:
        raise ValueError("Could not determine connected components for the selected nodes.")
    if start_comp != goal_comp:
        raise ValueError(
            f"Start node {start_node} (component {start_comp}) and goal node {goal_node} "
            f"(component {goal_comp}) are disconnected. Pick nodes within the same component."
        )

    result = compute_route(graph, start_node, goal_node)
    return _finalize_route_flow(session, result, resample_points=resample_points)


def start_manual_route(session: GraphSession | None, start_node: int) -> ManualRouteState:
    if session is None:
        raise ValueError("Run the graph preparation step first.")
    graph = session.pruned_graph
    if start_node not in graph.nodes:
        raise ValueError(f"Start node {start_node} is not present in the graph.")
    return ManualRouteState(
        start=start_node,
        current=start_node,
        visits=[],
        node_path=[start_node],
    )


def advance_manual_route(
    session: GraphSession | None,
    state: ManualRouteState | None,
    next_edge_id: int,
) -> ManualRouteState:
    if session is None:
        raise ValueError("Run the graph preparation step first.")
    if state is None:
        raise ValueError("Start the manual route first.")
    graph = session.pruned_graph
    edge = graph.edges.get(next_edge_id)
    if edge is None:
        raise ValueError(f"Edge {next_edge_id} is not present in the graph.")
    if state.current not in (edge.u, edge.v):
        raise ValueError(f"Edge {next_edge_id} is not adjacent to {state.current}.")
    next_node = edge.other(state.current)
    direction = 1 if edge.u == state.current and edge.v == next_node else -1
    visits = list(state.visits)
    visits.append(RouteEdgeVisit(edge_id=next_edge_id, direction=direction))
    node_path = list(state.node_path)
    node_path.append(next_node)
    return ManualRouteState(
        start=state.start,
        current=next_node,
        visits=visits,
        node_path=node_path,
    )


def undo_manual_route(state: ManualRouteState | None) -> ManualRouteState | None:
    if state is None:
        return None
    if not state.visits:
        return state
    visits = list(state.visits)
    visits.pop()
    node_path = list(state.node_path)
    node_path.pop()
    current = node_path[-1] if node_path else state.start
    return ManualRouteState(
        start=state.start,
        current=current,
        visits=visits,
        node_path=node_path,
    )


def manual_route_candidates(
    session: GraphSession | None,
    state: ManualRouteState | None,
) -> list[dict[str, int | float]]:
    if session is None or state is None:
        return []
    graph = session.pruned_graph
    counts: dict[int, int] = {}
    for visit in state.visits:
        counts[visit.edge_id] = counts.get(visit.edge_id, 0) + 1
    node_counts: dict[int, int] = {}
    for node_id in state.node_path:
        node_counts[node_id] = node_counts.get(node_id, 0) + 1
    rows: list[dict[str, int | float]] = []
    for neighbor, edge_ids in graph.adjacency.get(state.current, {}).items():
        for edge_id in edge_ids:
            edge = graph.edges[edge_id]
            rows.append(
                {
                    "node_id": neighbor,
                    "edge_id": edge_id,
                    "length": round(float(edge.weight), 2),
                    "edge_visits": counts.get(edge_id, 0),
                    "node_visits": node_counts.get(neighbor, 0),
                }
            )
    rows.sort(
        key=lambda item: (
            int(item["node_visits"]) > 0,
            int(item["node_visits"]),
            int(item["node_id"]),
        )
    )
    return rows


def manual_route_node_path(
    session: GraphSession | None,
    state: ManualRouteState | None,
) -> list[dict[str, int]]:
    if session is None or state is None:
        return []
    graph = session.pruned_graph
    return [
        {"node_id": node_id, "x": graph.nodes[node_id].x, "y": graph.nodes[node_id].y}
        for node_id in state.node_path
    ]


def render_manual_preview(
    session: GraphSession | None,
    state: ManualRouteState | None,
    base_image: Image.Image | None,
    *,
    offset: tuple[int, int] = (0, 0),
    highlight_ids: set[int] | None = None,
) -> Image.Image | None:
    if session is None or base_image is None:
        return None
    graph = session.pruned_graph
    base = base_image.convert("RGB")
    offset_x, offset_y = offset
    if state is not None and state.visits:
        polyline = route_to_polyline(graph, state.visits)
        shifted = [(x - offset_x, y - offset_y) for x, y in polyline]
        base = render_route_overlay(base, shifted)
    candidate_ids: set[int] = set()
    current_ids: set[int] = set()
    if state is not None:
        candidate_rows = manual_route_candidates(session, state)
        candidate_ids = {int(row["node_id"]) for row in candidate_rows}
        current_ids = {state.current}
    elif highlight_ids is not None:
        candidate_ids = set(highlight_ids)
    annotate_ids = candidate_ids | current_ids
    should_label = bool(annotate_ids)
    return render_graph_overlay(
        graph,
        base,
        draw_edges=False,
        offset=offset,
        annotate_nodes=should_label,
        annotate_node_ids=annotate_ids if should_label else None,
        node_radius=3,
        node_color=(200, 200, 200),
        leaf_ids=candidate_ids if candidate_ids else None,
        junction_ids=current_ids if current_ids else None,
        leaf_color=(64, 220, 64),
        junction_color=(255, 96, 96),
        leaf_radius=6,
        junction_radius=7,
        label_color=(255, 255, 255),
        label_stroke_fill=(0, 0, 0),
        label_stroke_width=2,
        label_font_size=18,
        leaf_label_scale=1.0,
        junction_label_scale=1.0,
    )


def finalize_manual_route(
    session: GraphSession | None,
    state: ManualRouteState | None,
    *,
    resample_points: int,
    goal_node: int | None = None,
) -> RouteFlowResult:
    if session is None:
        raise ValueError("Run the graph preparation step first.")
    if state is None:
        raise ValueError("Start the manual route first.")
    if not state.visits:
        raise ValueError("Add at least one edge before saving the manual route.")
    graph = session.pruned_graph
    result = _manual_route_result(graph, state, goal_node=goal_node)
    return _finalize_route_flow(session, result, resample_points=resample_points)


def _manual_route_result(
    graph,
    state: ManualRouteState,
    *,
    goal_node: int | None = None,
) -> RouteResult:
    visits = list(state.visits)
    total_length = sum(graph.edges[v.edge_id].weight for v in visits)
    edge_counts: dict[int, int] = {}
    for visit in visits:
        edge_counts[visit.edge_id] = edge_counts.get(visit.edge_id, 0) + 1
    duplicated_edges = {edge_id: count - 1 for edge_id, count in edge_counts.items() if count > 1}
    current = state.start
    for visit in visits:
        edge = graph.edges[visit.edge_id]
        tail, head = edge.endpoints(direction=visit.direction)
        if tail != current:
            raise ValueError("Manual route sequence is inconsistent with graph geometry.")
        current = head
    if goal_node is not None and current != goal_node:
        raise ValueError(f"Manual route ends at node {current}, not requested goal {goal_node}.")
    return RouteResult(
        start=state.start,
        goal=current,
        visits=visits,
        total_length=total_length,
        duplicated_edges=duplicated_edges,
    )


def _finalize_route_flow(
    session: GraphSession,
    result: RouteResult,
    *,
    resample_points: int,
) -> RouteFlowResult:
    graph = session.pruned_graph
    source = session.source
    duplicated_edges = {
        str(edge_id): count for edge_id, count in sorted(result.duplicated_edges.items())
    }

    route_payload = {
        "start": result.start,
        "goal": result.goal,
        "total_length": round(result.total_length, 3),
        "edge_traversals": len(result.visits),
        "duplicated_edges": duplicated_edges,
        "graph_json": str(session.graph_json_path),
        "branch_threshold": session.branch_threshold,
        "loop_threshold": session.loop_threshold,
        "visits": [
            {
                "edge_id": visit.edge_id,
                "direction": "forward" if visit.direction >= 0 else "reverse",
            }
            for visit in result.visits
        ],
    }

    node_path_ids = result.node_sequence(graph)
    node_path = [
        {
            "node_id": node_id,
            "x": graph.nodes[node_id].x,
            "y": graph.nodes[node_id].y,
        }
        for node_id in node_path_ids
    ]

    duplicates = sum(result.duplicated_edges.values())
    message = (
        f"Route covers {len(result.visits)} edge traversals "
        f"(duplicates: {duplicates})."
    )

    metadata = {
        "graph_json": str(session.graph_json_path),
        "branch_threshold": session.branch_threshold,
        "loop_threshold": session.loop_threshold,
        "start_node": result.start,
        "goal_node": result.goal,
        "resample_points": int(resample_points),
        "sample_base": source.sample_base,
        "segmented_filename": prefixed_name("segmented", source.sample_base, ".png"),
        "skeleton_filename": source.skeleton_path.name,
    }
    polyline_filename = prefixed_name("polyline", source.sample_base, ".json")
    polyline_path = session.polyline_dir / polyline_filename

    artifacts = compute_polyline_artifacts(
        graph,
        result,
        resample_points=resample_points,
        metadata=metadata,
        output_path=polyline_path,
    )

    route_preview = render_route_preview(source.skeleton_path, artifacts.polyline)

    return RouteFlowResult(
        route_payload=route_payload,
        node_path=node_path,
        message=message,
        result=result,
        polyline_payload=artifacts.payload or {},
        polyline_path=polyline_path,
        route_preview=route_preview,
    )


__all__ = [
    "ManualRouteState",
    "RouteFlowResult",
    "advance_manual_route",
    "compute_route_flow",
    "finalize_manual_route",
    "manual_route_candidates",
    "manual_route_node_path",
    "render_manual_preview",
    "start_manual_route",
    "undo_manual_route",
]
