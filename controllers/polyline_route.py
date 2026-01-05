"""Route computation helpers for skeleton polylines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from models.polyline_utils import compute_polyline_artifacts, render_route_preview
from models.route import RouteResult, compute_route
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


def compute_route_flow(
    session: GraphSession | None,
    start_node: int,
    goal_node: int,
    *,
    resample_points: int,
) -> RouteFlowResult:
    if session is None:
        raise ValueError("Run the graph preparation step first.")

    graph = session.graph

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
        "start_node": start_node,
        "goal_node": goal_node,
        "resample_points": int(resample_points),
        "sample_base": session.sample_base,
        "segmented_filename": prefixed_name("segmented", session.sample_base, ".png"),
        "skeleton_filename": session.skeleton_path.name,
    }
    polyline_filename = prefixed_name("polyline", session.sample_base, ".json")
    polyline_path = session.polyline_dir / polyline_filename

    artifacts = compute_polyline_artifacts(
        graph,
        result,
        resample_points=resample_points,
        metadata=metadata,
        output_path=polyline_path,
    )

    route_preview = render_route_preview(session.skeleton_path, artifacts.polyline)

    return RouteFlowResult(
        route_payload=route_payload,
        node_path=node_path,
        message=message,
        result=result,
        polyline_payload=artifacts.payload or {},
        polyline_path=polyline_path,
        route_preview=route_preview,
    )


__all__ = ["RouteFlowResult", "compute_route_flow"]
