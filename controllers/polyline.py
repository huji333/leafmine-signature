"""Controller helpers for skeleton graph exploration + routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from models.graph_render import render_graph_preview
from models.polyline_utils import compute_polyline_artifacts, render_route_preview
from models.route import RouteResult, compute_route
from models.skeleton_graph import (
    SkeletonGraph,
    build_skeleton_graph,
    prune_short_branches,
)


@dataclass(slots=True)
class PolylineTabConfig:
    """Filesystem layout for the polyline exploration tab."""

    skeleton_dir: Path = Path("data/skeletonized")
    tmp_dir: Path = Path("data/tmp")
    polyline_dir: Path = Path("data/polylines")

    def ensure(self) -> None:
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.polyline_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class GraphSession:
    graph: SkeletonGraph
    skeleton_path: Path
    graph_json_path: Path
    branch_threshold: float
    polyline_dir: Path
    component_map: dict[int, int]


@dataclass(slots=True)
class GraphPrepResult:
    skeleton_image: Image.Image
    pruned_overlay: Image.Image
    graph_payload: dict[str, object]
    graph_json_path: Path
    leaf_nodes: list[dict[str, int]]
    short_edges: list[dict[str, float | int]]
    default_start: int | None
    default_goal: int | None
    status_message: str
    session: GraphSession


@dataclass(slots=True)
class RouteFlowResult:
    route_payload: dict[str, object]
    node_path: list[dict[str, int]]
    message: str
    result: RouteResult
    polyline_payload: dict[str, object]
    polyline_path: Path | None
    route_preview: Image.Image | None


def prepare_graph(
    skeleton_filename: str | Path,
    branch_threshold: float,
    *,
    config: PolylineTabConfig | None = None,
) -> GraphPrepResult:
    cfg = config or PolylineTabConfig()
    cfg.ensure()

    skeleton_path = _resolve_filename(skeleton_filename, cfg.skeleton_dir)
    branch_threshold = max(0.0, float(branch_threshold))

    graph = build_skeleton_graph(skeleton_path)
    pruned_graph = (
        prune_short_branches(graph, branch_threshold)
        if branch_threshold > 0
        else graph
    )

    graph_json_path = cfg.tmp_dir / f"{skeleton_path.stem}_graph.json"
    pruned_graph.save(graph_json_path)

    components = _connected_components(pruned_graph)
    component_map = {
        node_id: index for index, comp in enumerate(components) for node_id in comp
    }

    leaf_nodes = _leaf_metadata(pruned_graph)
    leaf_ids = {leaf["id"] for leaf in leaf_nodes}
    skeleton_image, overlay = _render_images(
        pruned_graph,
        skeleton_path,
        annotate_node_ids=leaf_ids,
    )
    short_edges = _short_edge_summary(pruned_graph, leaf_ids=leaf_ids)

    payload = json.loads(graph_json_path.read_text())
    if not isinstance(payload, dict):
        payload = {"graph": payload}
    payload["metadata"] = {
        "skeleton_png": str(skeleton_path),
        "graph_json": str(graph_json_path),
        "branch_threshold": branch_threshold,
        "node_count": len(pruned_graph.nodes),
        "edge_count": len(pruned_graph.edges),
        "leaf_nodes": len(leaf_nodes),
    }

    defaults = _default_start_goal(pruned_graph, components)
    message = (
        f"Saved graph to {graph_json_path.name}: "
        f"{len(pruned_graph.nodes)} nodes / {len(pruned_graph.edges)} edges"
    )

    session = GraphSession(
        graph=pruned_graph,
        skeleton_path=skeleton_path,
        graph_json_path=graph_json_path,
        branch_threshold=branch_threshold,
        polyline_dir=cfg.polyline_dir,
        component_map=component_map,
    )

    return GraphPrepResult(
        skeleton_image=skeleton_image,
        pruned_overlay=overlay,
        graph_payload=payload,
        graph_json_path=graph_json_path,
        leaf_nodes=leaf_nodes,
        short_edges=short_edges,
        default_start=defaults[0],
        default_goal=defaults[1],
        status_message=message,
        session=session,
    )


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
    }
    stem = session.skeleton_path.stem
    polyline_path = session.polyline_dir / f"{stem}_route.json"

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


def _resolve_filename(filename: str | Path, skeleton_dir: Path) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = skeleton_dir / path.name
    if not path.exists():
        raise FileNotFoundError(f"Could not find skeleton file {path}")
    return path


def _leaf_metadata(
    graph: SkeletonGraph,
) -> list[dict[str, int]]:
    leaves: list[dict[str, int]] = []
    for node_id, node in graph.nodes.items():
        if graph.degree(node_id) == 1:
            leaves.append({"id": node_id, "x": node.x, "y": node.y})
    leaves.sort(key=lambda item: item["id"])
    return leaves


def _short_edge_summary(
    graph: SkeletonGraph,
    *,
    leaf_ids: set[int] | None = None,
    limit: int = 5,
) -> list[dict[str, float | int]]:
    if limit <= 0:
        return []
    records = []
    filtered_leaf_ids = leaf_ids or set()
    for edge in graph.edges.values():
        if filtered_leaf_ids and edge.u not in filtered_leaf_ids and edge.v not in filtered_leaf_ids:
            continue
        records.append(
            {
                "edge_id": edge.id,
                "length": round(float(edge.weight), 2),
                "u": edge.u,
                "v": edge.v,
            }
        )
    records.sort(key=lambda item: item["length"])
    return records[:limit]


def _default_start_goal(
    graph: SkeletonGraph,
    components: list[set[int]],
) -> tuple[int | None, int | None]:
    if not components:
        return None, None
    for comp in components:
        comp_nodes = sorted(comp)
        leaves = [node_id for node_id in comp_nodes if graph.degree(node_id) == 1]
        if len(leaves) >= 2:
            return leaves[0], leaves[1]
        if len(comp_nodes) >= 2:
            return comp_nodes[0], comp_nodes[1]
    first = next(iter(components[0]))
    return first, first


def _connected_components(graph: SkeletonGraph) -> list[set[int]]:
    components: list[set[int]] = []
    visited: set[int] = set()
    for node_id in graph.nodes:
        if node_id in visited:
            continue
        stack = [node_id]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in graph.adjacency.get(current, {}):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)
    return components


def _render_images(
    graph: SkeletonGraph,
    skeleton_path: Path,
    *,
    annotate_node_ids: set[int] | None = None,
) -> tuple[Image.Image, Image.Image]:
    return render_graph_preview(
        graph,
        skeleton_path,
        annotate_nodes=True,
        annotate_node_ids=annotate_node_ids,
        node_radius=5,
        edge_width=3,
        label_color=(255, 255, 0),
    )


__all__ = [
    "PolylineTabConfig",
    "GraphSession",
    "GraphPrepResult",
    "RouteFlowResult",
    "prepare_graph",
    "compute_route_flow",
]
