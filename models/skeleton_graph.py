from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

EIGHT_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(slots=True)
class GraphNode:
    """Graph vertex that stores its coordinate and cached degree."""

    id: int
    x: int
    y: int
    degree: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"id": self.id, "x": self.x, "y": self.y, "degree": self.degree}


@dataclass(slots=True)
class GraphEdge:
    """Undirected edge between two nodes with an intrinsic length."""

    id: int
    u: int
    v: int
    weight: float
    points: tuple[tuple[int, int], ...] | None = None

    def other(self, node_id: int) -> int:
        if node_id == self.u:
            return self.v
        if node_id == self.v:
            return self.u
        raise ValueError(f"Node {node_id} is not incident to edge {self.id}.")

    def endpoints(self, *, direction: int = 1) -> tuple[int, int]:
        return (self.u, self.v) if direction >= 0 else (self.v, self.u)

    def to_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "id": self.id,
            "source": self.u,
            "target": self.v,
            "weight": float(self.weight),
        }
        return payload


@dataclass(slots=True)
class SkeletonGraph:
    """Compact undirected graph derived from a skeletonized bitmap."""

    nodes: dict[int, GraphNode]
    edges: dict[int, GraphEdge]
    adjacency: dict[int, dict[int, int]]  # neighbor -> edge_id

    def copy(self) -> "SkeletonGraph":
        return SkeletonGraph(
            nodes={
                node_id: GraphNode(node.id, node.x, node.y, node.degree)
                for node_id, node in self.nodes.items()
            },
            edges={
                edge_id: GraphEdge(
                    edge.id,
                    edge.u,
                    edge.v,
                    edge.weight,
                    edge.points,
                )
                for edge_id, edge in self.edges.items()
            },
            adjacency={node_id: dict(neighbors) for node_id, neighbors in self.adjacency.items()},
        )

    def refresh_degrees(self) -> None:
        for node in self.nodes.values():
            node.degree = 0
        for edge in self.edges.values():
            if edge.u == edge.v:
                if edge.u in self.nodes:
                    self.nodes[edge.u].degree += 2
                continue
            if edge.u in self.nodes:
                self.nodes[edge.u].degree += 1
            if edge.v in self.nodes:
                self.nodes[edge.v].degree += 1

    def degree(self, node_id: int) -> int:
        node = self.nodes.get(node_id)
        return node.degree if node else 0

    def to_payload(self) -> dict[str, list[dict[str, float | int]]]:
        node_items = [self.nodes[nid].to_dict() for nid in sorted(self.nodes)]
        edge_items = [self.edges[edge_id].to_dict() for edge_id in sorted(self.edges)]
        return {"nodes": node_items, "edges": edge_items}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_payload(), indent=indent)

    def save(self, output_path: Path) -> Path:
        output_path.write_text(self.to_json())
        return output_path


def build_skeleton_graph(
    skeleton_source: Path | str | Image.Image | np.ndarray,
) -> SkeletonGraph:
    mask = _load_binary_mask(skeleton_source)
    nodes, edges, adjacency = _mask_to_graph(mask)
    c_nodes, c_edges, c_adjacency = _compress_graph(nodes, edges, adjacency)
    graph = SkeletonGraph(nodes=c_nodes, edges=c_edges, adjacency=c_adjacency)
    graph.refresh_degrees()
    return graph


def prune_short_branches(
    graph: SkeletonGraph,
    min_branch_length: float,
    *,
    loop_threshold: float | None = None,
) -> SkeletonGraph:
    result = graph.copy()
    prune_short_branches_inplace(
        result,
        min_branch_length,
        loop_threshold=loop_threshold,
    )
    return result


def prune_short_branches_inplace(
    graph: SkeletonGraph,
    min_branch_length: float,
    *,
    loop_threshold: float | None = None,
) -> None:
    branch_threshold = float(min_branch_length)
    loop_threshold = branch_threshold if loop_threshold is None else float(loop_threshold)
    if branch_threshold <= 0 and (loop_threshold is None or loop_threshold <= 0):
        return
    if branch_threshold > 0:
        _contract_short_internal_edges(graph, branch_threshold)
    while True:
        graph.refresh_degrees()
        removed = False
        if branch_threshold > 0:
            removed |= _trim_short_leaves(graph, branch_threshold)
        if loop_threshold is not None and loop_threshold > 0:
            removed |= _remove_short_loops(graph, loop_threshold)
        _cleanup_isolated_nodes(graph)
        if not removed:
            break


def _trim_short_leaves(graph: SkeletonGraph, threshold: float) -> bool:
    removed = False
    for node_id, node in list(graph.nodes.items()):
        if node.degree != 1:
            continue
        neighbors = graph.adjacency.get(node_id, {})
        if not neighbors:
            continue
        neighbor, edge_id = next(iter(neighbors.items()))
        edge = graph.edges.get(edge_id)
        if edge is None:
            continue
        if edge.weight < threshold:
            _remove_edge(graph, edge_id)
            removed = True
    return removed


def _remove_short_loops(graph: SkeletonGraph, threshold: float) -> bool:
    removed = False
    for edge_id, edge in list(graph.edges.items()):
        if edge.u != edge.v:
            continue
        if edge.weight < threshold:
            _remove_edge(graph, edge_id)
            removed = True
    return removed


def _remove_edge(graph: SkeletonGraph, edge_id: int) -> None:
    edge = graph.edges.pop(edge_id, None)
    if edge is None:
        return
    endpoints = ((edge.u, edge.v), (edge.v, edge.u))
    for node_id, neighbor_id in endpoints:
        neighbors = graph.adjacency.get(node_id)
        if neighbors is None:
            continue
        if neighbor_id in neighbors and neighbors[neighbor_id] == edge_id:
            neighbors.pop(neighbor_id)
    if edge.u == edge.v:
        node = graph.nodes.get(edge.u)
        if node:
            node.degree = max(0, node.degree - 2)
    else:
        for node_id in (edge.u, edge.v):
            node = graph.nodes.get(node_id)
            if node:
                node.degree = max(0, node.degree - 1)


def _cleanup_isolated_nodes(graph: SkeletonGraph) -> None:
    for node_id in list(graph.nodes):
        if not graph.adjacency.get(node_id):
            graph.adjacency.pop(node_id, None)
            graph.nodes.pop(node_id, None)


def _contract_short_internal_edges(graph: SkeletonGraph, threshold: float) -> None:
    if not graph.nodes or not graph.edges:
        return
    graph.refresh_degrees()
    parent = {node_id: node_id for node_id in graph.nodes}
    rank = {node_id: 0 for node_id in graph.nodes}

    def find(node_id: int) -> int:
        if parent[node_id] != node_id:
            parent[node_id] = find(parent[node_id])
        return parent[node_id]

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for edge in graph.edges.values():
        if edge.weight >= threshold:
            continue
        if graph.nodes[edge.u].degree <= 1 or graph.nodes[edge.v].degree <= 1:
            continue
        union(edge.u, edge.v)

    groups: dict[int, list[int]] = {}
    for node_id in graph.nodes:
        root = find(node_id)
        groups.setdefault(root, []).append(node_id)

    if all(len(members) == 1 for members in groups.values()):
        return

    new_id_map: dict[int, int] = {}
    new_nodes: dict[int, GraphNode] = {}
    for new_id, members in enumerate(groups.values()):
        for member in members:
            new_id_map[member] = new_id
        avg_x = int(round(sum(graph.nodes[m].x for m in members) / len(members)))
        avg_y = int(round(sum(graph.nodes[m].y for m in members) / len(members)))
        new_nodes[new_id] = GraphNode(id=new_id, x=avg_x, y=avg_y)

    edge_map: dict[tuple[int, int], tuple[float, tuple[tuple[int, int], ...] | None]] = {}
    for edge in graph.edges.values():
        u_new = new_id_map[edge.u]
        v_new = new_id_map[edge.v]
        if u_new == v_new:
            continue
        key = (min(u_new, v_new), max(u_new, v_new))
        if key in edge_map:
            continue
        edge_map[key] = (edge.weight, edge.points)

    new_edges: dict[int, GraphEdge] = {}
    new_adj: dict[int, dict[int, int]] = {node_id: {} for node_id in new_nodes}
    for edge_id, ((u, v), (weight, points)) in enumerate(sorted(edge_map.items())):
        new_edges[edge_id] = GraphEdge(
            id=edge_id,
            u=u,
            v=v,
            weight=weight,
            points=points,
        )
        new_adj[u][v] = edge_id
        new_adj[v][u] = edge_id

    graph.nodes = new_nodes
    graph.edges = new_edges
    graph.adjacency = new_adj
    graph.refresh_degrees()


def _mask_to_graph(
    mask: np.ndarray,
) -> tuple[dict[int, GraphNode], dict[int, GraphEdge], dict[int, dict[int, int]]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask does not contain foreground pixels.")
    nodes: dict[int, GraphNode] = {}
    edges: dict[int, GraphEdge] = {}
    adjacency: dict[int, dict[int, int]] = {}
    coord_to_id: dict[tuple[int, int], int] = {}
    for idx, (row, col) in enumerate(coords):
        row_i = int(row)
        col_i = int(col)
        nodes[idx] = GraphNode(id=idx, x=col_i, y=row_i)
        adjacency[idx] = {}
        coord_to_id[(row_i, col_i)] = idx
    edge_id = 0
    for (row, col), node_id in coord_to_id.items():
        for d_row, d_col in EIGHT_NEIGHBOR_OFFSETS:
            nr, nc = row + d_row, col + d_col
            neighbor_id = coord_to_id.get((nr, nc))
            if neighbor_id is None or neighbor_id == node_id:
                continue
            if neighbor_id in adjacency[node_id]:
                continue
            weight = math.hypot(d_row, d_col)
            edges[edge_id] = GraphEdge(id=edge_id, u=node_id, v=neighbor_id, weight=weight)
            adjacency[node_id][neighbor_id] = edge_id
            adjacency[neighbor_id][node_id] = edge_id
            edge_id += 1
    return nodes, edges, adjacency


def _compress_graph(
    nodes: dict[int, GraphNode],
    edges: dict[int, GraphEdge],
    adjacency: dict[int, dict[int, int]],
) -> tuple[dict[int, GraphNode], dict[int, GraphEdge], dict[int, dict[int, int]]]:
    if not nodes:
        return {}, {}, {}

    degrees = {node_id: len(neighbors) for node_id, neighbors in adjacency.items()}
    anchors: set[int] = {
        node_id for node_id, deg in degrees.items() if deg != 2 or deg == 0
    }

    if not anchors:
        anchors.add(next(iter(nodes)))

    visited: set[int] = set()
    for node_id in nodes:
        if node_id in visited:
            continue
        component: list[int] = []
        queue: deque[int] = deque([node_id])
        visited.add(node_id)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if not any(member in anchors for member in component):
            anchors.add(component[0])

    anchor_id_map: dict[int, int] = {}
    new_nodes: dict[int, GraphNode] = {}
    for new_id, original_id in enumerate(sorted(anchors)):
        anchor_id_map[original_id] = new_id
        original = nodes[original_id]
        new_nodes[new_id] = GraphNode(id=new_id, x=original.x, y=original.y)

    new_edges: dict[int, GraphEdge] = {}
    new_adj: dict[int, dict[int, int]] = {node_id: {} for node_id in new_nodes}
    visited_edges: set[int] = set()

    for anchor in sorted(anchors):
        for neighbor, edge_id in adjacency.get(anchor, {}).items():
            if edge_id in visited_edges:
                continue
            end_anchor, total_length, path_points = _traverse_chain(
                start=anchor,
                next_node=neighbor,
                initial_edge=edge_id,
                anchors=anchors,
                adjacency=adjacency,
                edges=edges,
                nodes=nodes,
                visited_edges=visited_edges,
            )
            if end_anchor not in anchor_id_map:
                new_id = len(anchor_id_map)
                anchor_id_map[end_anchor] = new_id
                node = nodes[end_anchor]
                new_nodes[new_id] = GraphNode(id=new_id, x=node.x, y=node.y)
                new_adj[new_id] = {}
                anchors.add(end_anchor)
            start_new = anchor_id_map[anchor]
            end_new = anchor_id_map[end_anchor]
            edge_idx = len(new_edges)
            new_edges[edge_idx] = GraphEdge(
                id=edge_idx,
                u=start_new,
                v=end_new,
                weight=total_length,
                points=tuple(path_points),
            )
            new_adj[start_new][end_new] = edge_idx
            new_adj[end_new][start_new] = edge_idx

    return new_nodes, new_edges, new_adj


def _traverse_chain(
    start: int,
    next_node: int,
    initial_edge: int,
    *,
    anchors: set[int],
    adjacency: dict[int, dict[int, int]],
    edges: dict[int, GraphEdge],
    nodes: dict[int, GraphNode],
    visited_edges: set[int],
) -> tuple[int, float, list[tuple[int, int]]]:
    total = edges[initial_edge].weight
    visited_edges.add(initial_edge)
    prev = start
    current = next_node
    points: list[tuple[int, int]] = []

    def _coord(node_id: int) -> tuple[int, int]:
        node = nodes[node_id]
        return (node.x, node.y)

    points.append(_coord(start))
    points.append(_coord(current))

    while current not in anchors:
        neighbors = adjacency.get(current, {})
        next_neighbor = None
        next_edge = None
        for neighbor, edge_id in neighbors.items():
            if neighbor == prev or edge_id in visited_edges:
                continue
            next_neighbor = neighbor
            next_edge = edge_id
            break
        if next_neighbor is None or next_edge is None:
            break
        visited_edges.add(next_edge)
        total += edges[next_edge].weight
        prev = current
        current = next_neighbor
        points.append(_coord(current))

    if points[-1] != _coord(current):
        points.append(_coord(current))

    return current, total, points


def _load_binary_mask(source: Path | str | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        if source.ndim != 2:
            raise ValueError("Expected a 2D array for skeleton mask.")
        return source.astype(bool)
    if not isinstance(source, Image.Image):
        with Image.open(source) as opened:
            source = opened.convert("L")
    if source.mode != "L":
        source = source.convert("L")
    return np.asarray(source) > 0
