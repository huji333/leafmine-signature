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

Pixel = tuple[int, int]


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
    adjacency: dict[int, dict[int, list[int]]]  # neighbor -> edge_ids

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
            adjacency={
                node_id: {nbr: list(edge_ids) for nbr, edge_ids in neighbors.items()}
                for node_id, neighbors in self.adjacency.items()
            },
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
    nodes, edges, adjacency = _trace_mask_to_graph(mask)
    graph = SkeletonGraph(nodes=nodes, edges=edges, adjacency=adjacency)
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
    # Collapse tiny junction clusters before pruning to avoid 1px artifacts.
    _merge_short_junctions(graph, threshold=min(2.0, max(0.0, branch_threshold)))
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
        neighbor, edge_ids = next(iter(neighbors.items()))
        if not edge_ids:
            continue
        edge = graph.edges.get(edge_ids[0])
        if edge is None:
            continue
        if edge.weight < threshold:
            _remove_edge(graph, edge.id)
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
        edge_ids = neighbors.get(neighbor_id)
        if not edge_ids:
            continue
        if edge_id in edge_ids:
            edge_ids.remove(edge_id)
            if not edge_ids:
                neighbors.pop(neighbor_id, None)
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
        neighbors = graph.adjacency.get(node_id)
        if neighbors is None:
            continue
        empty_neighbors = [nbr for nbr, edge_ids in neighbors.items() if not edge_ids]
        for nbr in empty_neighbors:
            neighbors.pop(nbr, None)
        if not neighbors:
            graph.adjacency.pop(node_id, None)
            graph.nodes.pop(node_id, None)


def _merge_short_junctions(graph: SkeletonGraph, threshold: float) -> None:
    if threshold <= 0 or not graph.nodes or not graph.edges:
        return
    graph.refresh_degrees()
    junctions = {node_id for node_id, node in graph.nodes.items() if node.degree >= 3}
    if len(junctions) < 2:
        return

    parent = {node_id: node_id for node_id in junctions}
    rank = {node_id: 0 for node_id in junctions}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

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
        if edge.weight > threshold:
            continue
        if edge.u in junctions and edge.v in junctions:
            union(edge.u, edge.v)

    groups: dict[int, list[int]] = {}
    for node_id in junctions:
        root = find(node_id)
        groups.setdefault(root, []).append(node_id)

    if all(len(members) == 1 for members in groups.values()):
        return

    merged_nodes = {node_id for members in groups.values() if len(members) > 1 for node_id in members}

    new_id_map: dict[int, int] = {}
    new_nodes: dict[int, GraphNode] = {}
    next_id = 0

    for members in groups.values():
        if len(members) == 1:
            continue
        for member in members:
            new_id_map[member] = next_id
        avg_x = int(round(sum(graph.nodes[m].x for m in members) / len(members)))
        avg_y = int(round(sum(graph.nodes[m].y for m in members) / len(members)))
        new_nodes[next_id] = GraphNode(id=next_id, x=avg_x, y=avg_y)
        next_id += 1

    for node_id, node in graph.nodes.items():
        if node_id in new_id_map:
            continue
        new_id_map[node_id] = next_id
        new_nodes[next_id] = GraphNode(id=next_id, x=node.x, y=node.y)
        next_id += 1

    new_edges: dict[int, GraphEdge] = {}
    new_adj: dict[int, dict[int, list[int]]] = {node_id: {} for node_id in new_nodes}
    for edge in graph.edges.values():
        u_new = new_id_map[edge.u]
        v_new = new_id_map[edge.v]
        if u_new == v_new:
            if edge.u == edge.v and edge.u not in merged_nodes:
                edge_id = len(new_edges)
                new_edges[edge_id] = GraphEdge(
                    id=edge_id,
                    u=u_new,
                    v=v_new,
                    weight=edge.weight,
                    points=edge.points,
                )
                new_adj[u_new].setdefault(v_new, []).append(edge_id)
            continue
        edge_id = len(new_edges)
        new_edges[edge_id] = GraphEdge(
            id=edge_id,
            u=u_new,
            v=v_new,
            weight=edge.weight,
            points=edge.points,
        )
        new_adj[u_new].setdefault(v_new, []).append(edge_id)
        new_adj[v_new].setdefault(u_new, []).append(edge_id)

    graph.nodes = new_nodes
    graph.edges = new_edges
    graph.adjacency = new_adj
    graph.refresh_degrees()


def _find_bridge_edges(graph: SkeletonGraph) -> set[int]:
    """Return edge ids that are bridges in the current undirected graph."""
    incident: dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
    for edge_id, edge in graph.edges.items():
        if edge.u not in incident or edge.v not in incident:
            continue
        incident[edge.u].append(edge_id)
        if edge.v != edge.u:
            incident[edge.v].append(edge_id)

    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[int] = set()
    time = 0

    def dfs(node_id: int, parent_edge: int | None) -> None:
        nonlocal time
        time += 1
        disc[node_id] = time
        low[node_id] = time
        for edge_id in incident.get(node_id, []):
            if edge_id == parent_edge:
                continue
            edge = graph.edges[edge_id]
            if edge.u == edge.v:
                # Self-loops are never bridges.
                continue
            neighbor = edge.other(node_id)
            if neighbor not in disc:
                dfs(neighbor, edge_id)
                low[node_id] = min(low[node_id], low[neighbor])
                if low[neighbor] > disc[node_id]:
                    bridges.add(edge_id)
            else:
                low[node_id] = min(low[node_id], disc[neighbor])

    for node_id in graph.nodes:
        if node_id not in disc:
            dfs(node_id, None)

    return bridges


def _contract_short_internal_edges(graph: SkeletonGraph, threshold: float) -> None:
    if not graph.nodes or not graph.edges:
        return
    graph.refresh_degrees()
    bridge_edges = _find_bridge_edges(graph)
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
        if edge.id not in bridge_edges:
            # Avoid contracting edges that sit on cycles; they encode loops.
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

    edge_entries: list[
        tuple[int, int, float, tuple[tuple[int, int], ...] | None, int]
    ] = []
    for edge in graph.edges.values():
        u_new = new_id_map[edge.u]
        v_new = new_id_map[edge.v]
        if u_new == v_new:
            if edge.u == edge.v:
                edge_entries.append((u_new, v_new, edge.weight, edge.points, edge.id))
            continue
        edge_entries.append((u_new, v_new, edge.weight, edge.points, edge.id))

    edge_entries.sort(
        key=lambda item: (min(item[0], item[1]), max(item[0], item[1]), item[4])
    )

    new_edges: dict[int, GraphEdge] = {}
    new_adj: dict[int, dict[int, list[int]]] = {node_id: {} for node_id in new_nodes}
    for edge_id, (u, v, weight, points, _) in enumerate(edge_entries):
        new_edges[edge_id] = GraphEdge(
            id=edge_id,
            u=u,
            v=v,
            weight=weight,
            points=points,
        )
        if u == v:
            new_adj[u].setdefault(v, []).append(edge_id)
        else:
            new_adj[u].setdefault(v, []).append(edge_id)
            new_adj[v].setdefault(u, []).append(edge_id)

    graph.nodes = new_nodes
    graph.edges = new_edges
    graph.adjacency = new_adj
    graph.refresh_degrees()


def _trace_mask_to_graph(
    mask: np.ndarray,
) -> tuple[dict[int, GraphNode], dict[int, GraphEdge], dict[int, dict[int, list[int]]]]:
    """Trace a skeleton mask into a compact undirected graph with polylines.

    Steps:
      1) Build 8-neighbor connectivity (skip diagonal edges in solid 2x2 blocks).
      2) Collapse connected junction pixels (deg>=3) into a single node each.
      3) Trace degree-2 chains between nodes into edges with polyline points.
      4) Convert pure cycles (no endpoints/junctions) into 3-node loops.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask does not contain foreground pixels.")

    fg: set[Pixel] = {(int(row), int(col)) for row, col in coords}

    def neighbors_of(row: int, col: int) -> list[Pixel]:
        neighbors: list[Pixel] = []
        for d_row, d_col in EIGHT_NEIGHBOR_OFFSETS:
            nr, nc = row + d_row, col + d_col
            if (nr, nc) not in fg:
                continue
            if abs(d_row) == 1 and abs(d_col) == 1:
                # Skip diagonal edges inside solid 2x2 blocks.
                if (row, nc) in fg and (nr, col) in fg:
                    continue
            neighbors.append((nr, nc))
        return neighbors

    neighbor_map: dict[Pixel, list[Pixel]] = {coord: neighbors_of(*coord) for coord in fg}
    degrees: dict[Pixel, int] = {
        coord: len(neighbors) for coord, neighbors in neighbor_map.items()
    }

    # Identify junction/endpoint/isolated pixels.
    junction_pixels = {coord for coord, deg in degrees.items() if deg >= 3}
    endpoint_pixels = {coord for coord, deg in degrees.items() if deg == 1}
    isolated_pixels = {coord for coord, deg in degrees.items() if deg == 0}

    # Collapse connected junction pixels into blobs.
    junction_map: dict[Pixel, int] = {}
    junction_clusters: list[list[Pixel]] = []
    visited_junctions: set[Pixel] = set()

    for coord in sorted(junction_pixels):
        if coord in visited_junctions:
            continue
        cluster: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([coord])
        visited_junctions.add(coord)
        while queue:
            current = queue.popleft()
            cluster.append(current)
            for neighbor in neighbor_map.get(current, []):
                if neighbor in junction_pixels and neighbor not in visited_junctions:
                    visited_junctions.add(neighbor)
                    queue.append(neighbor)
        cluster_id = len(junction_clusters)
        junction_clusters.append(cluster)
        for member in cluster:
            junction_map[member] = cluster_id

    def choose_rep(pixels: list[Pixel]) -> Pixel:
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_c = sum(p[1] for p in pixels) / len(pixels)
        return min(pixels, key=lambda p: (p[0] - avg_r) ** 2 + (p[1] - avg_c) ** 2)

    nodes: dict[int, GraphNode] = {}
    endpoint_node: dict[Pixel, int] = {}
    cluster_node: dict[int, int] = {}
    cluster_rep: dict[int, Pixel] = {}
    cluster_pixels: dict[int, set[Pixel]] = {}
    node_id = 0

    # Create one node per junction blob (use centroid-nearest pixel as node position).
    for cluster_id, pixels in enumerate(junction_clusters):
        rep = choose_rep(pixels)
        cluster_node[cluster_id] = node_id
        cluster_rep[cluster_id] = rep
        cluster_pixels[cluster_id] = set(pixels)
        nodes[node_id] = GraphNode(id=node_id, x=rep[1], y=rep[0])
        node_id += 1

    for coord in sorted(endpoint_pixels):
        if coord in junction_map:
            continue
        endpoint_node[coord] = node_id
        nodes[node_id] = GraphNode(id=node_id, x=coord[1], y=coord[0])
        node_id += 1

    for coord in sorted(isolated_pixels):
        if coord in junction_map or coord in endpoint_node:
            continue
        endpoint_node[coord] = node_id
        nodes[node_id] = GraphNode(id=node_id, x=coord[1], y=coord[0])
        node_id += 1

    def blob_path(blob: set[Pixel], start: Pixel, goal: Pixel) -> list[Pixel]:
        if start == goal:
            return [start]
        queue: deque[Pixel] = deque([start])
        parent: dict[Pixel, Pixel | None] = {start: None}
        while queue:
            current = queue.popleft()
            for neighbor in neighbor_map.get(current, []):
                if neighbor not in blob or neighbor in parent:
                    continue
                parent[neighbor] = current
                if neighbor == goal:
                    queue.clear()
                    break
                queue.append(neighbor)
        if goal not in parent:
            return [start, goal]
        path: list[Pixel] = []
        cursor: Pixel | None = goal
        while cursor is not None:
            path.append(cursor)
            cursor = parent.get(cursor)
        path.reverse()
        return path

    def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
        return (a, b) if a <= b else (b, a)

    visited_edges: set[tuple[Pixel, Pixel]] = set()

    new_edges: dict[int, GraphEdge] = {}
    new_adj: dict[int, dict[int, list[int]]] = {nid: {} for nid in nodes}

    def add_edge(u: int, v: int, points: list[Pixel]) -> None:
        if not points or len(points) < 2:
            return
        weight = 0.0
        for a, b in zip(points, points[1:]):
            weight += math.hypot(b[0] - a[0], b[1] - a[1])
        edge_id = len(new_edges)
        new_edges[edge_id] = GraphEdge(
            id=edge_id,
            u=u,
            v=v,
            weight=weight,
            points=tuple(points),
        )
        new_adj[u].setdefault(v, []).append(edge_id)
        if u != v:
            new_adj[v].setdefault(u, []).append(edge_id)

    def trace_edge(
        start_node_id: int,
        start_pixel: Pixel,
        next_pixel: Pixel,
        start_blob: set[Pixel] | None,
        start_rep: Pixel | None,
    ) -> None:
        path: list[Pixel] = []
        if start_blob is not None and start_rep is not None:
            path.extend(blob_path(start_blob, start_rep, start_pixel))
        else:
            path.append(start_pixel)

        prev = start_pixel
        curr = next_pixel
        visited_edges.add(edge_key(start_pixel, next_pixel))

        end_node_id: int | None = None
        end_blob: set[tuple[int, int]] | None = None
        end_rep: tuple[int, int] | None = None
        end_pixel: tuple[int, int] | None = None

        steps = 0
        max_steps = len(fg) + 1
        while True:
            path.append(curr)
            if curr in endpoint_node and curr != start_pixel:
                end_node_id = endpoint_node[curr]
                end_pixel = curr
                break
            if curr in junction_map:
                cluster_id = junction_map[curr]
                end_node_id = cluster_node[cluster_id]
                end_blob = cluster_pixels[cluster_id]
                end_rep = cluster_rep[cluster_id]
                end_pixel = curr
                break
            neighbors = [n for n in neighbor_map.get(curr, []) if n != prev]
            if not neighbors:
                if curr not in endpoint_node:
                    endpoint_node[curr] = len(nodes)
                    nodes[endpoint_node[curr]] = GraphNode(
                        id=endpoint_node[curr], x=curr[1], y=curr[0]
                    )
                    new_adj[endpoint_node[curr]] = {}
                end_node_id = endpoint_node[curr]
                end_pixel = curr
                break
            if len(neighbors) > 1:
                if curr not in endpoint_node:
                    endpoint_node[curr] = len(nodes)
                    nodes[endpoint_node[curr]] = GraphNode(
                        id=endpoint_node[curr], x=curr[1], y=curr[0]
                    )
                    new_adj[endpoint_node[curr]] = {}
                end_node_id = endpoint_node[curr]
                end_pixel = curr
                break
            nxt = neighbors[0]
            if edge_key(curr, nxt) in visited_edges:
                end_node_id = endpoint_node.get(curr)
                if end_node_id is None:
                    end_node_id = len(nodes)
                    nodes[end_node_id] = GraphNode(id=end_node_id, x=curr[1], y=curr[0])
                    endpoint_node[curr] = end_node_id
                    new_adj[end_node_id] = {}
                end_pixel = curr
                break
            visited_edges.add(edge_key(curr, nxt))
            prev, curr = curr, nxt
            steps += 1
            if steps > max_steps:
                end_node_id = endpoint_node.get(curr)
                if end_node_id is None:
                    end_node_id = len(nodes)
                    nodes[end_node_id] = GraphNode(id=end_node_id, x=curr[1], y=curr[0])
                    endpoint_node[curr] = end_node_id
                    new_adj[end_node_id] = {}
                end_pixel = curr
                break

        if end_node_id is None or end_pixel is None:
            return
        if end_blob is not None and end_rep is not None:
            suffix = blob_path(end_blob, end_pixel, end_rep)
            if suffix:
                path.extend(suffix[1:])

        points = [(p[1], p[0]) for p in path]
        add_edge(start_node_id, end_node_id, points)

    # Trace chains that start from endpoints.
    for coord, node in list(endpoint_node.items()):
        for neighbor in neighbor_map.get(coord, []):
            if edge_key(coord, neighbor) in visited_edges:
                continue
            trace_edge(node, coord, neighbor, None, None)

    # Trace chains that leave junction blobs.
    for cluster_id, pixels in cluster_pixels.items():
        start_node_id = cluster_node[cluster_id]
        rep = cluster_rep[cluster_id]
        for pixel in pixels:
            for neighbor in neighbor_map.get(pixel, []):
                if neighbor in pixels:
                    continue
                if edge_key(pixel, neighbor) in visited_edges:
                    continue
                trace_edge(start_node_id, pixel, neighbor, pixels, rep)

    # Handle pure cycles (no endpoints or junctions) by splitting into 3 nodes/edges.
    non_node_pixels = fg - set(endpoint_node) - set(junction_map)
    visited_component: set[Pixel] = set()

    def split_polyline(
        points: list[tuple[float, float]],
        fractions: list[float],
    ) -> list[list[Pixel]]:
        """Split a polyline by arc-length fractions (returns integer pixel coords)."""
        if len(points) < 2:
            return []
        total = 0.0
        for a, b in zip(points, points[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
        if total == 0.0:
            return []
        targets = [f * total for f in fractions]
        segments: list[list[tuple[float, float]]] = [[points[0]]]
        dist_so_far = 0.0
        target_idx = 0
        for i in range(1, len(points)):
            p0 = points[i - 1]
            p1 = points[i]
            seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            while target_idx < len(targets) and dist_so_far + seg_len >= targets[target_idx]:
                if seg_len == 0:
                    mid = p0
                else:
                    t = (targets[target_idx] - dist_so_far) / seg_len
                    mid = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
                segments[-1].append(mid)
                segments.append([mid])
                target_idx += 1
                p0 = mid
                seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                dist_so_far = targets[target_idx - 1]
            segments[-1].append(p1)
            dist_so_far += seg_len
        while len(segments) < len(fractions) + 1:
            segments.append([points[-1]])
        return [[(int(round(x)), int(round(y))) for x, y in seg] for seg in segments]

    for coord in sorted(non_node_pixels):
        if coord in visited_component:
            continue
        queue: deque[tuple[int, int]] = deque([coord])
        component: set[tuple[int, int]] = set()
        visited_component.add(coord)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in neighbor_map.get(current, []):
                if neighbor in non_node_pixels and neighbor not in visited_component:
                    visited_component.add(neighbor)
                    queue.append(neighbor)
        if not component:
            continue
        # Pure cycle: all degrees == 2, no junction/endpoint pixels.
        if any(degrees.get(p, 0) != 2 for p in component):
            continue
        start = next(iter(component))
        neighbors = neighbor_map.get(start, [])
        if len(neighbors) < 2:
            continue
        prev = start
        curr = neighbors[0]
        cycle: list[Pixel] = [start, curr]
        seen_cycle: set[Pixel] = {start, curr}
        while True:
            next_candidates = [n for n in neighbor_map.get(curr, []) if n != prev]
            if not next_candidates:
                break
            nxt = next_candidates[0]
            cycle.append(nxt)
            if nxt == start:
                break
            if nxt in seen_cycle:
                break
            seen_cycle.add(nxt)
            prev, curr = curr, nxt
        if len(cycle) < 4 or cycle[-1] != start:
            continue
        cycle_xy = [(p[1], p[0]) for p in cycle]
        segments = split_polyline(cycle_xy, [1 / 3, 2 / 3])
        if len(segments) != 3:
            continue
        a = segments[0][0]
        b = segments[0][-1]
        c = segments[1][-1]
        a_id = len(nodes)
        nodes[a_id] = GraphNode(id=a_id, x=a[0], y=a[1])
        new_adj[a_id] = {}
        b_id = a_id + 1
        nodes[b_id] = GraphNode(id=b_id, x=b[0], y=b[1])
        new_adj[b_id] = {}
        c_id = b_id + 1
        nodes[c_id] = GraphNode(id=c_id, x=c[0], y=c[1])
        new_adj[c_id] = {}
        add_edge(a_id, b_id, segments[0])
        add_edge(b_id, c_id, segments[1])
        add_edge(c_id, a_id, segments[2])

    return nodes, new_edges, new_adj


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
