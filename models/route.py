from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from .skeleton_graph import SkeletonGraph


@dataclass(slots=True)
class RouteEdgeVisit:
    """Represents traversing an undirected edge in a specific direction."""

    edge_id: int
    direction: int  # +1 for u→v, -1 for v→u


@dataclass(slots=True)
class RouteResult:
    start: int
    goal: int
    visits: list[RouteEdgeVisit]
    total_length: float
    duplicated_edges: dict[int, int]

    def node_sequence(self, graph: SkeletonGraph) -> list[int]:
        nodes: list[int] = [self.start]
        current = self.start
        for visit in self.visits:
            edge = graph.edges[visit.edge_id]
            tail, head = edge.endpoints(direction=visit.direction)
            if tail != current:
                raise ValueError("Route sequence is inconsistent with graph geometry.")
            nodes.append(head)
            current = head
        return nodes


MAX_MATCHING_DP_NODES = 16


def compute_route(
    graph: SkeletonGraph,
    start: int,
    goal: int | None = None,
) -> RouteResult:
    """Compute a Chinese-Postman-like walk that covers every edge at least once."""

    if not graph.nodes:
        raise ValueError("Cannot build a route on an empty graph.")
    if start not in graph.nodes:
        raise ValueError(f"Unknown start node id: {start}")
    if goal is None:
        goal = start
    if goal not in graph.nodes:
        raise ValueError(f"Unknown goal node id: {goal}")

    reachable = _component_nodes(graph, start)
    if goal not in reachable:
        raise ValueError("Goal is not in the same connected component as start.")
    active_nodes = {edge.u for edge in graph.edges.values()} | {
        edge.v for edge in graph.edges.values()
    }
    if reachable != active_nodes:
        missing = active_nodes - reachable
        raise ValueError(
            f"Graph contains {len(missing)} nodes outside the start component."
        )

    target_odds = {node_id for node_id in graph.nodes if graph.degree(node_id) % 2 == 1}
    if start != goal:
        for endpoint in (start, goal):
            if endpoint in target_odds:
                target_odds.remove(endpoint)
            else:
                target_odds.add(endpoint)

    if len(target_odds) % 2 != 0:
        raise ValueError("Odd-degree pairing set has odd cardinality; graph malformed.")

    odd_pairs: list[tuple[int, int]] = []
    edge_additions: dict[int, int] = {}
    if target_odds:
        distances, parents, parent_edges = _all_pairs_shortest(graph, target_odds)
        odd_pairs = _pair_odd_vertices(sorted(target_odds), distances)
        edge_additions = _expand_pairs_to_edges(graph, odd_pairs, parents, parent_edges)

    edge_use_counts: dict[int, int] = {edge_id: 1 for edge_id in graph.edges}
    for edge_id, count in edge_additions.items():
        edge_use_counts[edge_id] = edge_use_counts.get(edge_id, 0) + count

    visits = _hierholzer_trail(graph, start, edge_use_counts)
    final_node = _check_route_end(graph, start, visits)
    if final_node != goal:
        raise ValueError(
            f"Constructed walk terminates at {final_node}, not requested goal {goal}."
        )

    total_length = sum(graph.edges[v.edge_id].weight for v in visits)
    duplicated_edges = {
        edge_id: count - 1 for edge_id, count in edge_use_counts.items() if count > 1
    }

    return RouteResult(
        start=start,
        goal=goal,
        visits=visits,
        total_length=total_length,
        duplicated_edges=duplicated_edges,
    )


def _component_nodes(graph: SkeletonGraph, root: int) -> set[int]:
    stack = [root]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.adjacency.get(node, {}).keys())
    return seen


def _iter_adjacent_edges(
    graph: SkeletonGraph, node_id: int
) -> Iterable[tuple[int, int]]:
    for neighbor, edge_ids in graph.adjacency.get(node_id, {}).items():
        for edge_id in edge_ids:
            yield neighbor, edge_id


def _all_pairs_shortest(
    graph: SkeletonGraph, nodes: Iterable[int]
) -> tuple[
    dict[int, dict[int, float]],
    dict[int, dict[int, int]],
    dict[int, dict[int, int]],
]:
    distances: dict[int, dict[int, float]] = {}
    parents: dict[int, dict[int, int]] = {}
    parent_edges: dict[int, dict[int, int]] = {}
    for node in nodes:
        dist, prev, prev_edge = _dijkstra(graph, node)
        distances[node] = dist
        parents[node] = prev
        parent_edges[node] = prev_edge
    return distances, parents, parent_edges


def _dijkstra(
    graph: SkeletonGraph, source: int
) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    dist = {node_id: math.inf for node_id in graph.nodes}
    prev: dict[int, int] = {}
    prev_edge: dict[int, int] = {}
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > dist[node]:
            continue
        for neighbor, edge_id in _iter_adjacent_edges(graph, node):
            edge = graph.edges[edge_id]
            candidate = current_dist + edge.weight
            if candidate < dist[neighbor]:
                dist[neighbor] = candidate
                prev[neighbor] = node
                prev_edge[neighbor] = edge_id
                heapq.heappush(heap, (candidate, neighbor))
    return dist, prev, prev_edge


def _pair_odd_vertices(
    ordered_nodes: Sequence[int],
    distance_lookup: dict[int, dict[int, float]],
) -> list[tuple[int, int]]:
    if not ordered_nodes:
        return []
    if len(ordered_nodes) > MAX_MATCHING_DP_NODES:
        return _greedy_pairs(ordered_nodes, distance_lookup)
    dist_matrix = [
        [0.0 for _ in ordered_nodes] for _ in ordered_nodes
    ]
    for i, a in enumerate(ordered_nodes):
        for j, b in enumerate(ordered_nodes):
            if i == j:
                continue
            dist_matrix[i][j] = distance_lookup[a][b]
    full_mask = (1 << len(ordered_nodes)) - 1

    @lru_cache(maxsize=None)
    def _solve(mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if mask == 0:
            return 0.0, ()
        first_bit = mask & -mask
        first_idx = first_bit.bit_length() - 1
        best = math.inf
        best_pairs: tuple[tuple[int, int], ...] | None = None
        remaining = mask ^ first_bit
        cursor = remaining
        while cursor:
            bit = cursor & -cursor
            j = bit.bit_length() - 1
            new_mask = remaining ^ bit
            pair_cost = dist_matrix[first_idx][j]
            if math.isinf(pair_cost):
                cursor ^= bit
                continue
            rest_cost, rest_pairs = _solve(new_mask)
            total = pair_cost + rest_cost
            if total < best:
                best = total
                best_pairs = rest_pairs + ((first_idx, j),)
            cursor ^= bit
        if best_pairs is None:
            raise ValueError("Unable to pair odd nodes due to disconnected graph.")
        return best, best_pairs

    _, index_pairs = _solve(full_mask)
    return [(ordered_nodes[i], ordered_nodes[j]) for i, j in index_pairs]


def _greedy_pairs(
    ordered_nodes: Sequence[int],
    distance_lookup: dict[int, dict[int, float]],
) -> list[tuple[int, int]]:
    unused = list(ordered_nodes)
    pairs: list[tuple[int, int]] = []
    while unused:
        a = unused.pop()
        best_idx = -1
        best_cost = math.inf
        for idx, candidate in enumerate(unused):
            cost = distance_lookup[a][candidate]
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx == -1 or math.isinf(best_cost):
            raise ValueError("Cannot find a finite path between odd-degree nodes.")
        b = unused.pop(best_idx)
        pairs.append((a, b))
    return pairs


def _expand_pairs_to_edges(
    graph: SkeletonGraph,
    pairs: Sequence[tuple[int, int]],
    parents: dict[int, dict[int, int]],
    parent_edges: dict[int, dict[int, int]],
) -> dict[int, int]:
    edge_counts: dict[int, int] = {}
    for a, b in pairs:
        edge_path = _reconstruct_edge_path(parents[a], parent_edges[a], a, b)
        if not edge_path:
            continue
        for edge_id in edge_path:
            edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1
    return edge_counts


def _reconstruct_edge_path(
    prev: dict[int, int],
    prev_edge: dict[int, int],
    start: int,
    goal: int,
) -> list[int]:
    if start == goal:
        return []
    edges: list[int] = []
    current = goal
    while current != start:
        if current not in prev or current not in prev_edge:
            raise ValueError("Graph is disconnected; path reconstruction failed.")
        edges.append(prev_edge[current])
        current = prev[current]
    edges.reverse()
    return edges


def _hierholzer_trail(
    graph: SkeletonGraph,
    start: int,
    edge_use_counts: dict[int, int],
) -> list[RouteEdgeVisit]:
    local_counts = edge_use_counts.copy()
    stack: list[tuple[int, int | None, int]] = [(start, None, 0)]
    visits: list[RouteEdgeVisit] = []
    while stack:
        node, incoming_edge, incoming_dir = stack[-1]
        for neighbor, edge_id in _iter_adjacent_edges(graph, node):
            if local_counts.get(edge_id, 0) <= 0:
                continue
            local_counts[edge_id] -= 1
            edge = graph.edges[edge_id]
            direction = 1 if edge.u == node and edge.v == neighbor else -1
            stack.append((neighbor, edge_id, direction))
            break
        else:
            stack.pop()
            if incoming_edge is not None:
                visits.append(RouteEdgeVisit(edge_id=incoming_edge, direction=incoming_dir))
    visits.reverse()
    return visits


def _check_route_end(
    graph: SkeletonGraph,
    start: int,
    visits: Sequence[RouteEdgeVisit],
) -> int:
    current = start
    for visit in visits:
        edge = graph.edges[visit.edge_id]
        tail, head = edge.endpoints(direction=visit.direction)
        if tail != current:
            raise ValueError("Route construction produced an inconsistent walk.")
        current = head
    return current
