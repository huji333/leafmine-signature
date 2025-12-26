from __future__ import annotations

import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from .graph_render import render_route_overlay
from .route import RouteEdgeVisit, RouteResult
from .skeleton_graph import SkeletonGraph


@dataclass(slots=True)
class PolylineArtifacts:
    polyline: list[tuple[float, float]]
    resampled: list[tuple[float, float]]
    path_length: float
    polyline_path: Path | None = None
    payload: dict[str, object] | None = None


def route_to_polyline(
    graph: SkeletonGraph,
    visits: Sequence[RouteEdgeVisit],
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for visit in visits:
        edge = graph.edges[visit.edge_id]
        edge_points = edge.points
        if not edge_points:
            start = graph.nodes[edge.u]
            end = graph.nodes[edge.v]
            edge_points = ((start.x, start.y), (end.x, end.y))
        coords = list(edge_points)
        if visit.direction < 0:
            coords.reverse()
        if not points:
            points.extend(coords)
        else:
            points.extend(coords[1:])
    return points


def resample_polyline(
    points: Sequence[tuple[float, float]],
    num_samples: int,
) -> tuple[list[tuple[float, float]], float]:
    if not points:
        raise ValueError("Polyline is empty.")
    if len(points) == 1:
        return [points[0]] * max(1, num_samples), 0.0
    distances: list[float] = [0.0]
    total = 0.0
    for a, b in zip(points, points[1:]):
        seg = hypot(b[0] - a[0], b[1] - a[1])
        total += seg
        distances.append(total)
    if total == 0.0:
        return [points[0]] * max(1, num_samples), 0.0
    samples = max(2, int(num_samples))
    step = total / (samples - 1)
    resampled: list[tuple[float, float]] = []
    idx = 0
    for i in range(samples):
        target = i * step
        while idx + 1 < len(distances) and distances[idx + 1] < target:
            idx += 1
        if idx + 1 >= len(distances):
            resampled.append(points[-1])
            continue
        prev_dist = distances[idx]
        next_dist = distances[idx + 1]
        if next_dist == prev_dist:
            t = 0.0
        else:
            t = (target - prev_dist) / (next_dist - prev_dist)
        ax, ay = points[idx]
        bx, by = points[idx + 1]
        resampled.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return resampled, total


def save_polyline_json(
    *,
    polyline: Sequence[tuple[float, float]],
    resampled: Sequence[tuple[float, float]],
    path_length: float,
    metadata: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    payload = {
        "polyline": [[float(x), float(y)] for x, y in polyline],
        "resampled_polyline": [[float(x), float(y)] for x, y in resampled],
        "path_length": float(path_length),
        **metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def compute_polyline_artifacts(
    graph: SkeletonGraph,
    result: RouteResult,
    *,
    resample_points: int,
    metadata: dict[str, object],
    output_path: Path,
) -> PolylineArtifacts:
    polyline = route_to_polyline(graph, result.visits)
    resampled, path_length = resample_polyline(polyline, resample_points)
    payload = save_polyline_json(
        polyline=polyline,
        resampled=resampled,
        path_length=path_length,
        metadata=metadata,
        output_path=output_path,
    )
    return PolylineArtifacts(
        polyline=polyline,
        resampled=resampled,
        path_length=path_length,
        polyline_path=output_path,
        payload=payload,
    )


def render_route_preview(
    skeleton_path: Path,
    polyline: Sequence[tuple[float, float]],
) -> Image.Image:
    with Image.open(skeleton_path) as source:
        base = source.convert("RGB")
    return render_route_overlay(base, polyline)


__all__ = [
    "PolylineArtifacts",
    "route_to_polyline",
    "resample_polyline",
    "save_polyline_json",
    "compute_polyline_artifacts",
    "render_route_preview",
]
