from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw
import colorsys

from .skeleton_graph import SkeletonGraph

DEFAULT_HIGHLIGHT = (255, 255, 0)


def render_graph_overlay(
    graph: SkeletonGraph,
    base_image: Image.Image,
    *,
    edge_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_radius: int = 2,
    edge_width: int = 2,
    annotate_nodes: bool = False,
    label_color: tuple[int, int, int] = (255, 255, 255),
    label_font_size: int = 18,
) -> Image.Image:
    """Draw the graph edges + nodes onto a copy of ``base_image``."""

    overlay = base_image.copy()
    draw = ImageDraw.Draw(overlay)

    for edge in graph.edges.values():
        start = graph.nodes.get(edge.u)
        end = graph.nodes.get(edge.v)
        if not start or not end:
            continue
        draw.line([(start.x, start.y), (end.x, end.y)], fill=edge_color, width=edge_width)

    if node_radius > 0 or annotate_nodes:
        for node in graph.nodes.values():
            bbox = [
                (node.x - node_radius, node.y - node_radius),
                (node.x + node_radius, node.y + node_radius),
            ]
            if node_radius > 0:
                draw.ellipse(bbox, fill=node_color)
            if annotate_nodes:
                position = (node.x + node_radius + 2, node.y - node_radius - 10)
                draw.text(
                    position,
                    str(node.id),
                    fill=label_color,
                )

    return overlay


def render_graph_preview(
    graph: SkeletonGraph,
    skeleton_path: Path,
    *,
    edge_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_radius: int = 2,
    edge_width: int = 2,
    annotate_nodes: bool = False,
    label_color: tuple[int, int, int] = (255, 255, 255),
    label_font_size: int | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Open the skeleton PNG and return (original, overlay)."""

    with Image.open(skeleton_path) as source:
        base = source.convert("RGB")
    overlay = render_graph_overlay(
        graph,
        base,
        edge_color=edge_color,
        node_color=node_color,
        node_radius=node_radius,
        edge_width=edge_width,
        annotate_nodes=annotate_nodes,
        label_color=label_color,
        label_font_size=label_font_size,
    )
    return base, overlay


def render_route_overlay(
    base_image: Image.Image,
    polyline: Sequence[tuple[float, float]],
    *,
    width: int = 3,
) -> Image.Image:
    """Draw a rainbow gradient polyline over the base image."""

    overlay = base_image.copy()
    draw = ImageDraw.Draw(overlay)
    if len(polyline) < 2:
        return overlay
    segments = len(polyline) - 1
    for idx in range(segments):
        start = polyline[idx]
        end = polyline[idx + 1]
        t = idx / max(1, segments - 1)
        color = _rainbow_color(t)
        draw.line([start, end], fill=color, width=width)
    return overlay


def _rainbow_color(t: float) -> tuple[int, int, int]:
    hue = (1.0 - t) * 2 / 3  # map 0..1 to blue->red
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


__all__ = [
    "render_graph_overlay",
    "render_graph_preview",
    "render_route_overlay",
]
