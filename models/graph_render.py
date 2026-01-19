from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont
import colorsys

from .skeleton_graph import SkeletonGraph

DEFAULT_HIGHLIGHT = (255, 255, 0)


def render_graph_overlay(
    graph: SkeletonGraph,
    base_image: Image.Image,
    *,
    draw_edges: bool = True,
    offset: tuple[int, int] = (0, 0),
    edge_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_radius: int = 2,
    edge_width: int = 2,
    annotate_nodes: bool = False,
    annotate_node_ids: set[int] | None = None,
    leaf_ids: set[int] | None = None,
    junction_ids: set[int] | None = None,
    leaf_color: tuple[int, int, int] | None = None,
    junction_color: tuple[int, int, int] | None = None,
    leaf_radius: int | None = None,
    junction_radius: int | None = None,
    label_color: tuple[int, int, int] = (255, 255, 255),
    label_font_size: int = 0,
    label_stroke_fill: tuple[int, int, int] | None = (0, 0, 0),
    label_stroke_width: int = 2,
    leaf_label_scale: float = 0.75,
    junction_label_scale: float = 0.4,
) -> Image.Image:
    """Draw the graph edges + nodes onto a copy of ``base_image``."""

    overlay = base_image.copy()
    draw = ImageDraw.Draw(overlay)

    offset_x, offset_y = offset
    if draw_edges and edge_width > 0:
        for edge in graph.edges.values():
            start = graph.nodes.get(edge.u)
            end = graph.nodes.get(edge.v)
            if not start or not end:
                continue
            draw.line(
                [
                    (start.x - offset_x, start.y - offset_y),
                    (end.x - offset_x, end.y - offset_y),
                ],
                fill=edge_color,
                width=edge_width,
            )

    base_font = None
    leaf_font = None
    junction_font = None
    if annotate_nodes:
        base_font = _load_label_font(base_image.size, label_font_size)
        if leaf_ids:
            leaf_font = _load_label_font(
                base_image.size,
                label_font_size,
                scale=leaf_label_scale,
            )
        if junction_ids:
            junction_font = _load_label_font(
                base_image.size,
                label_font_size,
                scale=junction_label_scale,
            )
        if leaf_font is None:
            leaf_font = base_font
        if junction_font is None:
            junction_font = base_font

    annotate_ids = set(annotate_node_ids) if annotate_node_ids is not None else None

    if node_radius > 0 or annotate_nodes:
        for node in graph.nodes.values():
            color = node_color
            radius = node_radius
            if leaf_ids and node.id in leaf_ids:
                color = leaf_color or node_color
                radius = max(node_radius, (leaf_radius or node_radius))
            elif junction_ids and node.id in junction_ids:
                color = junction_color or node_color
                radius = max(node_radius, (junction_radius or node_radius))
            x = node.x - offset_x
            y = node.y - offset_y
            bbox = [
                (x - radius, y - radius),
                (x + radius, y + radius),
            ]
            if radius > 0:
                draw.ellipse(bbox, fill=color)
            should_label = annotate_nodes and (
                annotate_ids is None or node.id in annotate_ids
            )
            if should_label:
                label_font = base_font
                if leaf_ids and node.id in leaf_ids and leaf_font is not None:
                    label_font = leaf_font
                elif junction_ids and node.id in junction_ids and junction_font is not None:
                    label_font = junction_font
                position = (x + radius + 2, y - radius - 8)
                text_kwargs: dict[str, object] = {
                    "fill": label_color,
                    "font": label_font,
                }
                if label_stroke_fill is not None and label_stroke_width > 0:
                    text_kwargs["stroke_width"] = label_stroke_width
                    text_kwargs["stroke_fill"] = label_stroke_fill
                draw.text(
                    position,
                    str(node.id),
                    **text_kwargs,
                )

    return overlay


def render_graph_preview(
    graph: SkeletonGraph,
    skeleton_path: Path,
    *,
    draw_edges: bool = True,
    offset: tuple[int, int] = (0, 0),
    edge_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT,
    node_radius: int = 2,
    edge_width: int = 2,
    annotate_nodes: bool = False,
    annotate_node_ids: set[int] | None = None,
    leaf_ids: set[int] | None = None,
    junction_ids: set[int] | None = None,
    leaf_color: tuple[int, int, int] | None = None,
    junction_color: tuple[int, int, int] | None = None,
    leaf_radius: int | None = None,
    junction_radius: int | None = None,
    label_color: tuple[int, int, int] = (255, 255, 255),
    label_font_size: int = 0,
    label_stroke_fill: tuple[int, int, int] | None = (0, 0, 0),
    label_stroke_width: int = 2,
    leaf_label_scale: float = 0.75,
    junction_label_scale: float = 0.4,
) -> tuple[Image.Image, Image.Image]:
    """Open the skeleton PNG and return (original, overlay)."""

    with Image.open(skeleton_path) as source:
        base = source.convert("RGB")
    overlay = render_graph_overlay(
        graph,
        base,
        draw_edges=draw_edges,
        offset=offset,
        edge_color=edge_color,
        node_color=node_color,
        node_radius=node_radius,
        edge_width=edge_width,
        annotate_nodes=annotate_nodes,
        annotate_node_ids=annotate_node_ids,
        leaf_ids=leaf_ids,
        junction_ids=junction_ids,
        leaf_color=leaf_color,
        junction_color=junction_color,
        leaf_radius=leaf_radius,
        junction_radius=junction_radius,
        label_color=label_color,
        label_font_size=label_font_size,
        label_stroke_fill=label_stroke_fill,
        label_stroke_width=label_stroke_width,
        leaf_label_scale=leaf_label_scale,
        junction_label_scale=junction_label_scale,
    )
    return base, overlay


def _load_label_font(
    image_size: tuple[int, int],
    requested_size: int,
    *,
    scale: float = 1.0,
) -> ImageFont.ImageFont:
    font_size = int(requested_size or 0)
    clamped_scale = max(0.1, float(scale))
    if font_size <= 0:
        min_dim = max(1, min(image_size))
        auto_size = max(6, min_dim // 40)
        font_size = max(6, int(auto_size * clamped_scale))
    else:
        font_size = max(1, int(font_size * clamped_scale))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


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
