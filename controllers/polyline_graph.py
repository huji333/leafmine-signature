"""Graph preparation helpers for skeleton routing workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from controllers.artifacts import ensure_flat_stage_identifier, resolve_segmented_mask_path
from controllers.data_paths import DataPaths
from models.graph_render import render_graph_preview
from models.skeleton_graph import SkeletonGraph, build_skeleton_graph, prune_short_branches
from models.utils.image_io import load_image
from models.utils.naming import canonical_sample_name, prefixed_name


@dataclass(slots=True)
class GraphSource:
    sample_base: str
    skeleton_path: Path
    base_graph: SkeletonGraph


@dataclass(slots=True)
class GraphSession:
    source: GraphSource
    pruned_graph: SkeletonGraph
    graph_json_path: Path
    manifest_path: Path
    branch_threshold: float
    loop_threshold: float
    polyline_dir: Path
    component_map: dict[int, int]


@dataclass(slots=True)
class GraphPrepResult:
    skeleton_image: Image.Image
    segmented_image: Image.Image | None
    pruned_overlay: Image.Image
    graph_payload: dict[str, object]
    graph_json_path: Path
    manifest_path: Path
    short_edges: list[dict[str, float | int]]
    default_start: int | None
    default_goal: int | None
    status_message: str
    session: GraphSession


def prepare_graph(
    skeleton_filename: str | Path,
    branch_threshold: float,
    loop_threshold: float | None = None,
    *,
    data_paths: DataPaths | None = None,
) -> GraphPrepResult:
    paths = data_paths or DataPaths.from_data_dir()
    paths.ensure_directories()

    ensure_flat_stage_identifier(
        skeleton_filename,
        description="Skeleton filename",
    )
    skeleton_path = _resolve_filename(skeleton_filename, paths.skeleton_dir)
    branch_threshold = max(0.0, float(branch_threshold))
    loop_threshold = branch_threshold if loop_threshold is None else max(0.0, float(loop_threshold))
    should_prune = branch_threshold > 0 or loop_threshold > 0

    sample_base = canonical_sample_name(skeleton_path)
    source = _load_graph_source(skeleton_path, sample_base)
    pruned_graph = (
        prune_short_branches(
            source.base_graph,
            branch_threshold,
            loop_threshold=loop_threshold,
        )
        if should_prune
        else source.base_graph.copy()
    )

    graph_filename = prefixed_name("graph", sample_base, ".json")
    graph_json_path = paths.graph_dir / graph_filename
    graph_payload = pruned_graph.to_payload()
    pruned_graph.save(graph_json_path)
    manifest_path = _write_graph_manifest(
        graph_json_path,
        source,
        branch_threshold,
        loop_threshold,
        pruned_graph,
    )

    components = _connected_components(pruned_graph)
    component_map = {
        node_id: index for index, comp in enumerate(components) for node_id in comp
    }

    leaf_nodes = _leaf_metadata(pruned_graph)
    leaf_ids = {leaf["id"] for leaf in leaf_nodes}
    junction_ids = _junction_node_ids(pruned_graph)
    skeleton_image, overlay = _render_images(
        pruned_graph,
        skeleton_path,
        leaf_ids=leaf_ids,
        junction_ids=junction_ids,
    )
    segmented_preview = _load_segmented_preview(sample_base, skeleton_path, paths.segmented_dir)
    short_edges = _short_edge_summary(pruned_graph, leaf_ids=leaf_ids)

    payload = {**graph_payload}
    payload["metadata"] = {
        "skeleton_png": str(skeleton_path),
        "graph_json": str(graph_json_path),
        "branch_threshold": branch_threshold,
        "loop_threshold": loop_threshold,
        "node_count": len(pruned_graph.nodes),
        "edge_count": len(pruned_graph.edges),
        "leaf_nodes": len(leaf_nodes),
        "sample_base": sample_base,
        "segmented_filename": prefixed_name("segmented", sample_base, ".png"),
    }

    defaults = _default_start_goal(pruned_graph, components)
    message = (
        f"Saved graph to {graph_json_path.name}: "
        f"{len(pruned_graph.nodes)} nodes / {len(pruned_graph.edges)} edges"
    )

    session = GraphSession(
        source=source,
        pruned_graph=pruned_graph,
        graph_json_path=graph_json_path,
        manifest_path=manifest_path,
        branch_threshold=branch_threshold,
        loop_threshold=loop_threshold,
        polyline_dir=paths.polyline_dir,
        component_map=component_map,
    )

    return GraphPrepResult(
        skeleton_image=skeleton_image,
        segmented_image=segmented_preview,
        pruned_overlay=overlay,
        graph_payload=payload,
        graph_json_path=graph_json_path,
        manifest_path=manifest_path,
        short_edges=short_edges,
        default_start=defaults[0],
        default_goal=defaults[1],
        status_message=message,
        session=session,
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


def _junction_node_ids(graph: SkeletonGraph, *, min_degree: int = 3) -> set[int]:
    return {node_id for node_id in graph.nodes if graph.degree(node_id) >= min_degree}


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


def _load_segmented_preview(
    sample_base: str,
    skeleton_path: Path,
    segmented_dir: Path | None,
) -> Image.Image | None:
    suffix = skeleton_path.suffix or ".png"
    skeleton_dir = skeleton_path.parent
    search_dirs = [
        segmented_dir,
        skeleton_dir,
        skeleton_dir.parent / "segmented",
    ]
    candidate = resolve_segmented_mask_path(
        sample_base,
        search_dirs,
        default_suffix=suffix,
    )
    if candidate is None:
        return None
    return load_image(candidate, mode="RGB")


def _render_images(
    graph: SkeletonGraph,
    skeleton_path: Path,
    *,
    leaf_ids: set[int] | None = None,
    junction_ids: set[int] | None = None,
) -> tuple[Image.Image, Image.Image]:
    annotate_ids: set[int] | None = None
    if leaf_ids or junction_ids:
        annotate_ids = set()
        if leaf_ids:
            annotate_ids.update(leaf_ids)
        if junction_ids:
            annotate_ids.update(junction_ids)
    return render_graph_preview(
        graph,
        skeleton_path,
        annotate_nodes=True,
        annotate_node_ids=annotate_ids,
        node_radius=4,
        edge_width=3,
        node_color=(220, 220, 220),
        leaf_ids=leaf_ids,
        junction_ids=junction_ids,
        leaf_color=(255, 210, 64),
        junction_color=(64, 196, 255),
        leaf_radius=6,
        junction_radius=4,
        label_color=(255, 255, 255),
        label_stroke_fill=(0, 0, 0),
        label_stroke_width=1,
        leaf_label_scale=0.55,
        junction_label_scale=0.3,
    )


def _load_graph_source(skeleton_path: Path, sample_base: str) -> GraphSource:
    base_graph = _load_cached_skeleton_graph(skeleton_path)
    return GraphSource(
        sample_base=sample_base,
        skeleton_path=skeleton_path,
        base_graph=base_graph,
    )


def _write_graph_manifest(
    graph_json_path: Path,
    source: GraphSource,
    branch_threshold: float,
    loop_threshold: float,
    graph: SkeletonGraph,
) -> Path:
    manifest = {
        "graph_json": str(graph_json_path),
        "manifest_version": 1,
        "sample_base": source.sample_base,
        "skeleton_png": str(source.skeleton_path),
        "branch_threshold": branch_threshold,
        "loop_threshold": loop_threshold,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "segmented_filename": prefixed_name("segmented", source.sample_base, ".png"),
    }
    manifest_path = graph_json_path.with_suffix(".meta.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _load_cached_skeleton_graph(skeleton_path: Path) -> SkeletonGraph:
    resolved = skeleton_path.resolve()
    mtime_ns = _file_mtime_ns(resolved)
    return _skeleton_graph_cache(resolved.as_posix(), mtime_ns)


def _file_mtime_ns(path: Path) -> int:
    stat_result = path.stat()
    return getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))


@lru_cache(maxsize=16)
def _skeleton_graph_cache(path_str: str, mtime_ns: int) -> SkeletonGraph:
    _ = mtime_ns  # memoization key ensures cache busts when file timestamp changes
    return build_skeleton_graph(Path(path_str))


__all__ = [
    "GraphSource",
    "GraphPrepResult",
    "GraphSession",
    "prepare_graph",
]
