"""Model utilities for the leafmine signature pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "run_skeletonization": ("models.skeletonization", "run_skeletonization"),
    "GraphNode": ("models.skeleton_graph", "GraphNode"),
    "GraphEdge": ("models.skeleton_graph", "GraphEdge"),
    "SkeletonGraph": ("models.skeleton_graph", "SkeletonGraph"),
    "build_skeleton_graph": ("models.skeleton_graph", "build_skeleton_graph"),
    "prune_short_branches": ("models.skeleton_graph", "prune_short_branches"),
    "RouteEdgeVisit": ("models.route", "RouteEdgeVisit"),
    "RouteResult": ("models.route", "RouteResult"),
    "compute_route": ("models.route", "compute_route"),
    "PolylineArtifacts": ("models.polyline_utils", "PolylineArtifacts"),
    "route_to_polyline": ("models.polyline_utils", "route_to_polyline"),
    "resample_polyline": ("models.polyline_utils", "resample_polyline"),
    "save_polyline_json": ("models.polyline_utils", "save_polyline_json"),
    "compute_polyline_artifacts": ("models.polyline_utils", "compute_polyline_artifacts"),
    "render_route_preview": ("models.polyline_utils", "render_route_preview"),
    "LogSignatureResult": ("models.signature", "LogSignatureResult"),
    "log_signature_from_json": ("models.signature", "log_signature_from_json"),
    "append_log_signature_csv": ("models.signature", "append_log_signature_csv"),
    "default_log_signature_csv_path": ("models.signature", "default_log_signature_csv_path"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'models' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
