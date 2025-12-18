"""
Utilities for extracting the visually longest portion of a skeleton mask.

The module focuses on a single skeleton PNG at a time: load it, build an
8-neighborhood pixel graph, locate the longest simple path (tree assumption by
default), and export user-friendly artifacts (highlight image + polyline JSON).
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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
class LongestPathResult:
    """Stores the most prominent skeleton path and metadata."""

    path: list[tuple[int, int]]
    component: list[tuple[int, int]]
    shape: tuple[int, int]

    @property
    def path_points(self) -> int:
        return len(self.path)

    @property
    def path_length_px(self) -> int:
        return max(0, self.path_points - 1)

    @property
    def component_points(self) -> int:
        return len(self.component)

    def polyline(self) -> list[list[int]]:
        """Return the path as [[x, y], ...] coordinates for downstream tasks."""

        return [[int(col), int(row)] for row, col in self.path]

    def path_mask(self) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        for row, col in self.path:
            mask[row, col] = True
        return mask


def extract_longest_path(
    source: Image.Image | np.ndarray | str | Path,
) -> LongestPathResult:
    """
    Run the graph search on a skeleton mask (Pillow image, numpy array, or path).

    Returns:
        A LongestPathResult containing the pixel path and derived stats.
    """

    mask = _load_binary_mask(source)
    best_result: LongestPathResult | None = None

    for component in _iter_components(mask):
        path = _longest_path_in_component(component)
        if best_result is None or len(path) > best_result.path_points:
            best_result = LongestPathResult(
                path=list(path),
                component=list(component),
                shape=mask.shape,
            )

    if best_result is None:
        raise ValueError("No foreground pixels detected in mask.")

    return best_result


def export_longest_path(
    image_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Convenience helper: compute + save visualization and JSON side-by-side.

    Args:
        image_path: path to the skeleton PNG in data/skeletonized.
        output_dir: directory to store results; created if missing.

    Returns:
        Dict containing ``highlight`` (PNG path) and ``polyline`` (JSON path).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    result = extract_longest_path(image_path)
    highlight_path = output_dir / f"{image_path.stem}_longest.png"
    polyline_path = output_dir / f"{image_path.stem}_longest.json"

    with Image.open(image_path) as original:
        highlight = _render_highlight(original, result.path)
        highlight.save(highlight_path)

    payload = {
        "polyline": result.polyline(),
        "path_points": result.path_points,
        "path_length_px": result.path_length_px,
        "component_points": result.component_points,
    }
    polyline_path.write_text(json.dumps(payload, indent=2))

    return {"highlight": highlight_path, "polyline": polyline_path}


def _load_binary_mask(source: Image.Image | np.ndarray | str | Path) -> np.ndarray:
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


def _iter_components(mask: np.ndarray) -> Iterator[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True

            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in _neighbors(current, height, width):
                    nr, nc = neighbor
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append(neighbor)

            yield component


def _neighbors(
    coord: tuple[int, int], height: int, width: int
) -> Iterable[tuple[int, int]]:
    row, col = coord
    for d_row, d_col in EIGHT_NEIGHBOR_OFFSETS:
        nr, nc = row + d_row, col + d_col
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _longest_path_in_component(
    component: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not component:
        return []
    if len(component) == 1:
        return [component[0]]

    component_set = set(component)
    start = component[0]
    farthest_a, _ = _bfs_farthest(start, component_set)
    farthest_b, parents = _bfs_farthest(farthest_a, component_set)
    return _reconstruct_path(farthest_b, parents)


def _bfs_farthest(
    start: tuple[int, int],
    component_set: set[tuple[int, int]],
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
    queue: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distances: dict[tuple[int, int], int] = {start: 0}
    farthest = start

    while queue:
        current = queue.popleft()
        for neighbor in _neighbors_cached(current, component_set):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
            if distances[neighbor] >= distances[farthest]:
                farthest = neighbor

    return farthest, parents


def _neighbors_cached(
    coord: tuple[int, int],
    component_set: set[tuple[int, int]],
) -> Iterator[tuple[int, int]]:
    row, col = coord
    for d_row, d_col in EIGHT_NEIGHBOR_OFFSETS:
        neighbor = (row + d_row, col + d_col)
        if neighbor in component_set:
            yield neighbor


def _reconstruct_path(
    end: tuple[int, int],
    parents: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    current: tuple[int, int] | None = end
    while current is not None:
        path.append(current)
        current = parents[current]
    return list(reversed(path))


def _render_highlight(
    source_image: Image.Image,
    path: Sequence[tuple[int, int]],
) -> Image.Image:
    rgb = source_image.convert("RGB")
    canvas = np.array(rgb, copy=True)
    height, width, _ = canvas.shape
    highlight_color = np.array([255, 255, 0], dtype=np.float32)  # bright yellow
    alpha = 0.8

    for row, col in path:
        for d_row in range(-1, 2):
            for d_col in range(-1, 2):
                nr, nc = row + d_row, col + d_col
                if 0 <= nr < height and 0 <= nc < width:
                    base = canvas[nr, nc].astype(np.float32)
                    canvas[nr, nc] = np.clip(
                        (1 - alpha) * base + alpha * highlight_color, 0, 255
                    )

    return Image.fromarray(canvas.astype(np.uint8), mode="RGB")


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the longest path from a skeletonized image."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a skeletonized PNG (e.g., data/skeletonized/foo.png)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tmp"),
        help="Directory for visualization + JSON outputs (default: data/tmp).",
    )
    args = parser.parse_args(argv)

    artifacts = export_longest_path(args.source, args.out_dir)
    print(f"Wrote highlight to {artifacts['highlight']}")
    print(f"Wrote polyline JSON to {artifacts['polyline']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = ["LongestPathResult", "extract_longest_path", "export_longest_path"]
