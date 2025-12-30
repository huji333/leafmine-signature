from __future__ import annotations

from dataclasses import dataclass

from controllers.pipeline import PipelineConfig


@dataclass(slots=True)
class DataBrowser:
    """Thin helper that exposes dropdown choices derived from PipelineConfig."""

    config: PipelineConfig

    def segmented(self) -> list[str]:
        return _list_files(self.config.segmented_dir, suffix=".png", skip_prefix="preprocessed_")

    def skeletonized(self) -> list[str]:
        return _list_files(self.config.skeleton_dir, suffix=".png")

    def polylines(self) -> list[str]:
        return _list_files(self.config.polyline_dir, suffix=".json")


def _list_files(directory, *, suffix: str, skip_prefix: str | None = None) -> list[str]:
    directory = directory.expanduser()
    if not directory.exists():
        return []
    entries: list[str] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        name = path.name
        if skip_prefix and name.startswith(skip_prefix):
            continue
        entries.append(name)
    return entries


__all__ = ["DataBrowser"]
