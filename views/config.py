from __future__ import annotations

from dataclasses import dataclass

from controllers.data_paths import DataPaths


@dataclass(slots=True)
class DataBrowser:
    """Thin helper that exposes dropdown choices derived from DataPaths."""

    config: DataPaths

    def segmented(self) -> list[str]:
        return _list_files(self.config.segmented_dir, suffix=".png", skip_prefix="preprocessed_")

    def skeletonized(self) -> list[str]:
        return _list_files(self.config.skeleton_dir, suffix=".png")

def polylines(self) -> list[str]:
    return _list_files(self.config.polyline_dir, suffix=".json")


def resolve_runtime_paths(
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> tuple[DataPaths, DataBrowser]:
    """Return concrete DataPaths/DataBrowser instances for UI tabs."""

    cfg = data_paths or DataPaths.from_data_dir()
    browser = data_browser or DataBrowser(cfg)
    return cfg, browser


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


__all__ = ["DataBrowser", "resolve_runtime_paths"]
