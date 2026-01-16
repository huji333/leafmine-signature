from __future__ import annotations

from dataclasses import dataclass

from controllers.data_paths import DataPaths


@dataclass(slots=True)
class DataBrowser:
    """Thin helper that exposes dropdown choices derived from DataPaths."""

    config: DataPaths

    def segmented(self) -> list[str]:
        return self.config.segmented_names()

    def skeletonized(self) -> list[str]:
        return self.config.skeletonized_names()

    def polylines(self) -> list[str]:
        return self.config.polyline_names()


def resolve_runtime_paths(
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> tuple[DataPaths, DataBrowser]:
    """Return concrete DataPaths/DataBrowser instances for UI tabs."""

    cfg = data_paths or DataPaths.from_data_dir()
    browser = data_browser or DataBrowser(cfg)
    return cfg, browser


def reconcile_selection(choices: list[str], current_selection: list[str] | None) -> list[str]:
    """Preserve user selection where possible, otherwise select all."""

    if not choices:
        return []
    current = current_selection or []
    value = [item for item in current if item in choices]
    if not value:
        value = choices
    return value


__all__ = ["DataBrowser", "resolve_runtime_paths", "reconcile_selection"]
