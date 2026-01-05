from __future__ import annotations

from dataclasses import dataclass

from controllers.data_paths import DataPaths, fetch_artifact_paths


@dataclass(slots=True)
class DataBrowser:
    """Thin helper that exposes dropdown choices derived from DataPaths."""

    config: DataPaths

    def segmented(self) -> list[str]:
        return _artifact_names(
            fetch_artifact_paths(
                self.config.segmented_dir,
                "*.png",
                skip_prefix="preprocessed_",
            )
        )

    def skeletonized(self) -> list[str]:
        return _artifact_names(
            fetch_artifact_paths(
                self.config.skeleton_dir,
                "*.png",
            )
        )

    def polylines(self) -> list[str]:
        return _artifact_names(
            fetch_artifact_paths(
                self.config.polyline_dir,
                "*.json",
            )
        )


def resolve_runtime_paths(
    data_paths: DataPaths | None = None,
    data_browser: DataBrowser | None = None,
) -> tuple[DataPaths, DataBrowser]:
    """Return concrete DataPaths/DataBrowser instances for UI tabs."""

    cfg = data_paths or DataPaths.from_data_dir()
    browser = data_browser or DataBrowser(cfg)
    return cfg, browser


def _artifact_names(paths) -> list[str]:
    return [path.name for path in paths]


__all__ = ["DataBrowser", "resolve_runtime_paths"]
