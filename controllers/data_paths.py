"""Shared filesystem layout helpers for the Gradio UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from controllers.settings import load_data_dir

if TYPE_CHECKING:
    from models.signature import default_log_signature_csv_path


def _default_data_dir() -> Path:
    return load_data_dir()


def default_data_subdir(name: str) -> Path:
    """Return a default subdirectory path under the data directory."""
    return load_data_dir() / name


@dataclass(slots=True)
class DataPaths:
    """Canonical directories for segmented masks, skeletons, and signatures."""

    segmented_dir: Path = Path("data/segmented")
    skeleton_dir: Path = Path("data/skeletonized")
    polyline_dir: Path = Path("data/polylines")
    graph_dir: Path = Path("data/graphs")
    signatures_dir: Path = Path("data/logsig")

    def ensure_directories(self) -> None:
        """Ensure all directories exist."""
        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)

    def ensure_signature_directories(self) -> None:
        """Ensure directories needed for signature computation exist."""
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)

    def summary_csv_path(self) -> Path:
        """Return the default log signature CSV path."""
        from models.signature import default_log_signature_csv_path

        return default_log_signature_csv_path(self.signatures_dir)

    @classmethod
    def from_data_dir(cls, data_dir: Path | None = None) -> DataPaths:
        base = data_dir or _default_data_dir()
        base = base.expanduser().resolve()
        signatures_dir = base / "logsig"
        return cls(
            segmented_dir=base / "segmented",
            skeleton_dir=base / "skeletonized",
            polyline_dir=base / "polylines",
            graph_dir=base / "graphs",
            signatures_dir=signatures_dir,
        )


__all__ = ["DataPaths", "default_data_subdir"]
