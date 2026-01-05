"""Shared filesystem layout helpers for the Gradio UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from controllers.settings import load_data_dir
from models.utils.naming import canonical_sample_name


def _default_data_dir() -> Path:
    return load_data_dir()


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


def fetch_artifact_paths(
    directory: Path,
    pattern: str,
    *,
    skip_prefix: str | None = None,
) -> list[Path]:
    """Return sorted artifact paths inside ``directory`` matching ``pattern``."""

    directory = directory.expanduser()
    if not directory.exists():
        return []
    entries: list[Path] = []
    for path in sorted(directory.glob(pattern)):
        if skip_prefix and path.name.startswith(skip_prefix):
            continue
        if path.is_file():
            entries.append(path)
    return entries


def list_canonical_sample_ids(
    directory: Path,
    pattern: str,
    *,
    skip_prefix: str | None = None,
) -> list[str]:
    """Return canonical sample ids derived from filenames in ``directory``."""

    files = fetch_artifact_paths(directory, pattern, skip_prefix=skip_prefix)
    seen: set[str] = set()
    ordered: list[str] = []
    for path in files:
        sample_id = canonical_sample_name(path)
        if sample_id not in seen:
            seen.add(sample_id)
            ordered.append(sample_id)
    return ordered


__all__ = ["DataPaths", "fetch_artifact_paths", "list_canonical_sample_ids"]
