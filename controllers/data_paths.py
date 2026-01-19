"""Shared filesystem layout helpers for the Gradio UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from controllers.settings import load_data_dir
from models.utils.naming import canonical_sample_name, canonical_stage_name, stage_spec


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
    umap_dir: Path = Path("data/umap")
    annotations_dir: Path = Path("data")

    def ensure_directories(self) -> None:
        """Ensure all directories exist."""
        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.umap_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def ensure_signature_directories(self) -> None:
        """Ensure directories needed for signature computation exist."""
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)

    def summary_csv_path(self) -> Path:
        """Return the default log signature CSV path."""
        from models.signature import default_log_signature_csv_path

        return default_log_signature_csv_path(self.signatures_dir)

    def stage_dir(self, stage: str) -> Path:
        """Return the directory that stores artifacts for ``stage``."""

        name = canonical_stage_name(stage)
        if name in {"segmented", "preprocessed"}:
            return self.segmented_dir
        if name == "skeletonized":
            return self.skeleton_dir
        if name in {"polyline", "route"}:
            return self.polyline_dir
        if name == "graph":
            return self.graph_dir
        if name == "logsig":
            return self.signatures_dir
        raise KeyError(f"Unknown stage '{stage}'")

    def list_stage_paths(
        self,
        stage: str,
        *,
        skip_prefix: str | None = None,
    ) -> list[Path]:
        """Return artifact paths for ``stage``."""

        spec = stage_spec(stage)
        directory = self.stage_dir(stage)
        return fetch_artifact_paths(directory, spec.glob, skip_prefix=skip_prefix)

    def list_stage_names(
        self,
        stage: str,
        *,
        skip_prefix: str | None = None,
    ) -> list[str]:
        """Return artifact filenames for ``stage``."""

        return [path.name for path in self.list_stage_paths(stage, skip_prefix=skip_prefix)]

    def segmented_names(self) -> list[str]:
        """Return segmented mask filenames (excluding preprocessed variants)."""

        return self.list_stage_names(
            "segmented",
            skip_prefix=stage_spec("preprocessed").prefix,
        )

    def skeletonized_names(self) -> list[str]:
        """Return skeletonized mask filenames."""

        return self.list_stage_names("skeletonized")

    def polyline_names(self) -> list[str]:
        """Return polyline JSON filenames."""

        return self.list_stage_names("polyline")

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
            umap_dir=base / "umap",
            annotations_dir=base,
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
