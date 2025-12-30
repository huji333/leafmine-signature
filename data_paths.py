"""Shared filesystem layout helpers for the Gradio UI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from models.signature import default_log_signature_csv_path


def _default_data_dir() -> Path:
    env_dir = os.environ.get("LEAFMINE_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.cwd() / "data"


@dataclass(slots=True)
class DataPaths:
    """Canonical directories for segmented masks, skeletons, and signatures."""

    segmented_dir: Path = Path("data/segmented")
    skeleton_dir: Path = Path("data/skeletonized")
    polyline_dir: Path = Path("data/polylines")
    tmp_dir: Path = Path("data/tmp")
    signatures_dir: Path = Path("data/logsig")
    signature_csv: Path = field(default_factory=default_log_signature_csv_path)

    def ensure_directories(self) -> None:
        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        self.polyline_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.signature_csv.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_data_dir(cls, data_dir: Path | None = None) -> DataPaths:
        base = data_dir or _default_data_dir()
        base = base.expanduser().resolve()
        signatures_dir = base / "logsig"
        return cls(
            segmented_dir=base / "segmented",
            skeleton_dir=base / "skeletonized",
            polyline_dir=base / "polylines",
            tmp_dir=base / "tmp",
            signatures_dir=signatures_dir,
            signature_csv=default_log_signature_csv_path(signatures_dir),
        )


__all__ = ["DataPaths"]
