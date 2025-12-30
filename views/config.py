from __future__ import annotations

import os
from pathlib import Path

from controllers.pipeline import PipelineConfig
from models.signature import default_log_signature_csv_path

DATA_DIR = Path(os.environ.get('LEAFMINE_DATA_DIR', Path.cwd() / 'data'))


def build_pipeline_config(data_dir: Path | None = None) -> PipelineConfig:
    base = data_dir or DATA_DIR
    base = base.expanduser().resolve()
    signatures_dir = base / 'logsig'
    return PipelineConfig(
        segmented_dir=base / 'segmented',
        skeleton_dir=base / 'skeletonized',
        polyline_dir=base / 'polylines',
        signatures_dir=signatures_dir,
        signature_csv=default_log_signature_csv_path(signatures_dir),
    )


def list_segmented_masks(data_dir: Path | None = None) -> list[str]:
    """Return sorted segmented mask filenames under the configured data dir."""

    config = build_pipeline_config(data_dir)
    segmented_dir = config.segmented_dir
    if not segmented_dir.exists():
        return []
    names = []
    for path in segmented_dir.glob('*.png'):
        if path.name.startswith('preprocessed_'):
            continue
        names.append(path.name)
    return sorted(names)


def list_skeletonized_masks(data_dir: Path | None = None) -> list[str]:
    """Return sorted skeleton filenames under the configured data dir."""

    config = build_pipeline_config(data_dir)
    skeleton_dir = config.skeleton_dir
    if not skeleton_dir.exists():
        return []
    return sorted(path.name for path in skeleton_dir.glob('*.png'))


def list_polylines(data_dir: Path | None = None) -> list[str]:
    """Return sorted polyline JSON filenames under the configured data dir."""

    config = build_pipeline_config(data_dir)
    polyline_dir = config.polyline_dir
    if not polyline_dir.exists():
        return []
    return sorted(path.name for path in polyline_dir.glob('*.json'))


__all__ = [
    'DATA_DIR',
    'build_pipeline_config',
    'list_segmented_masks',
    'list_skeletonized_masks',
    'list_polylines',
]
