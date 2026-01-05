"""Load the shared data directory from config.toml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tomllib

CONFIG_FILENAME = "config.toml"
CONFIG_SECTION = "leafmine"
CONFIG_KEY = "data_dir"
DEFAULT_DATA_DIR = "data"


def load_data_dir() -> Path:
    """Return the canonical data directory resolved from config.toml."""

    configured = _read_config_data_dir()
    return _resolve_path(configured)


@lru_cache(maxsize=1)
def _read_config_data_dir() -> str:
    config_path = _project_root() / CONFIG_FILENAME
    if not config_path.is_file():
        return DEFAULT_DATA_DIR

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    section = config.get(CONFIG_SECTION)
    if isinstance(section, dict):
        candidate = section.get(CONFIG_KEY)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return DEFAULT_DATA_DIR


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_project_root() / path).resolve()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


__all__ = ["load_data_dir"]
