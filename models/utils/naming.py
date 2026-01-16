"""Shared helpers to keep artifact filenames consistent across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Canonical prefixes for each pipeline stage. Stages that map to the same prefix
# (e.g. "mask" and "segmented") allow callers to use the terminology that makes
# sense in their context while keeping the persisted filenames identical.
STAGE_PREFIXES: dict[str, str] = {
    "mask": "segmented_",
    "segmented": "segmented_",
    "preprocessed": "preprocessed_",
    "skeleton": "skeletonized_",
    "skeletonized": "skeletonized_",
    "graph": "graph_",
    "polyline": "polyline_",
    "route": "route_",
}


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Metadata describing a pipeline stage's on-disk representation."""

    name: str
    prefix: str
    default_suffix: str
    glob: str


STAGE_ALIASES: dict[str, str] = {
    "mask": "segmented",
    "skeleton": "skeletonized",
}

STAGE_SPECS: dict[str, StageSpec] = {
    "segmented": StageSpec("segmented", STAGE_PREFIXES["segmented"], ".png", "segmented_*.png"),
    "preprocessed": StageSpec(
        "preprocessed",
        STAGE_PREFIXES["preprocessed"],
        ".png",
        "preprocessed_*.png",
    ),
    "skeletonized": StageSpec(
        "skeletonized",
        STAGE_PREFIXES["skeletonized"],
        ".png",
        "skeletonized_*.png",
    ),
    "graph": StageSpec("graph", STAGE_PREFIXES["graph"], ".json", "graph_*.json"),
    "polyline": StageSpec("polyline", STAGE_PREFIXES["polyline"], ".json", "polyline_*.json"),
    "route": StageSpec("route", STAGE_PREFIXES["route"], ".json", "route_*.json"),
}


def canonical_stage_name(stage: str) -> str:
    """Return the canonical stage name (resolving aliases)."""

    key = stage.strip().lower()
    return STAGE_ALIASES.get(key, key)


def stage_spec(stage: str) -> StageSpec:
    """Return the stage metadata for ``stage``."""

    name = canonical_stage_name(stage)
    spec = STAGE_SPECS.get(name)
    if spec is None:  # pragma: no cover - developer errors
        raise KeyError(f"Unknown stage '{stage}'")
    return spec


def known_prefixes() -> tuple[str, ...]:
    """Return the known prefixes for sanitizing stems."""

    return tuple(dict.fromkeys(STAGE_PREFIXES.values()))


def strip_prefix(value: str, *, extra: Iterable[str] | None = None) -> str:
    """Remove and return ``value`` without any known prefix."""

    prefixes = list(known_prefixes())
    if extra:
        prefixes.extend(extra)
    for prefix in prefixes:
        if value.startswith(prefix) and len(value) > len(prefix):
            return value[len(prefix) :]
    return value


def apply_stage_prefix(stage: str, base: str) -> str:
    """Return ``base`` prefixed for ``stage`` (ensuring no duplicate prefixes)."""

    try:
        prefix = STAGE_PREFIXES[canonical_stage_name(stage)]
    except KeyError as exc:  # pragma: no cover - developer errors
        raise KeyError(f"Unknown stage '{stage}'") from exc
    cleaned = strip_prefix(base)
    return prefix + cleaned


def prefixed_name(stage: str, base: str, suffix: str) -> str:
    """Convenience helper returning the complete filename for ``stage``."""

    return f"{apply_stage_prefix(stage, base)}{suffix}"


def canonical_sample_name(path: Path | str) -> str:
    """Best-effort attempt at deriving the logical sample name from a path."""

    stem = Path(path).stem
    cleaned = strip_prefix(stem)
    return cleaned or stem
