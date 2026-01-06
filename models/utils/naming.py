"""Shared helpers to keep artifact filenames consistent across pipeline stages."""

from __future__ import annotations

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
        prefix = STAGE_PREFIXES[stage]
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
