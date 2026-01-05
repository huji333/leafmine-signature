"""Helpers to resolve artifact filenames from loose identifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from models.utils.naming import canonical_sample_name, prefixed_name


def resolve_stage_artifact_path(
    identifier: str | Path,
    directories: Iterable[Path | str | None],
    *,
    stage_names: Sequence[str],
    default_suffix: str = ".png",
    extra_suffixes: Sequence[str] | None = None,
) -> Path | None:
    """Return the first artifact path that matches ``identifier`` within ``directories``."""

    candidate = Path(identifier).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_dirs = _dedupe_paths(directories)
    if not search_dirs:
        return None

    names = _candidate_names(candidate, stage_names, default_suffix, extra_suffixes)
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in names:
            resolved = directory / name
            if resolved.exists():
                return resolved
    return None


def resolve_segmented_mask_path(
    identifier: str | Path,
    directories: Iterable[Path | str | None],
    *,
    default_suffix: str = ".png",
) -> Path | None:
    """Resolve ``identifier`` to a segmented mask path (supports bare basenames)."""

    return resolve_stage_artifact_path(
        identifier,
        directories,
        stage_names=("segmented", "mask"),
        default_suffix=default_suffix,
    )


def _candidate_names(
    candidate: Path,
    stage_names: Sequence[str],
    default_suffix: str,
    extra_suffixes: Sequence[str] | None,
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    if candidate.name:
        names.append(candidate.name)
        seen.add(candidate.name)
    if candidate.parts and len(candidate.parts) > 1 and not candidate.is_absolute():
        relative = candidate.as_posix()
        if relative not in seen:
            names.append(relative)
            seen.add(relative)

    suffixes: list[str] = []
    if candidate.suffix:
        suffixes.append(candidate.suffix)
    if extra_suffixes:
        for suffix in extra_suffixes:
            if suffix and suffix not in suffixes:
                suffixes.append(suffix)
    if default_suffix and default_suffix not in suffixes:
        suffixes.append(default_suffix)
    if not suffixes:
        suffixes.append("")

    stem = candidate.stem or candidate.name or str(candidate)
    sample_base = canonical_sample_name(stem)
    if sample_base:
        for suffix in suffixes:
            for stage in stage_names:
                prefixed = prefixed_name(stage, sample_base, suffix)
                if prefixed not in seen:
                    names.append(prefixed)
                    seen.add(prefixed)
            plain = f"{sample_base}{suffix}"
            if plain not in seen:
                names.append(plain)
                seen.add(plain)
    return names


def _dedupe_paths(paths: Iterable[Path | str | None]) -> list[Path]:
    seen: set[Path] = set()
    results: list[Path] = []
    for item in paths:
        if not item:
            continue
        path = Path(item).expanduser()
        if path in seen:
            continue
        seen.add(path)
        results.append(path)
    return results


__all__ = [
    "resolve_stage_artifact_path",
    "resolve_segmented_mask_path",
]
