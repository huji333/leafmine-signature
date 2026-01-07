"""Helpers for surfacing downstream-processing status in the UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Iterable

from controllers.data_paths import DataPaths, fetch_artifact_paths
from models.utils.naming import canonical_sample_name


class ActionType(Enum):
    """Pipeline steps that surface completion badges in the UI."""

    SKELETON = auto()
    ROUTE = auto()


@dataclass(frozen=True, slots=True)
class _ActionSpec:
    directory_attr: str
    pattern: str


_ACTION_SPECS: dict[ActionType, _ActionSpec] = {
    ActionType.SKELETON: _ActionSpec("skeleton_dir", "skeletonized_*.png"),
    ActionType.ROUTE: _ActionSpec("polyline_dir", "polyline_*.json"),
}


class ProcessingStatusService:
    """Caches which samples have completed downstream processing per action."""

    def __init__(self, paths: DataPaths):
        self._paths = paths
        self._cache: dict[ActionType, set[str]] = {}

    def refresh(self, actions: Iterable[ActionType] | None = None) -> None:
        """Rescan filesystem artifacts for the provided actions."""

        targets = list(actions) if actions is not None else list(ActionType)
        for action in targets:
            spec = _ACTION_SPECS.get(action)
            if spec is None:
                continue
            directory = getattr(self._paths, spec.directory_attr)
            files = fetch_artifact_paths(directory, spec.pattern)
            samples = {canonical_sample_name(path) for path in files}
            self._cache[action] = samples

    def refresh_action(self, action: ActionType) -> None:
        self.refresh([action])

    def is_done(self, identifier: str | Path, action: ActionType) -> bool:
        """Return True if the identifier already has the downstream artifact."""

        sample_name = canonical_sample_name(Path(identifier).stem)
        if not sample_name:
            return False
        seen = self._cache.get(action)
        if seen is None:
            self.refresh_action(action)
            seen = self._cache.get(action, set())
        return sample_name in seen


__all__ = ["ActionType", "ProcessingStatusService"]
