"""Algorithm utilities for Random Forest proximity and UMAP embedding."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency for UMAP
    import umap
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "umap-learn is required for UMAP embedding; install it via `uv pip install umap-learn`."
    ) from exc

try:  # pragma: no cover - optional dependency for RandomForest
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "scikit-learn is required for RF proximity; install it via `uv pip install scikit-learn`."
    ) from exc


def rf_proximity_distance(
    X: np.ndarray,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None,
    random_state: int | None,
    max_samples: int | None = 5000,
) -> np.ndarray:
    """Compute RF proximity distance matrix (1 - proximity)."""

    n_samples, n_features = X.shape
    if max_samples and n_samples > max_samples:
        raise ValueError("Too many samples for RF proximity; reduce max samples.")
    rng = np.random.default_rng(random_state)
    synth = X.copy()
    for j in range(n_features):
        rng.shuffle(synth[:, j])

    X_all = np.vstack([X, synth])
    y_all = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_all, y_all)

    leaf = rf.apply(X)
    n_trees = leaf.shape[1]
    proximity = np.zeros((n_samples, n_samples), dtype=np.float32)
    for tree_idx in range(n_trees):
        leaf_values = leaf[:, tree_idx]
        order = np.argsort(leaf_values, kind="mergesort")
        sorted_leaf = leaf_values[order]
        start = 0
        while start < n_samples:
            end = start + 1
            while end < n_samples and sorted_leaf[end] == sorted_leaf[start]:
                end += 1
            group = order[start:end]
            proximity[np.ix_(group, group)] += 1.0
            start = end
    proximity /= float(n_trees)
    return 1.0 - proximity


def umap_embedding(
    distance: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int | None,
) -> np.ndarray:
    """Run UMAP on a precomputed distance matrix."""

    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("UMAP expects a square distance matrix.")
    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=random_state,
    )
    return reducer.fit_transform(distance)


__all__ = ["rf_proximity_distance", "umap_embedding"]
