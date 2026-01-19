"""UMAP visualization helpers for log-signature CSVs and annotations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from controllers.artifacts import ensure_flat_stage_identifier
from controllers.data_paths import DataPaths
from models.umap import rf_proximity_distance, umap_embedding
from models.utils.naming import canonical_sample_name

try:  # pragma: no cover - optional dependency for visualization
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "matplotlib is required for UMAP visualization; install it via `uv pip install matplotlib`."
    ) from exc

 


@dataclass(slots=True)
class UmapConfig:
    n_estimators: int = 800
    min_samples_leaf: int = 5
    max_depth: int | None = None
    n_neighbors: int = 30
    min_dist: float = 0.05
    random_state: int | None = 0
    max_samples: int | None = 500
    canonicalize_keys: bool = True


@dataclass(slots=True)
class UmapResult:
    figure: object
    table_rows: list[list[object]]
    table_headers: list[str]
    status: str
    csv_path: Path | None = None


def list_logsignatures(data_paths: DataPaths) -> list[str]:
    return [path.name for path in _list_csvs(data_paths.signatures_dir)]


def list_annotations(data_paths: DataPaths) -> list[str]:
    return [path.name for path in _list_csvs(data_paths.annotations_dir)]


def load_csv_columns(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or []


def compute_umap_flow(
    *,
    data_paths: DataPaths,
    logsig_entry: str | None,
    annotation_entry: str | None,
    color_column: str,
    config: UmapConfig,
) -> UmapResult:
    if not logsig_entry:
        raise ValueError("Select a log-signature CSV first.")
    if not annotation_entry:
        raise ValueError("Select an annotation CSV first.")

    logsig_path = _resolve_logsig_path(logsig_entry, data_paths.signatures_dir)
    annotation_path = _resolve_annotation_path(annotation_entry, data_paths.annotations_dir)

    logs, keys, vectors = _load_logsignatures(
        logsig_path,
        canonicalize=config.canonicalize_keys,
    )
    ann_map, ann_logs = _load_annotation_map(
        annotation_path,
        value_column=color_column,
        canonicalize=config.canonicalize_keys,
    )

    matched_keys: list[str] = []
    matched_vectors: list[np.ndarray] = []
    matched_labels: list[str] = []
    missing = 0
    for key, vector in zip(keys, vectors):
        label = ann_map.get(key)
        if label is None:
            missing += 1
            continue
        matched_keys.append(key)
        matched_vectors.append(vector)
        matched_labels.append(label)

    if not matched_vectors:
        raise ValueError(
            "No overlapping samples between log-signatures and annotations. "
            "Check the join columns or disable canonicalization."
        )

    X = np.vstack(matched_vectors)
    labels = matched_labels
    keys = matched_keys

    if config.max_samples and X.shape[0] > config.max_samples:
        original_count = X.shape[0]
        rng = np.random.default_rng(config.random_state)
        indices = rng.choice(X.shape[0], size=config.max_samples, replace=False)
        X = X[indices]
        labels = [labels[i] for i in indices]
        keys = [keys[i] for i in indices]
        logs.append(f"[info] sampled {config.max_samples} / {original_count} rows")

    if X.shape[0] < 3:
        raise ValueError("Need at least 3 samples after matching to run UMAP.")

    distance = rf_proximity_distance(
        X,
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )
    embedding = umap_embedding(
        distance,
        n_neighbors=int(config.n_neighbors),
        min_dist=float(config.min_dist),
        random_state=config.random_state,
    )

    figure, color_info = _plot_embedding(embedding, labels, color_column)
    table_headers = ["sample_key", "label", "umap_x", "umap_y"]
    table_rows = _build_table_rows(keys, labels, embedding)
    csv_path = _write_umap_csv(
        output_dir=data_paths.umap_dir,
        logsig_path=logsig_path,
        annotation_path=annotation_path,
        color_column=color_column,
        headers=table_headers,
        rows=table_rows,
    )

    status_lines = [
        f"Loaded {len(vectors)} log-signature rows from `{logsig_path.name}`.",
        f"Matched {len(keys)} rows with `{annotation_path.name}` "
        f"using {color_column}.",
    ]
    if missing:
        status_lines.append(f"[info] {missing} rows had no annotation match")
    status_lines.extend(logs)
    status_lines.extend(ann_logs)
    status_lines.append(color_info)
    if csv_path is not None:
        status_lines.append(f"Saved UMAP CSV to `{csv_path.name}`.")

    return UmapResult(
        figure=figure,
        table_rows=table_rows,
        table_headers=table_headers,
        status="\n".join(status_lines),
        csv_path=csv_path,
    )


def _list_csvs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.csv") if path.is_file())


def _resolve_logsig_path(entry: str, signatures_dir: Path) -> Path:
    ensure_flat_stage_identifier(entry, description="Log-signature CSV")
    candidate = Path(entry).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = (signatures_dir / candidate).with_suffix(candidate.suffix or ".csv")
    if resolved.exists():
        return resolved
    raise ValueError(f"Log-signature CSV {entry} was not found in {signatures_dir}.")


def _resolve_annotation_path(entry: str, annotations_dir: Path) -> Path:
    ensure_flat_stage_identifier(entry, description="Annotation CSV")
    candidate = Path(entry).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = (annotations_dir / candidate).with_suffix(candidate.suffix or ".csv")
    if resolved.exists():
        return resolved
    raise ValueError(f"Annotation CSV {entry} was not found in {annotations_dir}.")


def _load_logsignatures(
    csv_path: Path,
    *,
    canonicalize: bool,
) -> tuple[list[str], list[str], list[np.ndarray]]:
    logs: list[str] = []
    keys: list[str] = []
    vectors: list[np.ndarray] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        key_column = "filename"
        if key_column not in header:
            raise ValueError("Log-signature CSV missing column 'filename'.")
        if "log_signature" not in header:
            raise ValueError("Log-signature CSV missing 'log_signature' column.")
        for row in reader:
            raw_key = (row.get(key_column) or "").strip()
            if not raw_key:
                continue
            key = canonical_sample_name(raw_key) if canonicalize else raw_key
            raw_vec = (row.get("log_signature") or "").strip()
            if not raw_vec:
                continue
            try:
                values = json.loads(raw_vec)
            except json.JSONDecodeError:
                logs.append(f"[warn] skipped malformed log_signature for {raw_key}")
                continue
            vector = np.asarray(values, dtype=np.float64).ravel()
            if vector.size == 0:
                logs.append(f"[warn] skipped empty log_signature for {raw_key}")
                continue
            keys.append(key)
            vectors.append(vector)
    if not vectors:
        raise ValueError("No log-signature rows found in the CSV.")
    dim = vectors[0].size
    filtered_keys: list[str] = []
    filtered_vectors: list[np.ndarray] = []
    dropped = 0
    for key, vector in zip(keys, vectors):
        if vector.size != dim:
            dropped += 1
            continue
        filtered_keys.append(key)
        filtered_vectors.append(vector)
    if dropped:
        logs.append(f"[warn] dropped {dropped} rows with inconsistent dimensions")
    return logs, filtered_keys, filtered_vectors


def _load_annotation_map(
    csv_path: Path,
    *,
    value_column: str,
    canonicalize: bool,
) -> tuple[dict[str, str], list[str]]:
    logs: list[str] = []
    mapping: dict[str, str] = {}
    duplicates = 0
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        key_column = "sample_id"
        if key_column not in header:
            raise ValueError("Annotation CSV missing column 'sample_id'.")
        if value_column not in header:
            raise ValueError(f"Annotation CSV missing column '{value_column}'.")
        for row in reader:
            raw_key = (row.get(key_column) or "").strip()
            if not raw_key:
                continue
            key = canonical_sample_name(raw_key) if canonicalize else raw_key
            value = (row.get(value_column) or "").strip()
            if key in mapping:
                duplicates += 1
                continue
            mapping[key] = value
    if duplicates:
        logs.append(f"[info] ignored {duplicates} duplicate annotation keys")
    return mapping, logs


def _plot_embedding(
    embedding: np.ndarray,
    labels: list[str],
    color_label: str,
) -> tuple[object, str]:
    fig, ax = plt.subplots(figsize=(6, 6))
    numeric_values = _coerce_numeric(labels)
    if numeric_values is not None:
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=numeric_values,
            cmap="viridis",
            s=20,
            alpha=0.9,
        )
        fig.colorbar(scatter, ax=ax, label=color_label)
        color_info = f"[color] numeric values from '{color_label}'"
    else:
        categories, color_map = _categorical_color_map(labels)
        colors = [color_map[label] for label in labels]
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            s=20,
            alpha=0.9,
        )
        color_info = f"[color] categorical values from '{color_label}' ({len(categories)} classes)"
        if len(categories) <= 20:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=category,
                    markerfacecolor=color_map[category],
                    markersize=6,
                )
                for category in categories
            ]
            ax.legend(handles=handles, title=color_label, bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            color_info += "; legend omitted (>20 categories)"
    ax.set_title("UMAP (RF proximity)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig, color_info


def _categorical_color_map(values: Iterable[str]) -> tuple[list[str], dict[str, tuple[float, float, float, float]]]:
    categories: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            categories.append(value)
            seen.add(value)
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    return categories, {category: cmap(idx) for idx, category in enumerate(categories)}


def _coerce_numeric(values: Iterable[str]) -> list[float] | None:
    numeric: list[float] = []
    for value in values:
        if value is None or value == "":
            numeric.append(float("nan"))
            continue
        try:
            numeric.append(float(value))
        except ValueError:
            return None
    return numeric


def _build_table_rows(
    keys: list[str],
    labels: list[str],
    embedding: np.ndarray,
) -> list[list[object]]:
    rows: list[list[object]] = []
    for key, label, coords in zip(keys, labels, embedding):
        rows.append([key, label, float(coords[0]), float(coords[1])])
    return rows


def _write_umap_csv(
    *,
    output_dir: Path,
    logsig_path: Path,
    annotation_path: Path,
    color_column: str,
    headers: list[str],
    rows: list[list[object]],
) -> Path | None:
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    logsig_slug = _safe_slug(logsig_path.stem, "logsig")
    annotation_slug = _safe_slug(annotation_path.stem, "annotation")
    color_slug = _safe_slug(color_column, "color")
    filename = f"umap_{logsig_slug}_{annotation_slug}_{color_slug}_{stamp}.csv"
    csv_path = output_dir / filename
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return csv_path


def _safe_slug(value: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


__all__ = [
    "UmapConfig",
    "UmapResult",
    "compute_umap_flow",
    "list_logsignatures",
    "list_annotations",
    "load_csv_columns",
]
