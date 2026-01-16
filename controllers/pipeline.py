"""Batch helpers for running skeletonization."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from controllers.artifacts import ensure_flat_stage_identifier, resolve_segmented_mask_path
from controllers.data_paths import DataPaths
from controllers.skeletonization import (
    SkeletonizationConfig,
    process_mask,
    resolve_mask_source,
)
from models.utils.naming import canonical_sample_name, prefixed_name


def run_pipeline_flow(
    *,
    data_paths: DataPaths,
    selected_files: Sequence[str] | None,
    uploaded_files: Sequence[str] | None,
    smooth_radius: float | int,
    hole_area: float | int,
    erode_radius: float | int,
    skip_existing_skeleton: bool,
) -> tuple[list[list[str]], list[str], str]:
    """Run skeletonization for selected masks."""

    if not selected_files and not uploaded_files:
        raise ValueError("Select or upload at least one segmented mask before running.")

    cfg = data_paths
    cfg.ensure_directories()

    skeleton_config = SkeletonizationConfig(
        smooth_radius=int(smooth_radius),
        hole_area_threshold=int(hole_area),
        erode_radius=int(erode_radius),
    )
    headers = ["sample_id", "segmented", "skeleton", "status"]
    rows: list[list[str]] = []
    ok = 0
    skipped = 0
    errors = 0
    logs: list[str] = []

    selected_files = list(selected_files or [])
    uploaded_files = list(uploaded_files or [])

    for entry in selected_files:
        rows, ok, skipped, errors = _process_entry(
            entry=str(entry),
            cfg=cfg,
            skeleton_config=skeleton_config,
            skip_existing_skeleton=skip_existing_skeleton,
            from_upload=False,
            rows=rows,
            ok=ok,
            skipped=skipped,
            errors=errors,
            logs=logs,
        )

    for entry in uploaded_files:
        rows, ok, skipped, errors = _process_entry(
            entry=str(entry),
            cfg=cfg,
            skeleton_config=skeleton_config,
            skip_existing_skeleton=skip_existing_skeleton,
            from_upload=True,
            rows=rows,
            ok=ok,
            skipped=skipped,
            errors=errors,
            logs=logs,
        )

    summary = (
        f"Processed {len(selected_files)} file(s): ok {ok}, "
        f"skipped {skipped}, errors {errors}."
    )
    config_line = (
        "Skeleton config: "
        f"closing={skeleton_config.smooth_radius}, "
        f"hole<={skeleton_config.hole_area_threshold}, "
        f"erode={skeleton_config.erode_radius}."
    )
    summary_lines = [summary, config_line]
    if logs:
        summary_lines.append("\n".join(logs))
    return rows, headers, "\n".join(summary_lines)


def _resolve_segmented_path(entry: str, segmented_dir: Path) -> Path:
    ensure_flat_stage_identifier(entry, description="Segmented filename")
    candidate = resolve_segmented_mask_path(
        entry,
        [segmented_dir],
        default_suffix=".png",
    )
    if candidate is None:
        raise ValueError(f"Segmented mask {entry} was not found in {segmented_dir}.")
    return candidate


def _process_entry(
    *,
    entry: str,
    cfg: DataPaths,
    skeleton_config: SkeletonizationConfig,
    skip_existing_skeleton: bool,
    from_upload: bool,
    rows: list[list[str]],
    ok: int,
    skipped: int,
    errors: int,
    logs: list[str],
) -> tuple[list[list[str]], int, int, int]:
    sample_id = canonical_sample_name(entry)
    segmented_path = None
    skeleton_path = cfg.skeleton_dir / prefixed_name("skeletonized", sample_id, ".png")
    status_parts: list[str] = []

    try:
        if from_upload:
            if skip_existing_skeleton and skeleton_path.exists():
                status_parts.append("skeleton: skipped")
                skipped += 1
            else:
                mask_image, source_name, existing_mask_path = resolve_mask_source(
                    cfg,
                    uploaded_file=entry,
                    selected_filename=None,
                )
                skeleton_result = process_mask(
                    mask_image,
                    original_name=source_name,
                    data_paths=cfg,
                    config=skeleton_config,
                    persist_mask=True,
                    existing_mask_path=existing_mask_path,
                )
                segmented_path = skeleton_result.mask_path
                skeleton_path = skeleton_result.skeleton_path
                status_parts.append("skeletonized")
        else:
            if not skip_existing_skeleton or not skeleton_path.exists():
                mask_image, source_name, existing_mask_path = resolve_mask_source(
                    cfg,
                    uploaded_file=None,
                    selected_filename=entry,
                )
                skeleton_result = process_mask(
                    mask_image,
                    original_name=source_name,
                    data_paths=cfg,
                    config=skeleton_config,
                    persist_mask=False,
                    existing_mask_path=existing_mask_path,
                )
                segmented_path = existing_mask_path
                skeleton_path = skeleton_result.skeleton_path
                status_parts.append("skeletonized")
            else:
                segmented_path = _resolve_segmented_path(entry, cfg.segmented_dir)
                status_parts.append("skeleton: skipped")
                skipped += 1

        ok += 1
    except Exception as exc:
        errors += 1
        status_parts = [f"error: {exc}"]
        logs.append(f"[error] {entry}: {exc}")

    rows.append(
        [
            sample_id or (Path(entry).stem if entry else ""),
            segmented_path.name if segmented_path else str(entry),
            skeleton_path.name if skeleton_path else "",
            "; ".join(status_parts) if status_parts else "ok",
        ]
    )

    return rows, ok, skipped, errors


__all__ = ["run_pipeline_flow"]
