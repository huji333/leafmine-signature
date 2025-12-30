# Leafmine Signature

Asset pipeline for calculating log signatures of curvilinear leaf mines.

## Launch the Management UI

1. `make setup` – builds the image. Apple silicon/ARM64 hosts automatically force `docker buildx build --load --platform linux/amd64`; override with `BUILD_PLATFORM=` or `BUILD_PLATFORM=linux/arm64` plus `BUILD_CMD="docker build"` if you want something else.
2. `make run` – brings up Gradio on http://localhost:7860. The default `RUN_PLATFORM` matches the image; set `RUN_PLATFORM=` (or another value) only when you intentionally built for a different architecture.
3. Need a different data mount? `DATA_DIR=/abs/path make run`.

## Run The Batch Pipeline

1. Drop segmented PNGs into `data/segmented/`.
2. Run `make process_segmented` (add `PIPELINE_ARGS="--limit 5"` or similar when needed).
3. Set `BUILD=1` to rebuild the image before executing, or adjust `DATA_DIR` like in the UI flow.

Artifacts land under `data/*` following the usual `segmented → skeletonized → polylines → logsig` chain, and every run emits a fresh CSV like `data/logsig/logsignatures_<timestamp>.csv`.

## Analyze Existing Polylines

`make analyze_polylines` scans `data/polylines/*.json` and appends new rows to a timestamped CSV under `data/logsig/logsignatures_<timestamp>.csv`. Limit or retarget the run with `POLYLINE_ARGS` (e.g., `POLYLINE_ARGS="--depth 5 foo.json"` or `POLYLINE_ARGS="--overwrite"`).

## Skeletonize Masks Quickly

The **Skeletonize Mask** tab lets you upload a fresh segmented mine or select an existing `segmented_*.png` from `data/segmented/`. Every upload is persisted with the canonical prefix, the preview thickens the skeleton overlay for quick QA, and the tab flags multi-component skeletons before you move on to routing.

## Batch Signatures In-UI

Use the **Signatures** tab to recompute log signatures for any subset of `data/polylines/*.json`. All polylines are selected by default—uncheck any you want to skip, adjust depth, and the tab will append rows to the same CSV that `make analyze_polylines` writes while showing a live preview of the latest entries.
