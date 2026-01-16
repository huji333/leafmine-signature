# Leafmine Signature

Asset pipeline for calculating log signatures of curvilinear leaf mines.

## Launch the Management UI

1. `make setup` – builds the image. Apple silicon/ARM64 hosts automatically force `docker buildx build --load --platform linux/amd64`; override with `BUILD_PLATFORM=` or `BUILD_PLATFORM=linux/arm64` plus `BUILD_CMD="docker build"` if you want something else.
2. `make run` – brings up Gradio on http://localhost:7860. The default `RUN_PLATFORM` matches the image; set `RUN_PLATFORM=` (or another value) only when you intentionally built for a different architecture.
3. Need a different data mount? `DATA_DIR=/abs/path make run`.

## Configure the Data Directory

`controllers.settings.load_data_dir()` reads `config.toml` in the repo root (falling back to `data/`). Edit the `[leafmine] data_dir = "..."` entry to relocate artifacts; leave it alone for the default `data/`.

- Relative paths are resolved relative to `config.toml`, so the tracked `data` value behaves exactly like the old hardcoded paths.
- Absolute paths or `~/` expansions work too—handy when you mount a host directory into Docker and copy the same `config.toml` into the container.
- For Docker images, copy the same `config.toml` alongside the app (the default one already lives next to the code) and mount your preferred data directory into `/app/data`.

## Guided Pipeline Flow (UI Only)

1. **Skeletonize Mask tab** — upload a new segmented PNG (or pick an existing one). Uploaded masks are persisted automatically, previews highlight the skeleton, and multi-component skeletons are flagged so you can tweak preprocessing before moving on.
2. **Route Builder tab** — select a skeleton from `data/skeletonized/`, independently tune branch and loop pruning thresholds, inspect the pruned graph, and compute a traversal. Start/goal defaults are suggested per component, and the generated polyline JSON lands in `data/polylines/`.
3. **Signatures tab** — choose any subset of stored polylines, set the depth, and click *Compute Signatures*. Each run writes to a brand-new CSV named `data/logsig/logsignatures_<timestamp>.csv`, mirroring the older CLI behavior but entirely from the UI.

## Skeletonize Masks Quickly

The **Skeletonize Mask** tab lets you upload a fresh segmented mine or select an existing file from `data/segmented/`. Every upload is persisted with the canonical prefix, the preview thickens the skeleton overlay for quick QA, and the tab flags multi-component skeletons before you move on to routing.

## Batch Signatures In-UI

Use the **Signatures** tab to recompute log signatures for any subset of `data/polylines/*.json`. All polylines are selected by default—uncheck any you want to skip, adjust depth, and each run appends rows to its own timestamped CSV while showing a preview of the latest entries.

## Development

- `UV_CACHE_DIR=.uv-cache uv run --frozen ruff check .` lints the repository, automatically syncing the virtualenv if missing while keeping downloads inside the repo.
- `UV_CACHE_DIR=.uv-cache uv run --frozen ruff check . --fix` auto-applies trivial fixes (matches `make lint` inside the container).
