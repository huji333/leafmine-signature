# Leafmine Signature

Asset pipeline for calculating log signatures of curvilinear leaf mines.

## Setup

### 1. Building the app

```shell
make setup
```

### 2. Launching the app

```shell
make run
```

Access <http://localhost:7860> on your browser.

### 3. Using the app

## How to use

1. **Skeletonize Mask tab** — upload a new segmented PNG (or pick an existing `segmented_*.png`). Uploaded masks are persisted automatically, previews highlight the skeleton, and multi-component skeletons are flagged so you can tweak preprocessing before moving on.
2. **Route Builder tab** — select a skeleton from `data/skeletonized/`, inspect the pruned graph, and compute a traversal. Start/goal defaults are suggested per component, and the generated polyline JSON lands in `data/polylines/`.
3. **Signatures tab** — choose any subset of stored polylines, set the depth, and click *Compute Signatures*. Each run writes to a brand-new CSV named `data/logsig/logsignatures_<timestamp>.csv`, mirroring the older CLI behavior but entirely from the UI.

## Prepare Sample Annotations

Need sample-level metadata before running UMAP? `make annotation_csv` scans `data/segmented/segmented_*.png`, strips prefixes via the shared naming helpers, and upserts entries in `data/sample_annotation.csv` (preserving any extra columns you’ve added manually). Override the defaults with `SEGMENTED_DIR=...`, `SEGMENTED_GLOB=...`, or `ANNOTATION_CSV=...`, or run `python scripts/generate_sample_annotation.py --help` for advanced options.

## Development

- `UV_CACHE_DIR=.uv-cache uv run --frozen ruff check .` lints the repository, automatically syncing the virtualenv if missing while keeping downloads inside the repo.
- `UV_CACHE_DIR=.uv-cache uv run --frozen ruff check . --fix` auto-applies trivial fixes (matches `make lint` inside the container).
