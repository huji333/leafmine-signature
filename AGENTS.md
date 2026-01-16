# Overview on this project

## Philosophy

Keep it Simple

## Architecture

### views

Gradio Blocks front-end. Mounts reusable components (e.g., dropdowns) from `views/components`, talks only to controllers, and keeps UI state (selectors, tables) within each tab.

### models

Pure data/algorithm layer: image preprocessing, graph construction, route traversal, and log-signature math. Controllers are the only layer importing models.

### data

Filesystem layout for pipeline artifacts: `segmented/`, `skeletonized/`, `polylines/`, `graphs/`, `logsig/`. Managed through `controllers.data_paths` so both UI and CLI scripts share identical discovery logic.

### controllers

Thin orchestration layer focused on the concrete Gradio tabs (skeletonization, polyline routing, signature batching).
- `settings.py` resolves `config.toml`.
- `data_paths.py` exposes canonical directories plus helpers for listing artifacts and sample IDs.
- `skeletonization.py` manages mask uploads, config defaults, and preprocessing.
- `bulk_skeletonization.py` runs batch skeletonization for selected/uploaded masks.
- `polyline_graph.py` builds/memoizes graphs and emits `.json/.meta.json` files.
- `polyline_route.py` consumes a `GraphSession` to compute traversal + polyline artifacts.
- `signature.py` batches log-signature computation into per-run CSVs.

## Architecture Notes

- `controllers.settings` centralizes reading `config.toml` and exposes the resolved data directory; all filesystem helpers (`controllers.data_paths.DataPaths`, `fetch_artifact_paths`, `list_canonical_sample_ids`) build on that single source of truth.
- The UI (`views/`) is purely a Gradio Blocks layer that calls the controllers. Shared widgets (e.g., dropdowns) live under `views/components`, which depend on the `DataBrowser` helper to surface the latest files from each stage directory.
- `controllers/skeletonization.py` owns the mask preprocessing pipeline and now exposes `DEFAULT_SKELETON_CONFIG`, so views no longer reach into `models.skeletonization`.
- Skeleton routing flows through two controllers: `controllers.polyline_graph` builds/serializes graphs (including a `.meta.json` manifest) and hands a lightweight `GraphSession` to `controllers.polyline_route`, which computes traversal orders and writes polylines without reloading skeleton PNGs.
- `controllers.signature` stays scoped to `data/polylines` and writes timestamped CSVs into `data/logsig`, mirroring the older CLI but entirely via the UI.
