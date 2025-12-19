# Leafmine Signature

Asset pipeline for calculating signatures of curvilinear leaf mines.

## Launch the Management UI

1. Build the Docker image (only once unless dependencies change):

   ```
   make setup
   ```

2. Start the Gradio admin panel:

   ```
   make run
   ```

3. Open http://localhost:7860. The container bind-mounts `./data` by default; override with `DATA_DIR=/abs/path make run` if you want outputs elsewhere.

## Run The Batch Pipeline

1. Place pre-segmented binary PNGs in `data/segmented/` (or the same folder under your custom `DATA_DIR`).
2. Execute the shared controller pipeline across every file:

   ```
   make process_segmented
   ```

3. Optional knobs:
   - Forward extra CLI flags (e.g., limit, CSV prefix) with `PIPELINE_ARGS="--limit 5 --csv-prefix nightly" make process_segmented`.
   - Add `BUILD=1` when you specifically want to rebuild the Docker image before running.

The command reuses the same controller as the UI to skeletonize each mask, emit overlays, and append signatures to a timestamped CSV under `data/signatures/`.
