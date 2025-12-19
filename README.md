# About

An asset pipeline for calculating signature of curvilinear leaf mines.

# Usage

## Management UI (Gradio)

1. Run the setup command

   ```
   make setup
   ```

2. Launch the Gradio interface

   ```
   make run
   ```

3. Open http://localhost:7860 in a browser.

### Persisting Saved Images

The app writes uploaded masks to `data/segmented/` and skeletonized results to `data/skeletonized/`. The `make run` command mounts your host `data/` directory into the container, so anything saved in the UI appears immediately on your filesystem under that folder.

To store the files somewhere else, point `DATA_DIR` (or `LEAFMINE_DATA_DIR`) at an absolute path when launching the container:

```
DATA_DIR=/abs/path/to/output make run
```

The custom directory will be bind-mounted into `/app/data`, which is the location the app reads and writes.

### Line Extension Tab

Inside the Gradio UI there is also a **Line Extension** tab. Type the filename of any skeleton stored in `data/skeletonized/` (for example, `skeleton_20251210-064040.png`) and press **Extract Longest Path** to display the red-highlighted overlay inline. The tab surfaces the same polyline JSON that the CLI writes so you can quickly inspect downstream inputs for signature analysis.

## Batch Processing Pre-Segmented Masks

When you want to process many binary masks in one shot, drop the PNGs into `data/segmented/` (or the segmented folder inside your `LEAFMINE_DATA_DIR`) and run the shared pipeline with a single Make target:

```
make process_segmented
```

Need extra flags (for example, only process a few files or rename the CSV prefix)? Pass them via `PIPELINE_ARGS` and they will be forwarded to the CLI:

```
PIPELINE_ARGS="--limit 5 --csv-prefix nightly" make process_segmented
```

By default this target reuses the last built Docker image; set `BUILD=1 make process_segmented` if you need to force a rebuild beforehand.

Under the hood the command reuses the same orchestration code as the UI: it skeletonizes every PNG it finds, saves overlays to `data/skeletonized/` and `data/tmp/`, and writes a timestamped CSV such as `data/signatures/nightly_20251219-153000.csv`.

If you prefer to skip `make`, you can call the CLI directly:

```
uv run python -m controllers.pipeline \
  --data-dir ./data \
  --directions forward reverse \
  --csv-prefix nightly
```

To stay inside the Docker image manually, run:

```
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  leafmine-signature:latest \
  uv run --frozen --no-sync python -m controllers.pipeline --data-dir /app/data
```

## Longest Skeleton Component (WIP)

Use the standalone helper to extract and visualize the longest skeleton path found in any PNG under `data/skeletonized/`:

```
python -m models.longest_component data/skeletonized/your_file.png --out-dir data/tmp
```

The command writes a `_longest.png` overlay (red path highlight) and a `_longest.json` file containing the polyline coordinates plus path statistics to `data/tmp/`.
