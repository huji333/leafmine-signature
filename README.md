# About

An asset pipeline for calculating signature of curvilinear leaf mines.

# Usage

1. Run the setup command

```
make setup
```

2. Launch the Gradio interface

```
make run
```

3. Open http://localhost:7860 in a browser

## Persisting Saved Images

The app writes uploaded masks to `data/segmented/` and skeletonized results to `data/skeletonized/`. The `make run` command now mounts your host `data/` directory into the container, so anything saved in the UI appears immediately on your filesystem under that folder.

To store the files somewhere else, point `DATA_DIR` at an absolute path when launching the container:

```
DATA_DIR=/abs/path/to/output make run
```

The custom directory will be bind-mounted into `/app/data`, which is the location the app reads and writes.

## Longest Skeleton Component (WIP)

Use the standalone helper to extract and visualize the longest skeleton path found in any PNG under `data/skeletonized/`:

```
python -m models.longest_component data/skeletonized/your_file.png --out-dir data/tmp
```

The command writes a `_longest.png` overlay (red path highlight) and a `_longest.json` file containing the polyline coordinates plus path statistics to `data/tmp/`.

## Line Extension Tab

Inside the Gradio UI there is also a **Line Extension** tab. Type the filename of any skeleton stored in `data/skeletonized/` (for example, `skeleton_20251210-064040.png`) and press **Extract Longest Path** to display the red-highlighted overlay inline. The tab surfaces the same polyline JSON that the CLI writes so you can quickly inspect downstream inputs for signature analysis.
