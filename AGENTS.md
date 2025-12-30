# Overview on this project

## Architecture

### views

Web frontend of app created by gradio.

### models

Backend logics used in the views

### data

Split by steps

### controllers

Thin orchestration layer that keeps filesystem + pipeline logic outside the Gradio tabs. `controllers/pipeline.process_segmented_mask` and the helpers in `controllers/polyline.py` / `controllers/polyline_signatures.py` are invoked directly from the UI; we no longer ship separate CLI entrypoints.

## Operational Notes

- `make setup` builds the Docker image with all Python deps (iisignature included) and `make run` launches the Gradio management UI on port 7860. Bind `DATA_DIR=/abs/path` (or `LEAFMINE_DATA_DIR`) if the default `./data` mount should live elsewhere.
- Apple silicon / other ARM64 hosts are auto-detected so `make` targets build and run the container as `linux/amd64` via Docker Buildx + QEMU. Override with `BUILD_PLATFORM=` / `RUN_PLATFORM=` if you deliberately want a native ARM image.
- Batch flows now run entirely through the Gradio UI, which imports the same controller functions used previously by the CLI. The controller layer is responsible for filesystem side-effects (`data/segmented`, `data/skeletonized`, `data/polylines`, `data/logsig/logsignatures_<timestamp>.csv` per batch run).
