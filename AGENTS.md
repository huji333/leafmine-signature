# Overview on this project

## Architecture

### views

Web frontend of app created by gradio.

### models

Backend logics used in the views

### data

Split by steps

### controllers

Thin orchestration layer that keeps filesystem + pipeline logic outside the Gradio tabs. `controllers/pipeline.py` exposes `process_segmented_mask` plus the batch CLI (`python -m controllers.pipeline`), while `controllers/polyline_signatures.py` batches log-signature computation for existing polylines (`python -m controllers.polyline_signatures`).

## Operational Notes

- `make setup` builds the Docker image with all Python deps (iisignature included) and `make run` launches the Gradio management UI on port 7860. Bind `DATA_DIR=/abs/path` (or `LEAFMINE_DATA_DIR`) if the default `./data` mount should live elsewhere.
- Apple silicon / other ARM64 hosts are auto-detected so `make` targets build and run the container as `linux/amd64` via Docker Buildx + QEMU. Override with `BUILD_PLATFORM=` / `RUN_PLATFORM=` if you deliberately want a native ARM image.
- Batch + UI flows stay in sync because both import `controllers.pipeline.process_segmented_mask`. The controller layer is responsible for filesystem side-effects (`data/segmented`, `data/skeletonized`, `data/polylines`, `data/logsig/<timestamp>` per batch run).
- To batch process every PNG inside `data/segmented/`, use `make process_segmented` for the short Docker-backed flow, optionally setting `PIPELINE_ARGS="--limit 5"` to forward flags to the CLI. Add `BUILD=1` only when you intentionally want it to rebuild the image first. Underneath it runs `python -m controllers.pipeline` with the shared controller logic.
- Docker users can reuse the existing image: `docker run --rm -it -v $(pwd)/data:/app/data leafmine-signature:latest uv run --frozen --no-sync python -m controllers.pipeline --data-dir /app/data`.
