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
