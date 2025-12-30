from __future__ import annotations

import os

import gradio as gr

from controllers.data_paths import DataPaths
from views.config import DataBrowser
from views.tabs import polyline, signature, skeletonization


def main() -> None:
    paths = DataPaths.from_data_dir()
    data_browser = DataBrowser(paths)

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("1. Skeletonize Mask"):
                skeletonization.render(data_paths=paths, data_browser=data_browser)
            with gr.Tab("2. Route Builder"):
                polyline.render(data_paths=paths, data_browser=data_browser)
            with gr.Tab("3. Signatures"):
                signature.render(data_paths=paths, data_browser=data_browser)

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
