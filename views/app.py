from __future__ import annotations

import os

import gradio as gr

from views.tabs import pipeline, skeletonization


def main() -> None:
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("Pipeline"):
                pipeline.render()
            with gr.Tab("Skeletonization"):
                skeletonization.render()

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
