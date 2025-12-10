import gradio as gr


def render() -> None:
    """Placeholder pipeline tab until other stages are implemented."""

    gr.Markdown("## Segmentation → Skeletonization → Polyline → Signature")
    with gr.Row():
        with gr.Column():
            gr.Image(label="Input", value=None)
    gr.Button("Run", variant="secondary", interactive=False)
    with gr.Row():
        with gr.Column():
            gr.Image(label="Segmented", value=None)
        with gr.Column():
            gr.Image(label="Skeletonized", value=None)
    gr.Textbox(label="Signature", value="", interactive=False)


__all__ = ["render"]
