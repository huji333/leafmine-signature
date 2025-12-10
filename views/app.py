import gradio as gr


def analyze_leaf_mine(image):
    return "Hello, World!"


def main():
    gr.Interface(fn=analyze_leaf_mine, inputs="image", outputs="text").launch(
        server_name="0.0.0.0",  # allow access from outside the container
        server_port=7860,
    )


if __name__ == "__main__":
    main()
