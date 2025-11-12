import gradio as gr

def analyze_leaf_mine(image):
    return "Hello, World!"

gr.Interface(fn=analyze_leaf_mine, inputs="image", outputs="text").launch()
