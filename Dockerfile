FROM ghcr.io/astral-sh/uv:latest

WORKDIR /app

COPY pyproject.toml uv.lock .

RUN uv sync

COPY gradio-app/

EXPOSE 7860:7860
CMD ["uv", "run", "gradio-app/app.py"]
