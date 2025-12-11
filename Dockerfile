FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY views/ ./views/
COPY models/ ./models/

RUN mkdir -p data/segmented data/skeletonized

RUN ["uv", "sync", "--locked"]

EXPOSE 7860
CMD ["uv", "run", "--frozen", "--no-sync", "gradio_app"]
