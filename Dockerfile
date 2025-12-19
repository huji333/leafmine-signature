FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

# Build iisignature from source (no wheels for py3.13) by installing toolchain + Boost
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY views/ ./views/
COPY models/ ./models/
COPY controllers/ ./controllers/

RUN mkdir -p data/segmented data/skeletonized

RUN ["uv", "sync", "--locked"]

EXPOSE 7860
CMD ["uv", "run", "--frozen", "--no-sync", "gradio_app"]
