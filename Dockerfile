# deps image: build toolchain + Python deps
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS deps

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_ESSENTIAL_VERSION=12.9
ARG PYTHON_DEV_VERSION=3.11.2-1+b1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=${BUILD_ESSENTIAL_VERSION} \
        python3-dev=${PYTHON_DEV_VERSION} \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen

# runtime image: copy venv + sources
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS runtime

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY views/ ./views/
COPY models/ ./models/
COPY controllers/ ./controllers/
# install project in editable mode for gradio entrypoints
RUN uv pip install --no-deps --editable .

EXPOSE 7860
CMD ["uv", "run", "--frozen", "--no-sync", "gradio_app"]
