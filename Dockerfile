# deps image: Python deps
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS deps

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen

# runtime image: copy venv + sources
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS runtime

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY config.toml ./
COPY views/ ./views/
COPY models/ ./models/
COPY controllers/ ./controllers/
# install project in editable mode for gradio entrypoints
RUN uv pip install --no-deps --editable .

EXPOSE 7860
CMD ["uv", "run", "--frozen", "--no-sync", "gradio_app"]
