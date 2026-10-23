FROM python:3.12-slim

# LightGBM links against OpenMP, which the slim image does not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so editing source does not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

# The MLflow database and its artifacts live on a mounted volume so a registered model
# survives the container. Create the store from inside the container (the train command),
# because the artifact paths MLflow records are absolute.
VOLUME /app/mlstore
ENV MLOPS_TRACKING_URI=sqlite:////app/mlstore/mlflow.db \
    MLOPS_PREDLOG_PATH=/app/mlstore/predlog.sqlite

EXPOSE 8000
CMD ["mlops-pipeline", "serve", "--host", "0.0.0.0", "--port", "8000"]
