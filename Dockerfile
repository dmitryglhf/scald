FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv pip install --system --no-cache --break-system-packages -e . --only-group main

FROM node:20-slim AS node-base

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=node-base /usr/local/bin/node /usr/local/bin/
COPY --from=node-base /usr/local/bin/npm /usr/local/bin/
COPY --from=node-base /usr/local/bin/npx /usr/local/bin/
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /usr/local/bin/

WORKDIR /app

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:$PYTHONPATH

CMD ["/bin/bash"]
