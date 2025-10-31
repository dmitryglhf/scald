FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY . .

RUN uv pip install --system --no-cache --break-system-packages -e . --group dev --group docs --group gaia

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:$PYTHONPATH
