FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy package metadata + minimal source for build backend
COPY pyproject.toml ./
RUN mkdir -p src/ctra
COPY src/ctra/__init__.py ./src/ctra/__init__.py

# Install dependencies (cached layer — only rebuilds when pyproject.toml changes)
ARG EXTRAS="api,dashboard"
RUN uv pip install --system --no-cache ".[$EXTRAS]"

# Copy full source
COPY src/ ./src/

# Re-install package itself (no-deps since deps already installed)
RUN uv pip install --system --no-cache --no-deps .

# Create non-root user
RUN useradd --create-home ctra
USER ctra

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); r.raise_for_status()" || exit 1

CMD ["ctra"]
