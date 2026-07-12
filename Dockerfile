# ============================================================
# SERA — Self-Evolving RAG Assistant
# Multi-stage Dockerfile
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
# Installs all Python dependencies in isolation so the final
# image only contains what is strictly needed at runtime.
FROM python:3.10-slim AS builder

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS build tools required by some Python packages
# (gcc for scikit-learn C extensions, git for pip VCS installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first so Docker caches this layer separately.
# pip install only re-runs when requirements.txt changes.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: spaCy model download ───────────────────────────
# Runs in a separate stage so the model download is also cached.
FROM builder AS spacy-downloader

COPY --from=builder /install /install
ENV PYTHONPATH=/install/lib/python3.10/site-packages

RUN python -m spacy download en_core_web_sm


# ── Stage 3: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime OS packages (no compilers needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed packages from builder and spaCy model from downloader
COPY --from=builder  /install /usr/local
COPY --from=spacy-downloader /root/.local /root/.local

# Copy the application source code
COPY app/        ./app/
COPY tests/      ./tests/
COPY scripts/    ./scripts/
COPY workflows/  ./workflows/

# Create the data directory layout.
# This is where ChromaDB and SQLite persist their files.
# In production this directory is MOUNTED as a Docker volume
# so data survives container restarts and rebuilds.
RUN mkdir -p data/sqlite data/chroma data/cache data/backups

# Expose the FastAPI port
EXPOSE 8000

# Healthcheck — polls the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start uvicorn
# Use --host 0.0.0.0 so the container exposes the port to the outside world.
# Workers=1 is intentional: APScheduler (Phase 3) must run in a single process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
