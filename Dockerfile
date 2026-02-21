# =============================================================================
# Aegis Agent Platform - Multi-stage Docker Build
# =============================================================================
# Usage:
#   docker build -t aegis:latest .
#   docker build --target dev -t aegis:dev .
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with Python and dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 aegis && \
    useradd --uid 1000 --gid aegis --shell /bin/bash --create-home aegis

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2: Builder stage for installing dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install optional dependencies (can be customized via build args)
ARG INSTALL_OPENAI=true
ARG INSTALL_ANTHROPIC=true
ARG INSTALL_FAISS=false

RUN if [ "$INSTALL_OPENAI" = "true" ]; then pip install openai>=1.10.0; fi && \
    if [ "$INSTALL_ANTHROPIC" = "true" ]; then pip install anthropic>=0.18.0; fi && \
    if [ "$INSTALL_FAISS" = "true" ]; then pip install faiss-cpu>=1.7.4; fi

# -----------------------------------------------------------------------------
# Stage 3: Development image
# -----------------------------------------------------------------------------
FROM base AS dev

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY --chown=aegis:aegis . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Switch to non-root user
USER aegis

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run development server with auto-reload
CMD ["uvicorn", "apps.api_server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# -----------------------------------------------------------------------------
# Stage 4: Production image
# -----------------------------------------------------------------------------
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code only (no dev files)
COPY --chown=aegis:aegis services/ ./services/
COPY --chown=aegis:aegis apps/ ./apps/
COPY --chown=aegis:aegis configs/ ./configs/
COPY --chown=aegis:aegis pyproject.toml .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Switch to non-root user
USER aegis

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run production server with single worker for offline mode
CMD ["uvicorn", "apps.api_server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

# -----------------------------------------------------------------------------
# Stage 5: Offline image (no external API dependencies)
# -----------------------------------------------------------------------------
FROM production AS offline

# Override environment for offline mode
ENV LLM_OFFLINE_MODE=true \
    LLM_DEFAULT_PROVIDER=stub \
    OFFLINE_MODE=true \
    REDIS_ENABLED=false

# =============================================================================
# Default target is production
# =============================================================================
FROM production
