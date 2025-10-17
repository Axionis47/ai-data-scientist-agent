# ============================================================================
# Bot Data Scientist - Production Dockerfile
# ============================================================================
# Multi-stage build for optimized image size and security
# ============================================================================

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 botds && \
    chown -R botds:botds /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/botds/.local

# Copy application code
COPY --chown=botds:botds . .

# Set environment variables
ENV PATH=/home/botds/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER botds

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import botds; print('OK')" || exit 1

# Default command - run the CLI help
CMD ["python", "-m", "cli.run", "--help"]

# ============================================================================
# Build Instructions:
# ============================================================================
# docker build -t botds:latest .
# docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY botds:latest
# ============================================================================

