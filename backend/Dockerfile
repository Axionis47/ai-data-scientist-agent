FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for scientific stack (use OpenBLAS/LAPACK; ATLAS is not available on Debian bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran \
    libopenblas-dev liblapack-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata and install runtime deps
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

# Copy application code
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

