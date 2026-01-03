# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/api/requirements.txt /app/services/api/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r services/api/requirements.txt

# Copy application code
COPY packages/ /app/packages/
COPY services/ /app/services/

# Create storage directory
RUN mkdir -p /app/services/api/storage/contexts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

