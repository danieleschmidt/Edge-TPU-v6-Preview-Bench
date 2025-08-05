# Multi-stage Docker build for Edge TPU v6 Benchmark Suite
# Production-ready container with security and optimization

# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure poetry: Don't create virtual env, install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy source code
COPY . .

# Build the package
RUN poetry build

# Runtime stage
FROM python:3.10-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/edgetpu/.local/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Core utilities
    curl \
    wget \
    gnupg2 \
    # Edge TPU runtime dependencies
    udev \
    libusb-1.0-0 \
    # Optional: for advanced power monitoring
    i2c-tools \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r edgetpu && useradd -r -g edgetpu -m edgetpu

# Install Edge TPU runtime (Coral libraries)
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y \
        libedgetpu1-std \
        python3-pycoral \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER edgetpu
WORKDIR /home/edgetpu

# Copy built package from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the package
RUN pip install --user *.whl \
    && rm *.whl

# Create directories for data and results
RUN mkdir -p ./models ./results ./config

# Copy configuration files
COPY --chown=edgetpu:edgetpu docker/config/ ./config/

# Set up volumes
VOLUME ["/home/edgetpu/models", "/home/edgetpu/results", "/home/edgetpu/config"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD edge-tpu-v6-bench devices || exit 1

# Default command
CMD ["edge-tpu-v6-bench", "--help"]

# Labels for metadata
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>" \
      version="0.1.0" \
      description="Edge TPU v6 Benchmark Suite - Production Container" \
      org.opencontainers.image.title="Edge TPU v6 Benchmark Suite" \
      org.opencontainers.image.description="Comprehensive benchmarking for Edge TPU v6 devices" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/Edge-TPU-v6-Preview-Bench"