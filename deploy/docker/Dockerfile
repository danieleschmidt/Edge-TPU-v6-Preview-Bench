# Edge TPU v6 Benchmark Suite - Production Dockerfile
FROM python:3.11-slim-bullseye

# Metadata
LABEL maintainer="Terragon Labs <support@terragonlabs.ai>"
LABEL version="1.0.0"
LABEL description="Edge TPU v6 Benchmarking Suite - Production Ready"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd -r edgetpu && useradd -r -g edgetpu -m edgetpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Core system tools
    curl \
    wget \
    gnupg2 \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    # Build tools
    build-essential \
    pkg-config \
    # Hardware support
    libusb-1.0-0-dev \
    libudev-dev \
    # Monitoring tools
    htop \
    iotop \
    # Network tools
    netcat-openbsd \
    # Security tools
    fail2ban \
    && rm -rf /var/lib/apt/lists/*

# Install Edge TPU runtime (if available)
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    (apt-get install -y libedgetpu1-std || echo "Edge TPU runtime not available, continuing...") && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package
RUN pip install -e .

# Create necessary directories with correct permissions
RUN mkdir -p /app/data /app/logs /app/cache /app/models /app/results && \
    chown -R edgetpu:edgetpu /app

# Copy configuration files
COPY deploy/config/ ./config/

# Health check script
COPY deploy/docker/healthcheck.py ./
RUN chmod +x healthcheck.py

# Switch to non-root user
USER edgetpu

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py || exit 1

# Default command
CMD ["python", "-m", "edge_tpu_v6_bench.cli", "server", "--host", "0.0.0.0", "--port", "8080"]