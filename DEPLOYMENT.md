# Edge TPU v6 Benchmark Suite - Production Deployment Guide

## 🚀 Overview

This document provides comprehensive guidance for deploying the Edge TPU v6 Benchmark Suite in production environments. The system is designed for enterprise-grade deployments with high availability, security, and scalability.

## 📋 Prerequisites

### Hardware Requirements

**Minimum Requirements:**
- CPU: 4 cores (x86_64 or ARM64)
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps connection

**Recommended for Production:**
- CPU: 8+ cores with Edge TPU support
- RAM: 16GB+
- Storage: 200GB+ NVMe SSD
- Network: 10Gbps connection
- Edge TPU devices: Coral Dev Board, USB Accelerator, or PCIe cards

### Software Requirements
- **Operating System**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Container Runtime**: Docker 20.10+ or Podman 3.0+
- **Orchestrator**: Kubernetes 1.24+ (optional)
- **Python**: 3.9+ (for development/testing)

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone https://github.com/terragonlabs/edge-tpu-v6-benchmark.git
cd edge-tpu-v6-benchmark

# Create required directories
mkdir -p deploy/{data,logs,cache,models,results,config}

# Copy example configuration
cp config/production.yaml deploy/config/

# Start the services
docker-compose -f deploy/docker-compose.yml up -d

# Check status
docker-compose -f deploy/docker-compose.yml ps
```

### Production Configuration

1. **Environment Variables**:
```bash
# Core configuration
EDGE_TPU_LOG_LEVEL=INFO
EDGE_TPU_CONFIG_PATH=/app/config
EDGE_TPU_DATA_PATH=/app/data
WORKERS=4
MAX_CONNECTIONS=1000

# Security
SECURITY_ENABLED=true
RATE_LIMIT_ENABLED=true
AUDIT_LOGGING=true

# Monitoring
MONITORING_ENABLED=true
METRICS_PORT=9090
```

2. **Volume Mounts**:
```yaml
volumes:
  - ./data:/app/data              # Persistent data
  - ./logs:/app/logs              # Application logs
  - ./cache:/app/cache            # Performance cache
  - ./models:/app/models          # Model storage
  - ./results:/app/results        # Benchmark results
  - ./config:/app/config:ro       # Configuration files
  - /dev/bus/usb:/dev/bus/usb     # USB device access
```

3. **Resource Limits**:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### Health Monitoring

```bash
# Check application health
curl http://localhost:8080/health

# View metrics
curl http://localhost:9090/metrics

# Check logs
docker logs edge-tpu-v6-benchmark
```

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Create namespace
kubectl create namespace edge-computing

# Apply RBAC and service account
kubectl apply -f deploy/k8s/deployment.yaml
```

### Configuration Management

1. **ConfigMap for application settings**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: edge-tpu-config
  namespace: edge-computing
data:
  config.yaml: |
    benchmark:
      default_warmup_runs: 10
      default_measurement_runs: 100
      default_timeout: 300
    security:
      enable_authentication: true
      enable_rate_limiting: true
    monitoring:
      enable_metrics: true
      log_level: "INFO"
```

2. **Secrets for sensitive data**:
```bash
# Create TLS secret
kubectl create secret tls edge-tpu-benchmark-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n edge-computing

# Create registry secret (if using private registry)
kubectl create secret docker-registry terragonlabs-registry-secret \
  --docker-server=registry.terragonlabs.ai \
  --docker-username=your-username \
  --docker-password=your-password \
  -n edge-computing
```

### Scaling Configuration

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: edge-tpu-benchmark-hpa
  namespace: edge-computing
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: edge-tpu-v6-benchmark
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Storage Configuration

```yaml
# Fast SSD storage class
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # Adjust for your cloud provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
volumeBindingMode: WaitForFirstConsumer
```

## 🔐 Security Configuration

### SSL/TLS Configuration

1. **Generate certificates**:
```bash
# Self-signed for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Production: Use cert-manager with Let's Encrypt
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

2. **Configure HTTPS**:
```yaml
# Ingress with TLS
spec:
  tls:
  - hosts:
    - edge-tpu-benchmark.yourdomain.com
    secretName: edge-tpu-benchmark-tls
```

### Authentication & Authorization

1. **JWT Configuration**:
```yaml
env:
- name: JWT_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: edge-tpu-secrets
      key: jwt-secret
```

2. **Rate Limiting**:
```yaml
# Nginx Ingress annotations
annotations:
  nginx.ingress.kubernetes.io/rate-limit: "100"
  nginx.ingress.kubernetes.io/rate-limit-rps: "10"
```

### Network Security

1. **NetworkPolicy**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: edge-tpu-network-policy
  namespace: edge-computing
spec:
  podSelector:
    matchLabels:
      app: edge-tpu-benchmark
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
```

## 📊 Monitoring & Observability

### Metrics Collection

1. **Prometheus Configuration**:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
- job_name: 'edge-tpu-benchmark'
  static_configs:
  - targets: ['edge-tpu-benchmark-service:9090']
```

2. **Custom Metrics**:
```python
# Application metrics
from prometheus_client import Counter, Histogram, Gauge

benchmark_requests = Counter('benchmark_requests_total', 'Total benchmark requests')
benchmark_duration = Histogram('benchmark_duration_seconds', 'Benchmark duration')
active_benchmarks = Gauge('active_benchmarks', 'Currently active benchmarks')
```

### Logging Configuration

1. **Structured Logging**:
```yaml
env:
- name: LOG_FORMAT
  value: "json"
- name: LOG_LEVEL
  value: "INFO"
```

2. **Log Aggregation**:
```yaml
# Fluentd/Fluent Bit configuration for log shipping
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/edge-tpu-*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
    </source>
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: edge-tpu-benchmark
  rules:
  - alert: HighErrorRate
    expr: rate(benchmark_errors_total[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate in Edge TPU benchmarks"
      
  - alert: ServiceDown
    expr: up{job="edge-tpu-benchmark"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Edge TPU benchmark service is down"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The repository includes a comprehensive CI/CD pipeline that:

1. **Continuous Integration**:
   - Security auditing (Bandit, Safety, custom scans)
   - Code quality checks (Black, flake8, mypy)
   - Multi-version testing (Python 3.9-3.11)
   - Performance benchmarking

2. **Continuous Deployment**:
   - Docker image building and publishing
   - Kubernetes deployment to staging/production
   - Health checks and rollback capabilities

3. **Security Scanning**:
   - Container vulnerability scanning
   - SBOM (Software Bill of Materials) generation
   - Dependency vulnerability assessment

### Manual Deployment Steps

```bash
# Build and tag image
docker build -f deploy/docker/Dockerfile -t edge-tpu-v6-benchmark:v1.0.0 .

# Push to registry
docker tag edge-tpu-v6-benchmark:v1.0.0 your-registry.com/edge-tpu-v6-benchmark:v1.0.0
docker push your-registry.com/edge-tpu-v6-benchmark:v1.0.0

# Deploy to Kubernetes
kubectl set image deployment/edge-tpu-v6-benchmark \
  edge-tpu-benchmark=your-registry.com/edge-tpu-v6-benchmark:v1.0.0 \
  -n edge-computing

# Check deployment status
kubectl rollout status deployment/edge-tpu-v6-benchmark -n edge-computing
```

## 🧪 Testing & Validation

### Pre-deployment Testing

```bash
# Run comprehensive quality gates
python quality_gates_direct.py

# Run security audit
python security_audit.py

# Performance testing
python -m pytest tests/benchmarks/ --benchmark-only
```

### Post-deployment Validation

```bash
# Health check
curl -f https://your-domain.com/health

# Functionality test
curl -X POST https://your-domain.com/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"model_path": "test_model.tflite", "device": "auto"}'

# Load testing (using Apache Bench)
ab -n 1000 -c 10 https://your-domain.com/health
```

## 🚨 Troubleshooting

### Common Issues

1. **Edge TPU Device Not Detected**:
```bash
# Check USB devices
lsusb | grep Google

# Check permissions
ls -la /dev/bus/usb/

# Verify Edge TPU runtime
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

2. **Performance Issues**:
```bash
# Check resource usage
kubectl top pods -n edge-computing

# Review metrics
curl http://your-service:9090/metrics | grep benchmark

# Check logs for bottlenecks
kubectl logs -f deployment/edge-tpu-v6-benchmark -n edge-computing
```

3. **Memory Issues**:
```bash
# Increase memory limits
kubectl patch deployment edge-tpu-v6-benchmark -n edge-computing -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"edge-tpu-benchmark","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/edge-tpu-v6-benchmark EDGE_TPU_LOG_LEVEL=DEBUG -n edge-computing

# Access debug shell
kubectl exec -it deployment/edge-tpu-v6-benchmark -n edge-computing -- /bin/bash
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks

1. **Daily**:
   - Monitor system health and alerts
   - Check disk usage and clean up old results
   - Review error logs

2. **Weekly**:
   - Update security patches
   - Rotate logs and backups
   - Performance trend analysis

3. **Monthly**:
   - Dependency updates
   - Security audit
   - Capacity planning review

### Getting Help

- **Documentation**: [docs.terragonlabs.ai](https://docs.terragonlabs.ai)
- **Issues**: [GitHub Issues](https://github.com/terragonlabs/edge-tpu-v6-benchmark/issues)
- **Support**: support@terragonlabs.ai
- **Community**: [Discord](https://discord.gg/terragonlabs)

## 📄 License & Compliance

This deployment guide and the Edge TPU v6 Benchmark Suite are licensed under the MIT License. Ensure compliance with:

- Edge TPU hardware licensing terms
- TensorFlow Lite licensing requirements
- Container image licensing (base images)
- Cloud provider terms of service

---

**🎉 Congratulations!** You now have a production-ready Edge TPU v6 Benchmark Suite deployment. For questions or issues, please refer to our support channels above.