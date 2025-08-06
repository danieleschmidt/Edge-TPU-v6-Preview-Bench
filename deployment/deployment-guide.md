# Quantum Task Planner - Production Deployment Guide

## 🎯 Overview

This guide provides step-by-step instructions for deploying the Quantum Task Planner to production environments using Kubernetes and enterprise-grade configurations.

## 📋 Prerequisites

### Infrastructure Requirements
- Kubernetes cluster (v1.25+) with at least 3 nodes
- Container registry access (GitHub Container Registry)
- Persistent storage with 100GB+ capacity
- Load balancer with SSL termination capability
- Redis cluster (v7+) for caching
- PostgreSQL database (v14+) for persistence

### Access Requirements
- Kubernetes cluster admin access
- Container registry push/pull permissions
- DNS management access for domain configuration
- SSL certificate management access

## 🚀 Deployment Steps

### Step 1: Prepare Container Images

```bash
# Build production Docker image
docker build -f deployment/Dockerfile -t quantum-task-planner:latest .

# Tag for registry
docker tag quantum-task-planner:latest ghcr.io/your-org/quantum-task-planner:latest

# Push to registry
docker push ghcr.io/your-org/quantum-task-planner:latest
```

### Step 2: Create Kubernetes Namespace

```bash
# Apply namespace configuration
kubectl apply -f deployment/kubernetes/namespace.yaml

# Verify namespace creation
kubectl get namespaces | grep quantum-planner
```

### Step 3: Configure Secrets

```bash
# Create production secrets
kubectl create secret generic quantum-planner-secrets \
  --from-literal=db-password="your-secure-db-password" \
  --from-literal=jwt-secret="your-jwt-secret-key" \
  --from-literal=encryption-key="your-encryption-key" \
  --from-literal=redis-password="your-redis-password" \
  -n quantum-planner-production

# Verify secrets
kubectl get secrets -n quantum-planner-production
```

### Step 4: Deploy Configuration

```bash
# Apply ConfigMap
kubectl apply -f deployment/kubernetes/configmap.yaml

# Apply Secret manifest (if using file-based secrets)
kubectl apply -f deployment/kubernetes/secret.yaml
```

### Step 5: Deploy Application

```bash
# Deploy main application
kubectl apply -f deployment/kubernetes/deployment.yaml

# Deploy service
kubectl apply -f deployment/kubernetes/service.yaml

# Deploy ingress
kubectl apply -f deployment/kubernetes/ingress.yaml

# Deploy auto-scaling
kubectl apply -f deployment/kubernetes/hpa.yaml

# Apply network policies (optional but recommended)
kubectl apply -f deployment/kubernetes/networkpolicy.yaml
```

### Step 6: Setup Monitoring

```bash
# Deploy ServiceMonitor for Prometheus
kubectl apply -f deployment/kubernetes/servicemonitor.yaml

# Start monitoring stack with Docker Compose (local monitoring)
docker-compose -f deployment/docker-compose.yml up -d prometheus grafana

# Import Grafana dashboard
curl -X POST \
  http://admin:quantum_admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @deployment/monitoring/quantum_dashboard.json
```

### Step 7: Verify Deployment

```bash
# Check deployment status
kubectl rollout status deployment/quantum-task-planner -n quantum-planner-production

# Check pod status
kubectl get pods -n quantum-planner-production

# Check service status
kubectl get services -n quantum-planner-production

# Test health endpoint
kubectl port-forward -n quantum-planner-production service/quantum-task-planner 8080:80
curl http://localhost:8080/health
```

## 🔧 Configuration Management

### Environment-Specific Configurations

#### Production Environment
```yaml
environment: production
region: us-east-1
min_instances: 3
max_instances: 20
cpu_limit: "4000m"
memory_limit: "8Gi"
enable_encryption: true
enable_monitoring: true
```

#### Staging Environment
```yaml
environment: staging
region: us-east-1
min_instances: 1
max_instances: 5
cpu_limit: "2000m"
memory_limit: "4Gi"
enable_encryption: true
enable_monitoring: true
```

### Feature Flags
```yaml
enable_quantum_algorithms: true
enable_advanced_scheduling: true
enable_performance_optimization: true
enable_compliance_features: true
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Install cert-manager (if not already installed)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create cluster issuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Network Security
```bash
# Apply network policies for security
kubectl apply -f deployment/kubernetes/networkpolicy.yaml

# Verify network policies
kubectl get networkpolicies -n quantum-planner-production
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# Add to prometheus.yml
scrape_configs:
  - job_name: 'quantum-task-planner'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - quantum-planner-production
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
```

### Key Metrics to Monitor
- `quantum_tasks_executed_total`: Total tasks executed
- `quantum_efficiency_score`: Current efficiency score
- `quantum_coherence_time`: Quantum coherence duration
- `task_execution_duration`: Task execution time
- `system_resource_usage`: CPU/Memory utilization

### Alert Rules
```yaml
groups:
- name: quantum_planner_alerts
  rules:
  - alert: QuantumEfficiencyLow
    expr: quantum_efficiency_score < 5.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Quantum efficiency below threshold"
      
  - alert: HighTaskFailureRate
    expr: rate(quantum_tasks_failed_total[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High task failure rate detected"
```

## 🌍 Global Deployment

### Multi-Region Setup
```bash
# Deploy to multiple regions
regions=("us-east-1" "eu-west-1" "ap-northeast-1")

for region in "${regions[@]}"; do
  # Update region-specific configurations
  sed "s/us-east-1/$region/g" deployment/kubernetes/deployment.yaml > deployment-$region.yaml
  
  # Apply to region-specific cluster
  kubectl --context=$region-context apply -f deployment-$region.yaml
done
```

### Compliance Configuration
```bash
# Set up GDPR compliance for EU deployment
kubectl patch deployment quantum-task-planner -n quantum-planner-production \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"quantum-planner","env":[{"name":"COMPLIANCE_REGION","value":"GDPR_EU"}]}]}}}}'

# Set up CCPA compliance for US deployment
kubectl patch deployment quantum-task-planner -n quantum-planner-production \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"quantum-planner","env":[{"name":"COMPLIANCE_REGION","value":"CCPA_US"}]}]}}}}'
```

## 🔄 CI/CD Integration

### GitHub Actions Setup
```yaml
# Required repository secrets
KUBECONFIG_PRODUCTION: <base64-encoded-kubeconfig>
DOCKER_REGISTRY_TOKEN: <github-token>
PROD_DB_PASSWORD: <production-db-password>
PROD_JWT_SECRET: <production-jwt-secret>
```

### Automated Deployment Pipeline
1. **Code Push** → Trigger CI/CD pipeline
2. **Security Scan** → Bandit + Safety checks
3. **Testing** → Unit + Integration tests
4. **Build** → Docker image creation
5. **Deploy Staging** → Automated staging deployment
6. **Deploy Production** → Manual approval + production deployment
7. **Validation** → Health checks + smoke tests

## 🩺 Health Checks

### Application Health Endpoints
```bash
# Liveness probe
curl -f http://quantum-planner/health

# Readiness probe  
curl -f http://quantum-planner/ready

# Metrics endpoint
curl http://quantum-planner/metrics
```

### Database Health Check
```bash
# Check database connectivity
kubectl exec -it deployment/quantum-task-planner -n quantum-planner-production -- \
  python -c "
from edge_tpu_v6_bench.quantum_planner.quantum_task_engine import QuantumTaskEngine
engine = QuantumTaskEngine()
print('Database connection: OK' if engine else 'Database connection: FAILED')
"
```

## 🚨 Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod logs
kubectl logs -f deployment/quantum-task-planner -n quantum-planner-production

# Check pod events
kubectl describe pod -l app=quantum-task-planner -n quantum-planner-production
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n quantum-planner-production

# Check HPA status
kubectl describe hpa quantum-task-planner-hpa -n quantum-planner-production
```

#### Connectivity Issues
```bash
# Test service connectivity
kubectl exec -it deployment/quantum-task-planner -n quantum-planner-production -- \
  curl -f http://quantum-task-planner:80/health
```

### Recovery Procedures

#### Rollback Deployment
```bash
# Rollback to previous version
kubectl rollout undo deployment/quantum-task-planner -n quantum-planner-production

# Check rollback status
kubectl rollout status deployment/quantum-task-planner -n quantum-planner-production
```

#### Scale Down/Up
```bash
# Scale down in emergency
kubectl scale deployment quantum-task-planner --replicas=0 -n quantum-planner-production

# Scale back up
kubectl scale deployment quantum-task-planner --replicas=3 -n quantum-planner-production
```

## 📞 Support Contacts

- **DevOps Team**: devops@example.com
- **Security Team**: security@example.com  
- **On-Call**: +1-555-QUANTUM (24/7)
- **Documentation**: https://docs.quantum-planner.com

---

**Version**: 1.0  
**Last Updated**: 2025-08-06  
**Next Review**: Monthly  
**Maintainer**: Quantum Task Planner Team