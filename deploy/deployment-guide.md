# Edge TPU v6 Progressive Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Edge TPU v6 benchmark suite using enterprise-grade deployment practices with progressive quality gates, advanced monitoring, and disaster recovery capabilities.

## Architecture Components

### Core Infrastructure
- **Multi-region Kubernetes deployment** across US-East, US-West, and EU-West
- **Auto-scaling** with HPA, VPA, and custom Edge TPU metrics
- **Blue-green and canary deployments** using Argo Rollouts
- **Service mesh** with Istio for advanced traffic management
- **Comprehensive monitoring** with Prometheus, Grafana, and Alertmanager

### Security & Compliance
- **Pod Security Standards** with restricted profiles
- **Network policies** for micro-segmentation
- **mTLS** and JWT authentication
- **Runtime security monitoring** with Falco
- **Vulnerability scanning** with Trivy

### Disaster Recovery
- **Automated backups** with Velero
- **Cross-region replication**
- **Database backup and restore**
- **Recovery time objectives** < 4 hours

## Prerequisites

### Infrastructure Requirements
```bash
# Kubernetes cluster (1.25+)
kubectl version --client

# Required operators and tools
kubectl apply -f https://github.com/operator-framework/operator-lifecycle-manager/releases/download/v0.25.0/crds.yaml
kubectl apply -f https://github.com/operator-framework/operator-lifecycle-manager/releases/download/v0.25.0/olm.yaml

# Cert-manager for TLS certificates
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Argo Rollouts
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
```

### Cloud Resources
```bash
# AWS S3 bucket for backups
aws s3 mb s3://edge-tpu-backups-terragonlabs --region us-east-1

# ECR registry for container images
aws ecr create-repository --repository-name edge-tpu-v6-benchmark --region us-east-1

# Load balancer and networking
# (Configure according to your cloud provider)
```

## Deployment Steps

### 1. Namespace and RBAC Setup
```bash
# Create namespaces with security policies
kubectl apply -f deploy/k8s/namespace.yaml

# Configure RBAC and service accounts
kubectl apply -f deploy/k8s/rbac.yaml
```

### 2. Security Hardening
```bash
# Apply Pod Security Standards
kubectl apply -f deploy/security/pod-security-standards.yaml

# Configure network policies
kubectl apply -f deploy/security/network-policies.yaml
```

### 3. Monitoring Stack Deployment
```bash
# Deploy Prometheus
kubectl apply -f deploy/monitoring/prometheus.yaml

# Deploy Grafana with dashboards
kubectl apply -f deploy/monitoring/grafana.yaml

# Deploy Alertmanager
kubectl apply -f deploy/monitoring/alertmanager.yaml

# Verify monitoring stack
kubectl get pods -n edge-tpu-monitoring
```

### 4. Service Mesh Setup
```bash
# Install Istio
istioctl install --set values.defaultRevision=default

# Apply Istio configurations
kubectl apply -f deploy/service-mesh/istio-setup.yaml

# Label namespaces for injection
kubectl label namespace edge-tpu-production istio-injection=enabled
kubectl label namespace edge-tpu-staging istio-injection=enabled
```

### 5. Auto-scaling Configuration
```bash
# Deploy HPA and VPA configurations
kubectl apply -f deploy/k8s/autoscaling.yaml

# Verify autoscaling is working
kubectl get hpa,vpa -n edge-tpu-production
```

### 6. Application Deployment with Blue-Green Strategy
```bash
# Deploy initial version (blue)
kubectl apply -f deploy/k8s/deployment.yaml

# Configure Argo Rollouts for blue-green deployment
kubectl apply -f deploy/k8s/argo-rollouts.yaml

# Verify deployment
kubectl get rollout -n edge-tpu-production
kubectl argo rollouts get rollout edge-tpu-v6-benchmark-rollout -n edge-tpu-production
```

### 7. Disaster Recovery Setup
```bash
# Deploy Velero for backup and restore
kubectl apply -f deploy/disaster-recovery/backup-restore.yaml

# Configure backup schedules
kubectl get schedule -n velero

# Test backup creation
velero backup create test-backup --include-namespaces edge-tpu-production
```

### 8. Operational Runbooks
```bash
# Deploy runbooks
kubectl apply -f deploy/runbooks/operational-runbooks.yaml

# Verify runbook availability
kubectl get configmap edge-tpu-runbooks -n edge-tpu-production -o yaml
```

## Progressive Quality Gates

### Stage 1: Development Deployment
```bash
# Deploy to staging with canary strategy
kubectl apply -f deploy/k8s/argo-rollouts.yaml

# Run integration tests
kubectl create job integration-tests --from=cronjob/integration-tests -n edge-tpu-staging
```

### Stage 2: Performance Validation
```bash
# Run performance benchmarks
kubectl create job performance-validation --from=cronjob/performance-tests -n edge-tpu-staging

# Check performance metrics meet SLA
curl -G "http://prometheus.edge-tpu-monitoring.svc.cluster.local:9090/api/v1/query" \
  --data-urlencode 'query=histogram_quantile(0.95, rate(edge_tpu_inference_duration_seconds_bucket[5m]))*1000'
```

### Stage 3: Security Validation
```bash
# Run security scans
kubectl create job security-validation --from=cronjob/security-scan -n edge-tpu-staging

# Check for vulnerabilities
kubectl logs job/security-validation -n edge-tpu-staging
```

### Stage 4: Production Promotion
```bash
# Promote to production with blue-green deployment
kubectl argo rollouts promote edge-tpu-v6-benchmark-rollout -n edge-tpu-production

# Monitor promotion
kubectl argo rollouts get rollout edge-tpu-v6-benchmark-rollout -n edge-tpu-production --watch
```

## Multi-Region Deployment

### Primary Region (us-east-1)
```bash
# Configure primary cluster
kubectl config use-context edge-tpu-primary

# Deploy full stack
kubectl apply -k deploy/overlays/production-primary/
```

### Secondary Regions
```bash
# Configure secondary cluster (us-west-2)
kubectl config use-context edge-tpu-secondary-west

# Deploy with traffic weight 0 initially
kubectl apply -k deploy/overlays/production-secondary-west/
```

### Cross-Region Traffic Management
```bash
# Configure Istio for multi-cluster
istioctl x create-remote-secret --context=edge-tpu-primary --name=primary | \
  kubectl apply -f - --context=edge-tpu-secondary-west

# Set up cross-region service discovery
kubectl apply -f deploy/service-mesh/multi-region-setup.yaml
```

## Monitoring and Observability

### Key Metrics to Monitor
- **Latency**: P95 < 5ms, P99 < 10ms
- **Throughput**: > 200 RPS sustained
- **Error Rate**: < 1% for warnings, < 5% for critical
- **Accuracy**: > 95% model accuracy
- **Resource Utilization**: CPU < 80%, Memory < 85%

### Dashboards
- **Performance Dashboard**: https://grafana.edge-tpu.terragonlabs.ai/d/edge-tpu-performance
- **System Dashboard**: https://grafana.edge-tpu.terragonlabs.ai/d/edge-tpu-system
- **Security Dashboard**: https://grafana.edge-tpu.terragonlabs.ai/d/edge-tpu-security

### Alert Channels
- **Critical Alerts**: PagerDuty + #alerts-critical Slack
- **Warning Alerts**: #alerts-warnings Slack
- **Edge TPU Specific**: #edge-tpu-alerts Slack

## Operational Procedures

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment edge-tpu-v6-benchmark-production --replicas=10 -n edge-tpu-production

# Check autoscaler status
kubectl describe hpa edge-tpu-benchmark-hpa -n edge-tpu-production
```

### Blue-Green Deployment Process
```bash
# Trigger new deployment
kubectl argo rollouts set image edge-tpu-v6-benchmark-rollout \
  edge-tpu-benchmark=terragonlabs/edge-tpu-v6-benchmark:v1.1.0 \
  -n edge-tpu-production

# Monitor analysis
kubectl argo rollouts get rollout edge-tpu-v6-benchmark-rollout -n edge-tpu-production

# Manual promotion if analysis passes
kubectl argo rollouts promote edge-tpu-v6-benchmark-rollout -n edge-tpu-production
```

### Emergency Procedures
```bash
# Emergency rollback
kubectl argo rollouts abort edge-tpu-v6-benchmark-rollout -n edge-tpu-production
kubectl argo rollouts undo edge-tpu-v6-benchmark-rollout -n edge-tpu-production

# Traffic rerouting to secondary region
kubectl patch virtualservice edge-tpu-production-vs -n edge-tpu-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/http/0/route/0/destination/host", "value": "edge-tpu-benchmark-active.edge-tpu-production-west.svc.cluster.local"}]'
```

## Disaster Recovery Procedures

### Backup Verification
```bash
# Check recent backups
velero backup get

# Verify backup completeness
velero backup describe <backup-name> --details
```

### Disaster Recovery Scenarios

#### Full Cluster Recovery
```bash
# 1. Provision new cluster
# 2. Install Velero with same configuration
# 3. Restore from backup
velero restore create cluster-restore-$(date +%Y%m%d) \
  --from-backup <latest-backup> \
  --restore-volumes=true

# 4. Update DNS to point to new cluster
# 5. Verify all services are operational
```

#### Namespace Recovery
```bash
# Restore specific namespace
velero restore create namespace-restore-$(date +%Y%m%d) \
  --from-backup <backup-name> \
  --include-namespaces edge-tpu-production
```

## Security Considerations

### Runtime Security
```bash
# Check Falco alerts
kubectl logs -n edge-tpu-monitoring -l app=falco | grep -i "priority.*critical"

# Review security scan results
kubectl get job security-scan -n edge-tpu-production -o yaml
```

### Compliance Validation
```bash
# Verify Pod Security Standards
kubectl get pods -n edge-tpu-production -o jsonpath='{.items[*].spec.securityContext}'

# Check network policy compliance
kubectl get networkpolicy -n edge-tpu-production
```

## Performance Tuning

### Resource Optimization
```bash
# Check VPA recommendations
kubectl get vpa edge-tpu-benchmark-vpa -n edge-tpu-production -o yaml

# Apply recommended resources
kubectl patch deployment edge-tpu-v6-benchmark-production -n edge-tpu-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources", "value": <vpa-recommendations>}]'
```

### Istio Performance Tuning
```bash
# Optimize proxy resources
kubectl patch deployment edge-tpu-v6-benchmark-production -n edge-tpu-production --type='json' \
  -p='[{"op": "add", "path": "/spec/template/metadata/annotations/sidecar.istio.io~1proxyCPU", "value": "100m"}]'
```

## Troubleshooting

### Common Issues and Solutions

#### High Latency
1. Check HPA scaling: `kubectl get hpa -n edge-tpu-production`
2. Verify resource limits: `kubectl describe deployment -n edge-tpu-production`
3. Check service mesh configuration: `istioctl analyze -n edge-tpu-production`

#### Failed Deployments
1. Check rollout status: `kubectl argo rollouts get rollout edge-tpu-v6-benchmark-rollout -n edge-tpu-production`
2. Review analysis results: `kubectl get analysisrun -n edge-tpu-production`
3. Check pod logs: `kubectl logs -n edge-tpu-production -l app=edge-tpu-benchmark`

#### Monitoring Issues
1. Verify service monitors: `kubectl get servicemonitor -n edge-tpu-monitoring`
2. Check Prometheus targets: Access Prometheus UI and verify targets
3. Validate alert rules: `kubectl get prometheusrule -n edge-tpu-monitoring`

## Maintenance Windows

### Scheduled Maintenance
```bash
# Scale down gracefully
kubectl patch hpa edge-tpu-benchmark-hpa -n edge-tpu-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/minReplicas", "value": 1}]'

# Enable maintenance mode
kubectl patch service edge-tpu-benchmark-active -n edge-tpu-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/selector", "value": {"app": "maintenance"}}]'
```

### Post-Maintenance Verification
```bash
# Restore normal operations
kubectl patch hpa edge-tpu-benchmark-hpa -n edge-tpu-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/minReplicas", "value": 3}]'

# Run health checks
kubectl create job post-maintenance-checks --from=cronjob/health-checks -n edge-tpu-production
```

## Cost Optimization

### Resource Right-sizing
```bash
# Analyze resource utilization over time
curl -G "http://prometheus.edge-tpu-monitoring.svc.cluster.local:9090/api/v1/query_range" \
  --data-urlencode 'query=avg(rate(container_cpu_usage_seconds_total{container="edge-tpu-benchmark"}[5m]))' \
  --data-urlencode 'start=now-7d' \
  --data-urlencode 'end=now' \
  --data-urlencode 'step=1h'
```

### Auto-scaling Optimization
```bash
# Adjust HPA parameters based on usage patterns
kubectl patch hpa edge-tpu-benchmark-hpa -n edge-tpu-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/behavior/scaleDown/stabilizationWindowSeconds", "value": 600}]'
```

## Support and Documentation

### Team Contacts
- **SRE Team**: sre@terragonlabs.ai
- **ML Engineering**: ml-team@terragonlabs.ai  
- **Security Team**: security@terragonlabs.ai
- **DevOps**: devops@terragonlabs.ai

### Additional Resources
- **Runbooks**: Available in cluster ConfigMaps
- **Dashboards**: https://grafana.edge-tpu.terragonlabs.ai
- **Documentation**: https://docs.terragonlabs.ai/edge-tpu-v6
- **Incident Response**: https://docs.terragonlabs.ai/incident-response

### Emergency Contacts
- **Critical Issues**: PagerDuty escalation
- **Security Incidents**: security-oncall@terragonlabs.ai
- **Service Degradation**: sre-oncall@terragonlabs.ai

---

This deployment guide provides a comprehensive framework for enterprise-grade deployment of the Edge TPU v6 benchmark suite with production-ready monitoring, security, and disaster recovery capabilities.