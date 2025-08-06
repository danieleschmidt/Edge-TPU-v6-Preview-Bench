"""
Production Deployment Module for Quantum Task Planner
Enterprise-grade deployment configurations and orchestration
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"

class ScalingStrategy(Enum):
    FIXED = "fixed"
    AUTO_SCALE = "auto_scale"
    SCHEDULED = "scheduled"
    PREDICTIVE = "predictive"

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    environment: DeploymentEnvironment
    region: str = "us-east-1"
    availability_zones: List[str] = field(default_factory=lambda: ["us-east-1a", "us-east-1b"])
    
    # Scaling configuration
    min_instances: int = 2
    max_instances: int = 10
    scaling_strategy: ScalingStrategy = ScalingStrategy.AUTO_SCALE
    cpu_target_percent: int = 70
    memory_target_percent: int = 80
    
    # Resource limits
    cpu_limit: str = "2000m"  # 2 CPU cores
    memory_limit: str = "4Gi"  # 4GB RAM
    cpu_request: str = "500m"  # 0.5 CPU cores
    memory_request: str = "1Gi"  # 1GB RAM
    
    # Storage configuration
    storage_class: str = "gp3"
    storage_size: str = "20Gi"
    backup_retention_days: int = 30
    
    # Networking
    load_balancer_type: str = "application"
    ssl_certificate_arn: Optional[str] = None
    enable_cdn: bool = True
    
    # Security
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True
    enable_network_policies: bool = True
    enable_pod_security_policies: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    log_level: str = "INFO"
    metrics_retention_days: int = 90
    
    # Health checks
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    
    # Feature flags
    enable_quantum_algorithms: bool = True
    enable_advanced_scheduling: bool = True
    enable_performance_optimization: bool = True
    enable_compliance_features: bool = True

class DeploymentOrchestrator:
    """Production deployment orchestration and management"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_id = f"quantum-planner-{int(time.time())}"
        
        logger.info(f"DeploymentOrchestrator initialized for {config.environment.value}")
    
    def generate_kubernetes_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests"""
        
        manifests = {}
        
        # Namespace
        manifests["namespace.yaml"] = self._generate_namespace()
        
        # Deployment
        manifests["deployment.yaml"] = self._generate_deployment()
        
        # Service  
        manifests["service.yaml"] = self._generate_service()
        
        # ConfigMap
        manifests["configmap.yaml"] = self._generate_configmap()
        
        # Secret
        manifests["secret.yaml"] = self._generate_secret()
        
        # HorizontalPodAutoscaler
        if self.config.scaling_strategy == ScalingStrategy.AUTO_SCALE:
            manifests["hpa.yaml"] = self._generate_hpa()
        
        # Ingress
        manifests["ingress.yaml"] = self._generate_ingress()
        
        # NetworkPolicy
        if self.config.enable_network_policies:
            manifests["networkpolicy.yaml"] = self._generate_network_policy()
        
        # ServiceMonitor (for Prometheus)
        if self.config.enable_metrics:
            manifests["servicemonitor.yaml"] = self._generate_service_monitor()
        
        return manifests
    
    def _generate_namespace(self) -> str:
        """Generate Kubernetes namespace"""
        namespace_name = f"quantum-planner-{self.config.environment.value}"
        
        return f"""apiVersion: v1
kind: Namespace
metadata:
  name: {namespace_name}
  labels:
    app: quantum-planner
    environment: {self.config.environment.value}
    managed-by: quantum-deployment-orchestrator
"""
    
    def _generate_deployment(self) -> str:
        """Generate Kubernetes deployment manifest"""
        app_name = "quantum-task-planner"
        namespace = f"quantum-planner-{self.config.environment.value}"
        
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    version: v1
    environment: {self.config.environment.value}
spec:
  replicas: {self.config.min_instances}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
        version: v1
        environment: {self.config.environment.value}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        fsGroup: 65534
      containers:
      - name: quantum-planner
        image: quantum-task-planner:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        - containerPort: 8081
          name: metrics
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: {self.config.environment.value}
        - name: LOG_LEVEL  
          value: {self.config.log_level}
        - name: QUANTUM_ALGORITHMS_ENABLED
          value: "{str(self.config.enable_quantum_algorithms).lower()}"
        - name: PERFORMANCE_OPTIMIZATION_ENABLED
          value: "{str(self.config.enable_performance_optimization).lower()}"
        - name: COMPLIANCE_FEATURES_ENABLED
          value: "{str(self.config.enable_compliance_features).lower()}"
        envFrom:
        - configMapRef:
            name: quantum-planner-config
        - secretRef:
            name: quantum-planner-secrets
        resources:
          limits:
            cpu: {self.config.cpu_limit}
            memory: {self.config.memory_limit}
          requests:
            cpu: {self.config.cpu_request}
            memory: {self.config.memory_request}
        livenessProbe:
          httpGet:
            path: {self.config.health_check_path}
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: {self.config.health_check_interval_seconds}
          timeoutSeconds: {self.config.health_check_timeout_seconds}
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: {self.config.readiness_check_path}
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: {self.config.health_check_timeout_seconds}
          failureThreshold: 3
        volumeMounts:
        - name: quantum-data
          mountPath: /data
        - name: logs
          mountPath: /logs
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
      volumes:
      - name: quantum-data
        persistentVolumeClaim:
          claimName: quantum-planner-pvc
      - name: logs
        emptyDir: {{}}
      restartPolicy: Always
      terminationGracePeriodSeconds: 30
"""
    
    def _generate_service(self) -> str:
        """Generate Kubernetes service manifest"""
        app_name = "quantum-task-planner"
        namespace = f"quantum-planner-{self.config.environment.value}"
        
        return f"""apiVersion: v1
kind: Service
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {self.config.environment.value}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "{self.config.load_balancer_type}"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "{self.config.ssl_certificate_arn or ''}"
spec:
  type: LoadBalancer
  selector:
    app: {app_name}
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: https
    port: 443
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 8081
    targetPort: 8081
    protocol: TCP
"""
    
    def _generate_configmap(self) -> str:
        """Generate ConfigMap for application configuration"""
        namespace = f"quantum-planner-{self.config.environment.value}"
        
        config_data = {
            "QUANTUM_ENGINE_MAX_WORKERS": "8",
            "QUANTUM_COHERENCE_TIME": "30.0", 
            "PERFORMANCE_SCALING_MODE": "auto_scale",
            "PERFORMANCE_MAX_WORKERS": str(self.config.max_instances * 2),
            "PERFORMANCE_MEMORY_LIMIT_MB": "4096",
            "PERFORMANCE_CACHE_STRATEGY": "adaptive",
            "SECURITY_ENABLE_ENCRYPTION": str(self.config.enable_encryption_at_rest).lower(),
            "COMPLIANCE_DEFAULT_REGION": "global",
            "MONITORING_METRICS_ENABLED": str(self.config.enable_metrics).lower(),
            "MONITORING_TRACING_ENABLED": str(self.config.enable_tracing).lower(),
        }
        
        config_yaml = "\\n".join([f"  {k}: \"{v}\"" for k, v in config_data.items()])
        
        return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-planner-config
  namespace: {namespace}
  labels:
    app: quantum-task-planner
    environment: {self.config.environment.value}
data:
{config_yaml}
"""
    
    def _generate_secret(self) -> str:
        """Generate Secret for sensitive configuration"""
        namespace = f"quantum-planner-{self.config.environment.value}"
        
        return f"""apiVersion: v1
kind: Secret
metadata:
  name: quantum-planner-secrets
  namespace: {namespace}
  labels:
    app: quantum-task-planner
    environment: {self.config.environment.value}
type: Opaque
data:
  # Base64 encoded secrets - replace with actual values
  redis-password: cXVhbnR1bS1yZWRpcy1wYXNzd29yZA==
  jwt-secret: cXVhbnR1bS1qd3Qtc2VjcmV0LWtleQ==
  encryption-key: cXVhbnR1bS1lbmNyeXB0aW9uLWtleQ==
"""
    
    def _generate_hpa(self) -> str:
        """Generate HorizontalPodAutoscaler for auto-scaling"""
        app_name = "quantum-task-planner"
        namespace = f"quantum-planner-{self.config.environment.value}"
        
        return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {app_name}-hpa
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {self.config.environment.value}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {app_name}
  minReplicas: {self.config.min_instances}
  maxReplicas: {self.config.max_instances}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {self.config.cpu_target_percent}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {self.config.memory_target_percent}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
"""
    
    def _generate_ingress(self) -> str:
        """Generate Ingress for external access"""
        app_name = "quantum-task-planner"
        namespace = f"quantum-planner-{self.config.environment.value}"
        host = f"quantum-planner-{self.config.environment.value}.example.com"
        
        return f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {app_name}-ingress
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {self.config.environment.value}
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - {host}
    secretName: {app_name}-tls
  rules:
  - host: {host}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {app_name}
            port:
              number: 80
"""
    
    def _generate_network_policy(self) -> str:
        """Generate NetworkPolicy for security"""
        app_name = "quantum-task-planner"
        namespace = f"quantum-planner-{self.config.environment.value}"
        
        return f"""apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {app_name}-netpol
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {self.config.environment.value}
spec:
  podSelector:
    matchLabels:
      app: {app_name}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 8081
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
  - to:
    - namespaceSelector:
        matchLabels:
          name: redis
    ports:
    - protocol: TCP
      port: 6379
"""
    
    def _generate_service_monitor(self) -> str:
        """Generate ServiceMonitor for Prometheus"""
        app_name = "quantum-task-planner"
        namespace = f"quantum-planner-{self.config.environment.value}"
        
        return f"""apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {app_name}-metrics
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {self.config.environment.value}
    monitoring: prometheus
spec:
  selector:
    matchLabels:
      app: {app_name}
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    scheme: http
  namespaceSelector:
    matchNames:
    - {namespace}
"""
    
    def generate_docker_files(self) -> Dict[str, str]:
        """Generate Docker-related files"""
        
        files = {}
        
        # Dockerfile
        files["Dockerfile"] = self._generate_dockerfile()
        
        # Docker Compose (for local development)
        files["docker-compose.yml"] = self._generate_docker_compose()
        
        # .dockerignore
        files[".dockerignore"] = self._generate_dockerignore()
        
        return files
    
    def _generate_dockerfile(self) -> str:
        """Generate production Dockerfile"""
        return """# Multi-stage build for production deployment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY pyproject.toml setup.py ./
COPY src/edge_tpu_v6_bench/__init__.py ./src/edge_tpu_v6_bench/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY README.md LICENSE ./

# Create required directories
RUN mkdir -p /data /logs && \\
    chown -R quantum:quantum /app /data /logs

# Security: Remove unnecessary packages and files
RUN apt-get autoremove -y && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Switch to non-root user
USER quantum

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Start command
CMD ["python", "-m", "edge_tpu_v6_bench.quantum_planner.quantum_cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
"""
    
    def _generate_docker_compose(self) -> str:
        """Generate Docker Compose for local development"""
        return """version: '3.8'

services:
  quantum-planner:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - QUANTUM_ALGORITHMS_ENABLED=true
      - PERFORMANCE_OPTIMIZATION_ENABLED=true
      - COMPLIANCE_FEATURES_ENABLED=true
      - REDIS_URL=redis://redis:6379
    volumes:
      - quantum_data:/data
      - quantum_logs:/logs
    depends_on:
      - redis
      - prometheus
    networks:
      - quantum_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - quantum_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=90d'
      - '--web.enable-lifecycle'
    networks:
      - quantum_network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=quantum_admin
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - quantum_network
    restart: unless-stopped

volumes:
  quantum_data:
  quantum_logs:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  quantum_network:
    driver: bridge
"""
    
    def _generate_dockerignore(self) -> str:
        """Generate .dockerignore file"""
        return """.git
.gitignore
README.md
Dockerfile
.dockerignore
docker-compose.yml
.pytest_cache
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.idea
.vscode
.DS_Store
tests/
docs/
examples/
.github/
*.md
!README.md
"""
    
    def generate_monitoring_configs(self) -> Dict[str, str]:
        """Generate monitoring configuration files"""
        
        configs = {}
        
        # Prometheus configuration
        configs["prometheus.yml"] = self._generate_prometheus_config()
        
        # Grafana dashboard
        configs["quantum_dashboard.json"] = self._generate_grafana_dashboard()
        
        # Alerting rules
        configs["alert_rules.yml"] = self._generate_alert_rules()
        
        return configs
    
    def _generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration"""
        return """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

scrape_configs:
  - job_name: 'quantum-task-planner'
    static_configs:
      - targets: ['quantum-planner:8081']
    scrape_interval: 30s
    metrics_path: /metrics
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
      
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"""
    
    def _generate_grafana_dashboard(self) -> str:
        """Generate Grafana dashboard configuration"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Quantum Task Planner - Production Monitoring",
                "tags": ["quantum", "task-planner", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Task Execution Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(quantum_tasks_executed_total[5m])",
                                "legendFormat": "Tasks/sec"
                            }
                        ]
                    },
                    {
                        "id": 2, 
                        "title": "Quantum Efficiency Score",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "quantum_efficiency_score",
                                "legendFormat": "Efficiency"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "System Resource Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(process_cpu_seconds_total[5m])",
                                "legendFormat": "CPU Usage"
                            },
                            {
                                "expr": "process_resident_memory_bytes",
                                "legendFormat": "Memory Usage"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        return json.dumps(dashboard, indent=2)
    
    def _generate_alert_rules(self) -> str:
        """Generate Prometheus alerting rules"""
        return """groups:
- name: quantum_planner_alerts
  rules:
  - alert: QuantumPlannerDown
    expr: up{job="quantum-task-planner"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Quantum Task Planner is down"
      description: "Quantum Task Planner has been down for more than 1 minute"

  - alert: HighTaskFailureRate
    expr: rate(quantum_tasks_failed_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High task failure rate detected"
      description: "Task failure rate is {{ $value }} tasks/sec"

  - alert: LowQuantumEfficiency
    expr: quantum_efficiency_score < 5.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low quantum efficiency detected"
      description: "Quantum efficiency score is {{ $value }}"

  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes > 3.5e9  # 3.5GB
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value | humanizeBytes }}"

  - alert: HighCPUUsage
    expr: rate(process_cpu_seconds_total[5m]) > 1.5
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value | humanizePercent }}"
"""
    
    def save_deployment_files(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Save all deployment files to specified directory"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Kubernetes manifests
        k8s_dir = output_path / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        k8s_manifests = self.generate_kubernetes_manifests()
        for filename, content in k8s_manifests.items():
            file_path = k8s_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            saved_files[f"kubernetes/{filename}"] = str(file_path)
        
        # Docker files
        docker_files = self.generate_docker_files()
        for filename, content in docker_files.items():
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                f.write(content)
            saved_files[filename] = str(file_path)
        
        # Monitoring configurations
        monitoring_dir = output_path / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        monitoring_configs = self.generate_monitoring_configs()
        for filename, content in monitoring_configs.items():
            file_path = monitoring_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            saved_files[f"monitoring/{filename}"] = str(file_path)
        
        # Deployment configuration
        config_data = {
            "deployment_id": self.deployment_id,
            "environment": self.config.environment.value,
            "region": self.config.region,
            "created_at": time.time(),
            "config": {
                "min_instances": self.config.min_instances,
                "max_instances": self.config.max_instances,
                "scaling_strategy": self.config.scaling_strategy.value,
                "resource_limits": {
                    "cpu": self.config.cpu_limit,
                    "memory": self.config.memory_limit
                },
                "features": {
                    "quantum_algorithms": self.config.enable_quantum_algorithms,
                    "performance_optimization": self.config.enable_performance_optimization,
                    "compliance_features": self.config.enable_compliance_features
                }
            }
        }
        
        config_file = output_path / "deployment-config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        saved_files["deployment-config.json"] = str(config_file)
        
        logger.info(f"Deployment files saved to {output_path}")
        logger.info(f"Generated {len(saved_files)} deployment files")
        
        return saved_files

def create_production_deployment(environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION,
                               region: str = "us-east-1") -> DeploymentOrchestrator:
    """Create production deployment configuration"""
    
    config = DeploymentConfig(
        environment=environment,
        region=region,
        min_instances=3,
        max_instances=20,
        scaling_strategy=ScalingStrategy.AUTO_SCALE,
        cpu_limit="4000m",
        memory_limit="8Gi",
        cpu_request="1000m", 
        memory_request="2Gi",
        enable_encryption_at_rest=True,
        enable_encryption_in_transit=True,
        enable_network_policies=True,
        enable_pod_security_policies=True,
        enable_metrics=True,
        enable_logging=True,
        enable_tracing=True,
        enable_quantum_algorithms=True,
        enable_advanced_scheduling=True,
        enable_performance_optimization=True,
        enable_compliance_features=True
    )
    
    return DeploymentOrchestrator(config)