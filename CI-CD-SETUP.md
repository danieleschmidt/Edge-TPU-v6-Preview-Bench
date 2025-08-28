# CI/CD Setup Instructions

## GitHub Actions Workflow Setup

Due to GitHub App permission restrictions, the CI/CD workflow must be manually copied to enable full automation.

### Quick Setup

1. **Copy the workflow template**:
   ```bash
   cp ci-cd-templates/github-actions-template.yml .github/workflows/ci-cd.yml
   ```

2. **Commit the workflow**:
   ```bash
   git add .github/workflows/ci-cd.yml
   git commit -m "feat(ci): add comprehensive CI/CD pipeline"
   git push
   ```

### Workflow Features

The CI/CD pipeline includes:

- ✅ **Security Audit**: Automated security scanning
- ✅ **Code Quality**: Quality gate validation  
- ✅ **Multi-Platform Testing**: Python 3.8-3.11 on Ubuntu
- ✅ **Docker Build & Push**: Container registry integration
- ✅ **Staging Deployment**: Automated staging environment
- ✅ **Production Deployment**: Production-ready deployment
- ✅ **Performance Monitoring**: Post-deployment monitoring

### Required GitHub Secrets

Configure these secrets in your GitHub repository:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Deployment Environments

The workflow uses GitHub Environments:
- `staging` - Staging deployment environment
- `production` - Production deployment environment

### Quality Gates

The pipeline enforces quality gates:
- Security Score > 70/100
- Quality Score > 60/100

Only deployments meeting these criteria will proceed to production.

## Alternative CI/CD Platforms

### GitLab CI/CD
```yaml
# Copy ci-cd-templates/gitlab-ci-template.yml to .gitlab-ci.yml
```

### Azure DevOps
```yaml
# Copy ci-cd-templates/azure-pipelines-template.yml to azure-pipelines.yml
```

### Jenkins
```groovy
// Copy ci-cd-templates/Jenkinsfile to root directory
```

## Manual Deployment

If automated CI/CD is not available, use manual deployment:

```bash
# Run quality gates
python3 quality_gates_comprehensive.py

# Build Docker image
docker build -t edge-tpu-v6-bench .

# Deploy to Kubernetes
kubectl apply -f deploy/k8s/

# Validate deployment
python3 validate_deployment.py
```

## Next Steps

1. Copy the workflow template to `.github/workflows/`
2. Configure GitHub secrets and environments
3. Push changes to trigger the first pipeline run
4. Monitor deployment in GitHub Actions tab

The CI/CD pipeline will automatically handle:
- Quality validation
- Security scanning  
- Multi-environment deployment
- Performance monitoring
- Rollback capabilities

🚀 **Ready for Production Deployment with Full CI/CD Automation**