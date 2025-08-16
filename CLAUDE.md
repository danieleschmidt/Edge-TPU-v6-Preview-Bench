# Claude Code Configuration

## Project: Edge TPU v6 Preview Benchmark Suite

This file configures Claude Code for optimal assistance with the Edge TPU v6 benchmark project.

### Project Overview
- **Type**: Advanced ML/AI Benchmark Suite
- **Domain**: Edge AI Hardware Acceleration (Google Edge TPU v6) 
- **Language**: Python 3.8+
- **Framework**: TensorFlow, PyTorch support
- **Architecture**: Multi-generational implementation (Simple → Robust → Scalable)

### Development Environment
- **Python**: 3.8+ required
- **Package Manager**: Poetry
- **Testing**: pytest with coverage
- **Linting**: black, flake8, mypy
- **Security**: bandit, safety

### Key Commands
```bash
# Install dependencies
poetry install

# Run tests
pytest tests/ -v

# Run quality gates
python quality_gates_comprehensive.py

# Build and run containers
docker-compose up edge-tpu-bench

# Run benchmarks
edge-tpu-v6-bench benchmark --model models/test.tflite
```

### Architecture Generations
1. **Generation 1 (Simple)**: Basic functionality - `core/simple_benchmark.py`
2. **Generation 2 (Robust)**: Error handling + security - `core/robust_benchmark.py`  
3. **Generation 3 (Scalable)**: Concurrency + optimization - `core/scalable_benchmark.py`

### Research Components
- **Novel Algorithms**: `research/novel_optimization.py`
- **Experimental Design**: `research/experimental_design.py`
- **Statistical Testing**: `research/statistical_testing.py`
- **Publication Tools**: `research/publication_tools.py`

### Quality Standards
- **Code Coverage**: >85%
- **Type Checking**: mypy strict mode
- **Security Scan**: bandit + custom security audit
- **Performance**: Sub-5ms inference latency targets

### Deployment Targets
- **Docker**: Multi-stage optimized containers
- **Kubernetes**: Full orchestration with auto-scaling
- **CI/CD**: GitHub Actions with quality gates
- **Monitoring**: Prometheus + Grafana stack

### Claude Code Assistance Areas
1. **Algorithmic Optimization**: Help with Edge TPU v6 performance tuning
2. **Research Implementation**: Novel quantization and optimization algorithms
3. **Testing Strategy**: Comprehensive test coverage and validation
4. **Documentation**: Academic-quality documentation and papers
5. **Deployment**: Production-ready containerization and orchestration

### File Patterns
- Core modules: `src/edge_tpu_v6_bench/core/*.py`
- Research code: `src/edge_tpu_v6_bench/research/*.py`
- Tests: `tests/test_*.py`
- Configs: `*.toml`, `*.yml`, `docker-compose.yml`
- Results: `*_results/*.json`

### Performance Targets
- **Latency**: <5ms average inference time
- **Throughput**: >200 FPS sustained
- **Accuracy**: <1% drop with quantization
- **Scalability**: Linear scaling to 8 workers
- **Memory**: <2GB peak usage

This configuration enables Claude Code to provide optimal assistance for:
- Algorithm development and optimization
- Research experiment design and implementation  
- Production deployment and monitoring
- Academic publication preparation