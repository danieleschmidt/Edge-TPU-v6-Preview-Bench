# Contributing to Edge TPU v6 Benchmark Suite

Thank you for your interest in contributing to the Edge TPU v6 Benchmark Suite! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Edge-TPU-v6-Preview-Bench.git
   cd Edge-TPU-v6-Preview-Bench
   ```

3. **Set up development environment**:
   ```bash
   # Install Poetry (if not already installed)
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install --with dev

   # Install pre-commit hooks
   poetry run pre-commit install
   ```

4. **Create a new branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+ 
- Poetry 1.6+
- Git
- Docker (optional, for container testing)

### Development Dependencies

```bash
# Install all dependencies including dev tools
poetry install --with dev --extras all

# Activate virtual environment
poetry shell
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=edge_tpu_v6_bench --cov-report=html

# Run specific test file
poetry run pytest tests/test_device_manager.py

# Run tests with specific markers
poetry run pytest -m "not slow"
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Lint code
poetry run flake8 .

# Type checking
poetry run mypy src/edge_tpu_v6_bench

# Security scanning
poetry run bandit -r src/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## 📋 Contributing Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 100)
- Use type hints for all functions and methods
- Write docstrings for all public functions, classes, and modules
- Follow [Google docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(quantization): add INT4 mixed precision support
fix(device): resolve USB device detection on Windows
docs(readme): update installation instructions
test(benchmark): add concurrent execution tests
```

### Pull Request Process

1. **Update documentation** if you're changing APIs or adding features
2. **Add tests** for new functionality
3. **Ensure all tests pass** and coverage remains above 85%
4. **Update CHANGELOG.md** with your changes
5. **Submit pull request** with a clear title and description

### Testing Requirements

- All new code must have corresponding tests
- Tests should cover both happy path and edge cases
- Maintain minimum 85% code coverage
- Use meaningful test names that describe what is being tested
- Mock external dependencies appropriately

### Documentation

- Update docstrings for any changed functions/classes
- Add examples to docstrings where helpful
- Update README.md if adding new features
- Consider adding tutorials or guides for complex features

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- **Model Support**: Add support for new model architectures
- **Quantization Strategies**: Implement additional quantization methods
- **Device Support**: Extend compatibility to new Edge TPU variants
- **Performance Optimization**: Improve benchmark execution speed
- **Documentation**: Tutorials, examples, and API documentation

### Medium Priority
- **Analysis Tools**: Enhanced result visualization and analysis
- **CI/CD Improvements**: Better automation and testing
- **Error Handling**: More robust error recovery
- **Configuration**: More flexible configuration options
- **Internationalization**: Additional language translations

### Technical Details

#### Adding New Quantization Strategies

1. Create new strategy class in `src/edge_tpu_v6_bench/quantization/strategies/`
2. Implement required methods: `quantize()`, `validate()`
3. Add strategy to `AutoQuantizer.STRATEGY_PRIORITY`
4. Write comprehensive tests
5. Update documentation

#### Adding New Benchmark Suites

1. Create new benchmark class in `src/edge_tpu_v6_bench/benchmarks/`
2. Inherit from base benchmark interfaces
3. Implement concurrent execution support
4. Add CLI integration
5. Include result visualization
6. Write integration tests

#### Adding Device Support

1. Extend `DeviceManager` detection logic
2. Add device-specific optimizations
3. Update capability mapping
4. Test on actual hardware (if available)
5. Document device-specific features

## 🐛 Reporting Issues

### Bug Reports

Please include:
- **Environment details**: OS, Python version, hardware
- **Steps to reproduce**: Minimal example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full stack traces
- **Hardware info**: Edge TPU version, driver versions

### Feature Requests

Please include:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches
- **Implementation ideas**: Technical approach (if any)

## 🏗️ Architecture Overview

### Core Components

```
src/edge_tpu_v6_bench/
├── core/                 # Core benchmarking engine
│   ├── device_manager.py # Device detection and management
│   ├── benchmark.py      # Main benchmark orchestration
│   ├── metrics.py        # Performance metrics collection
│   └── power.py          # Power monitoring
├── quantization/         # Quantization strategies
│   ├── auto_quantizer.py # Automatic quantization
│   └── strategies/       # Individual quantization methods
├── benchmarks/           # Benchmark suites
│   ├── micro.py          # Micro-benchmarks
│   ├── standard.py       # Standard model benchmarks
│   └── applications.py   # Application benchmarks
├── analysis/             # Result analysis and visualization
├── compatibility/        # Device compatibility layers
└── cli.py               # Command-line interface
```

### Design Principles

- **Modularity**: Each component has clear responsibilities
- **Extensibility**: Easy to add new devices, models, and strategies
- **Performance**: Concurrent execution and optimization
- **Reliability**: Comprehensive error handling and validation
- **Usability**: Simple APIs and clear documentation
- **Global Compatibility**: I18n support and cross-platform compatibility

## 🌍 Internationalization

We support multiple languages. To add a new language:

1. Add translations to `TRANSLATIONS` dict in `cli.py`
2. Test with `--lang` flag
3. Update documentation

Currently supported languages:
- English (en) - Default
- Spanish (es)
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)

## 📊 Performance Guidelines

- Use concurrent execution where appropriate
- Profile performance-critical code
- Optimize for memory usage
- Consider hardware-specific optimizations
- Measure and document performance impact

## 🔒 Security Considerations

- Never commit secrets or API keys
- Validate all user inputs
- Use secure communication protocols
- Follow security best practices
- Run security scans regularly

## 💬 Community

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs and request features
- **Pull Requests**: Contribute code improvements
- **Documentation**: Help improve docs and examples

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Documentation credits

Thank you for contributing to make Edge TPU benchmarking better for everyone! 🚀