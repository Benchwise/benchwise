---
sidebar_position: 4
---

# Contributing

Contribute to Benchwise development.

## Getting Started

Instructions for setting up your development environment to contribute to Benchwise.

1. Fork the repository
2. Clone your fork
3. Install development dependencies

```bash
git clone https://github.com/YOUR_USERNAME/benchwise.git
cd benchwise
pip install -e ".[dev]"
```

## Development Workflow

Key steps and tools for local development.

### Running Tests

```bash
# All tests
python run_tests.py

# Basic tests only
python run_tests.py --basic

# Specific test file
pytest tests/test_core.py -v
```

### Code Quality

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Pre-commit Hooks

```bash
pre-commit run --all-files
```

## Contributing Guidelines

Step-by-step process for making contributions.

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make your changes**
   - Write clear, concise code
   - Add tests for new features
   - Update documentation

3. **Test your changes**
   ```bash
   python run_tests.py
   ```

4. **Commit your changes**
   ```bash
   git commit -m "feat: add new feature"
   ```

   Use conventional commits:
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation
   - `test:` - Tests
   - `refactor:` - Code refactoring

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature
   ```

## Code Style

Guidelines for maintaining consistent code style.

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Keep functions focused and concise

## Reporting Issues

How to report bugs or suggest features.

- Use [GitHub Issues](https://github.com/Benchwise/benchwise/issues)
- Provide clear description
- Include code examples
- Specify environment details

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
