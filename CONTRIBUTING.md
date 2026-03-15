# Contributing to Purrxy

Thanks for your interest in contributing to purrxy!

## Getting Started

1. Fork and clone the repository
2. Copy `.env.example` to `.env` and adjust if needed
3. Run tests to make sure everything passes:

```bash
python3 -m unittest discover tests/ -v
```

## Development

purrxy has **zero external dependencies** — it uses only the Python 3.9+ standard library. Please keep it that way. If you think a third-party dependency is truly necessary, open an issue to discuss it first.

### Project Structure

```
purrxy.py              Entry point
src/
  process/                 Generic process management (reusable, zero deps)
  backend/                 llama.cpp-specific configuration
  proxy/                   Proxy core (server, config, errors)
    translation/           API format conversion (pure functions)
    middleware/             Optional features (cache, queue, budget, etc.)
    monitoring/            Observability (stats, dashboard, discovery)
tests/                     Unit and integration tests
docs/                      User documentation
```

### Code Style

- Follow existing patterns in the codebase
- Use type hints for function signatures
- Add docstrings to public functions and classes
- Keep modules focused — one responsibility per file
- All shared state must be thread-safe (use `threading.Lock`)

### Running Tests

```bash
# All tests
python3 -m unittest discover tests/ -v

# Single module
python3 -m unittest tests.test_translate_request -v

# With type checking
python3 -m mypy src/ --ignore-missing-imports

# With linting
python3 -m ruff check src/ tests/
```

### Writing Tests

- Add tests to the appropriate `tests/test_*.py` module
- Test both happy paths and edge cases
- Translation tests should verify exact SDK format compliance
- Use `unittest.mock` for isolating components — but avoid mocking when you can test pure functions directly

## Submitting Changes

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass
4. Open a pull request with:
   - A clear description of what changed and why
   - Any relevant context or discussion links

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (OS, Python version, llama.cpp version)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
