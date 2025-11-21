# Show available commands
default:
    @just --list

# Run tests
test:
    python -m pytest tests/ -v

# Check code quality
lint:
    ruff check src/
    ruff format --check src/
    mypy src/

# Format code
format:
    ruff format src/ tests/
    ruff check --fix src/ tests/

# Run pre-commit hooks on all files
prek-all:
    uv run prek run --all-files

# Run pre-commit hooks on staged files
prek:
    uv run prek run

# Install pre-commit git hooks
prek-install:
    uv run prek install

# Update pre-commit hook versions
prek-update:
    uv run prek autoupdate
