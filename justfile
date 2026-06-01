venv:
    uv sync
    cp -n .env.example .env 2>/dev/null || true
    @echo "User environment ready"

venv-dev:
    uv sync --group dev
    uv run prek install
    @echo "Dev environment ready"

venv-test:
    uv sync --group test
    @echo "Test environment ready"

upd-hooks:
    prek uninstall
    prek install

lint:
    uv run ruff check . --fix
    uv run ruff format .

typecheck:
    uv run ty check src/scald

check: lint typecheck

test-unit:
    uv run pytest tests/ -v
