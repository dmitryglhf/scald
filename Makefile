.PHONY: help test lint format

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run tests
	@python -m pytest tests/ -v

lint: ## Check code quality
	@ruff check src/
	@ruff format --check src/
	@mypy src/

format: ## Format code
	@ruff format src/ tests/
	@ruff check --fix src/ tests/
