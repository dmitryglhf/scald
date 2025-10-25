.PHONY: help install setup test

# Detect OS
UNAME_S := $(shell uname -s)
OS_ID := $(shell if [ -f /etc/os-release ]; then . /etc/os-release && echo $$ID; fi)

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install everything (deps, Docker, container-use, Python env)
	@echo "Installing SCALD environment..."
	@$(MAKE) install-deps
	@$(MAKE) install-docker
	@$(MAKE) install-container-use
	@$(MAKE) setup
	@echo "Installation complete"
	@echo "Next: cp .env.example .env and add OPENROUTER_API_KEY"

install-deps: ## Install system dependencies
ifeq ($(OS),Windows_NT)
	@echo "Windows: Install Python 3.11+, Git, Docker Desktop manually"
else ifeq ($(UNAME_S),Linux)
	@if [ "$(OS_ID)" = "arch" ]; then \
		sudo pacman -Sy --needed --noconfirm python python-pip git base-devel curl; \
	elif [ "$(OS_ID)" = "ubuntu" ] || [ "$(OS_ID)" = "debian" ]; then \
		sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv git build-essential curl; \
	fi
else ifeq ($(UNAME_S),Darwin)
	@brew install python@3.11 git
endif

install-docker: ## Install Docker
ifeq ($(OS),Windows_NT)
	@echo "Windows: Install Docker Desktop from https://www.docker.com/products/docker-desktop"
else ifeq ($(UNAME_S),Linux)
	@if ! command -v docker &> /dev/null; then \
		if [ "$(OS_ID)" = "arch" ]; then \
			sudo pacman -S --needed --noconfirm docker docker-compose; \
			sudo systemctl enable docker && sudo systemctl start docker; \
			sudo usermod -aG docker $$USER; \
		elif [ "$(OS_ID)" = "ubuntu" ] || [ "$(OS_ID)" = "debian" ]; then \
			curl -fsSL https://get.docker.com | sudo sh; \
			sudo usermod -aG docker $$USER; \
			sudo systemctl enable docker && sudo systemctl start docker; \
		fi; \
		echo "Log out and back in for Docker permissions"; \
	fi
else ifeq ($(UNAME_S),Darwin)
	@if ! command -v docker &> /dev/null; then brew install --cask docker; fi
endif

install-container-use: ## Install container-use
ifeq ($(UNAME_S),Darwin)
	@if ! command -v container-use &> /dev/null; then \
		brew tap dagger/tap && brew install container-use; \
	fi
else
	@if ! command -v container-use &> /dev/null; then \
		curl -fsSL https://raw.githubusercontent.com/dagger/container-use/main/install.sh | bash; \
	fi
endif

setup: ## Setup Python environment
	@echo "Setting up Python environment..."
	@if ! command -v uv &> /dev/null; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv venv
	@uv sync --group dev
	@echo "Done. Activate: source .venv/bin/activate"

test: ## Run tests
	@python -m pytest tests/ -v

test-mcp: ## Run MCP tests
	@python -m pytest tests/mcp/ -v

lint: ## Check code quality
	@ruff check src/
	@ruff format --check src/
	@mypy src/

format: ## Format code
	@ruff format src/ tests/
	@ruff check --fix src/ tests/
