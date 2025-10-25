# SCALD

Actor-Critic data science framework with MCP servers and container isolation.

## Installation

```bash
make install          # Installs Docker, container-use, and Python env via uv
cp .env.example .env  # Add OPENROUTER_API_KEY
```

## Usage

```python
from pathlib import Path
from scald import Scald
from scald.common.types import TaskType

scald = Scald(max_iterations=5)
result = await scald.run(
    csv_path=Path("data/train.csv"),
    target="target",
    task_type=TaskType.CLASSIFICATION,
)
```

## Architecture

- Actor: Creates isolated environments, analyzes data, trains models
- Critic: Evaluates solutions, provides feedback, decides acceptance
- MCP Servers: container-use, data_analysis, data_load, data_processing, machine_learning

## Development

```bash
make test      # Run tests
make lint      # Check code quality
make format    # Format code
make help      # Show all commands
```

## Platform Support

- Arch Linux, Ubuntu/Debian, macOS: Fully automated
- Windows: Requires WSL2

## Requirements

- Python 3.11+
- Docker
- container-use
- uv (Python package manager)
- OpenRouter API key
