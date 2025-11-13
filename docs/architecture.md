# Architecture

Scald's architecture is built around collaborative agents, MCP servers, and a learning system.

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                      Scald                          │
│  (Orchestrator)                                     │
└──────────────┬──────────────────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
┌─────▼─────┐     ┌─────▼─────┐
│   Actor   │────▶│   Critic  │
│ (Solver)  │◀────│ (Reviewer)│
└─────┬─────┘     └───────────┘
      │
      │ Uses
      │
┌─────▼──────────────────────────────────┐
│          MCP Servers                   │
│  • data-analysis                       │
│  • data-preview                        │
│  • data-processing                     │
│  • machine-learning                    │
│  • file-operations                     │
│  • sequential-thinking                 │
└────────────────────────────────────────┘
```

## Core Components

### Scald (Orchestrator)

The main controller that:

- Manages the Actor-Critic loop
- Coordinates iterations and convergence
- Tracks costs, tokens, and performance
- Handles logging and artifact storage
- Manages workspace isolation

### Actor (Data Scientist Agent)

An LLM-powered agent that:

- Explores and analyzes data
- Engineers features and handles preprocessing
- Selects and trains models
- Generates predictions
- Uses MCP servers as tools

The Actor has access to 6 specialized MCP servers that provide data operations, ML capabilities, and structured thinking.

### Critic (Reviewer Agent)

An LLM-powered agent that:

- Evaluates Actor's solutions
- Provides constructive feedback
- Decides whether to accept or reject solutions
- Suggests improvements
- Determines convergence

The Critic doesn't have access to MCP servers—it reviews based on the Actor's code and results.

### Memory Manager

A ChromaDB-based system that:

- Stores past task experiences
- Uses Jina embeddings for semantic search
- Retrieves relevant examples for new tasks
- Enables transfer learning across problems

### MCP Servers

Six specialized servers provide tools for:

1. **data-analysis**: Statistical analysis, correlations, distributions
2. **data-preview**: Quick data inspection and schema viewing
3. **data-processing**: Encoding, scaling, feature engineering
4. **machine-learning**: Model training, prediction, evaluation
5. **file-operations**: Reading/writing data and artifacts
6. **sequential-thinking**: Structured problem decomposition

## Workflow

1. **Initialization**: Scald creates isolated workspace and session
2. **Memory Retrieval**: Past relevant experiences loaded
3. **Iteration Loop** (default 5 times):
   - Actor analyzes data using MCP tools
   - Actor preprocesses and trains models
   - Actor generates code and artifacts
   - Critic reviews solution quality
   - Critic provides feedback for next iteration
4. **Convergence**: Best solution selected
5. **Prediction**: Final model applied to test data
6. **Cleanup**: Artifacts saved, logs written, costs reported

## Data Flow

```
train.csv ──┐
            ├──▶ Actor ──▶ preprocessing ──▶ model ──▶ Critic
test.csv ───┘                    │                        │
                                 │                        │
                            artifacts/                feedback
                                 │                        │
                                 └────────────────────────┘
                                          │
                                   predictions.csv
```

## Session Management

Each run creates a timestamped session directory:

```
sessions/session_YYYYMMDD_HHMMSS/
├── session.log          # Execution logs
├── artifacts/           # Generated code
│   ├── actor_iter_1.py
│   ├── actor_iter_2.py
│   └── ...
└── predictions.csv      # Final output
```

## Configuration

Scald behavior is controlled via:

- Environment variables (`.env`)
- Constructor parameters
- Runtime arguments

See [Configuration](usage/configuration.md) for details.

## Scalability

Scald scales through:

- **Workspace isolation**: Each session runs in separate directory
- **Stateless execution**: Agents don't maintain state between iterations
- **Memory efficiency**: ChromaDB handles large experience databases
- **Cost tracking**: Monitor API usage per session

## Next Steps

- [Actor-Critic Pattern](actor-critic.md) - Deep dive into agent collaboration
- [MCP Servers](mcp-servers.md) - Learn about available tools
- [Python API](usage/api.md) - Programmatic usage
