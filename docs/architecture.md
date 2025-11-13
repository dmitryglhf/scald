# Architecture

Scald orchestrates collaborative AI agents, MCP servers, and a learning system to automate machine learning workflows.

<div align="center">
  <img src="assets/arch.svg" alt="Scald Architecture" width="700"/>
</div>

## Core Components

**Scald Orchestrator** manages the Actor-Critic loop, tracking iterations, costs, and performance metrics. It handles workspace isolation, session management, and artifact preservation. Each run creates an independent session with dedicated logging and storage.

**Actor Agent** is an LLM-powered data scientist that explores data, engineers features, and trains models. It has access to six MCP servers providing data operations, statistical analysis, preprocessing, model training, file management, and structured reasoning. The Actor generates executable code for each pipeline stage, storing artifacts for reproducibility.

**Critic Agent** evaluates Actor solutions without access to MCP servers. This separation ensures objective review based on code quality and reasoning rather than direct data manipulation. The Critic provides targeted feedback, identifies issues, and decides whether to accept solutions or request refinement.

**Memory Manager** uses ChromaDB with Jina embeddings to store and retrieve past experiences. When facing a new task, the system queries memory for similar problems, retrieving relevant preprocessing strategies, feature engineering patterns, and algorithm choices. This enables transfer learning across datasets.

**MCP Servers** provide specialized capabilities: data-preview for inspection, data-analysis for statistics and correlations, data-processing for transformations, machine-learning for model training, file-operations for I/O, and sequential-thinking for complex reasoning decomposition.

## Workflow

Initialization creates an isolated workspace and session directory. The system retrieves relevant experiences from memory based on task characteristics. The Actor-Critic loop then executes for the specified number of iterations.

Within each iteration, the Actor reviews previous feedback (if any), analyzes data characteristics, designs preprocessing strategy, trains models, and submits the solution. The Critic evaluates code quality, methodology, and performance, providing specific feedback for improvement. This continues until convergence or maximum iterations.

After the loop completes, the best solution is selected, applied to test data for final predictions, and all artifacts are saved with comprehensive logging and cost reports.

## Data Flow

Training and test CSV files flow to the Actor, which applies preprocessing, trains a model, and produces artifacts. The Critic reviews these artifacts and provides feedback, which influences the next iteration. The final model generates predictions saved to CSV.

## Session Management

Each execution creates a timestamped directory:

```
sessions/session_YYYYMMDD_HHMMSS/
├── session.log          # Execution logs
├── artifacts/           # Code per iteration
└── predictions.csv      # Final output
```

Sessions are independent and can run concurrently without interference.

## Scalability

Workspace isolation prevents conflicts between concurrent runs. Agents operate statelessly, requiring explicit context passing. ChromaDB efficiently handles large experience databases. Built-in cost tracking monitors API usage per session, enabling budget control.

Continue to [Actor-Critic Pattern](actor-critic.md) for deeper understanding of agent collaboration.
