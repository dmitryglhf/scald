# Actor-Critic Pattern

Scald uses a collaborative Actor-Critic pattern where two specialized agents work together to solve ML tasks.

## Overview

The Actor-Critic pattern divides responsibility:

- **Actor**: Proposes solutions
- **Critic**: Evaluates solutions and provides feedback

Through iterative refinement, this pattern converges on high-quality solutions.

## Actor: The Solver

### Responsibilities

The Actor is a data scientist agent that:

1. Analyzes training and test data
2. Performs exploratory data analysis
3. Identifies preprocessing needs
4. Engineers features
5. Selects appropriate algorithms
6. Trains and evaluates models
7. Generates predictions

### Tools

The Actor has access to all 6 MCP servers:

- **data-preview**: Inspect data structure and samples
- **data-analysis**: Compute statistics, correlations, distributions
- **data-processing**: Apply encodings, scaling, transformations
- **machine-learning**: Train models (CatBoost, LightGBM, XGBoost)
- **file-operations**: Read/write data and artifacts
- **sequential-thinking**: Break down complex reasoning

### Workflow

In each iteration, the Actor:

1. Reviews Critic's feedback (if not first iteration)
2. Analyzes data characteristics
3. Decides on preprocessing strategy
4. Applies transformations
5. Trains model with selected algorithm
6. Evaluates performance on validation set
7. Generates predictions for test data
8. Submits solution to Critic

## Critic: The Reviewer

### Responsibilities

The Critic evaluates solutions and:

1. Reviews Actor's code and approach
2. Checks for logical errors or issues
3. Assesses model performance
4. Provides constructive feedback
5. Suggests specific improvements
6. Decides to accept or request refinement

### No Tools

Critically, the Critic **does not** have access to MCP servers. This ensures:

- Reviews are based on code quality and reasoning
- No direct data access prevents overfitting to specific examples
- Feedback focuses on methodology, not implementation details

### Decision Making

The Critic evaluates:

- **Code Quality**: Is the approach sound?
- **Data Handling**: Are preprocessing steps appropriate?
- **Model Selection**: Is the algorithm suitable for the task?
- **Performance**: Are metrics acceptable?
- **Generalization**: Will this work on test data?

Based on evaluation, the Critic either:

- **Accepts**: Solution is good enough, proceed to next iteration
- **Rejects**: Provide specific feedback for improvement

## Iteration Loop

```
Iteration 1:
  Actor analyzes data → trains model → submits solution
  Critic reviews → provides feedback

Iteration 2:
  Actor reads feedback → adjusts approach → trains improved model
  Critic reviews → provides feedback

...

Iteration N:
  Actor refines final solution
  Critic accepts → predictions generated
```

## Feedback Examples

### Typical Critic Feedback

**Iteration 1:**
> "The initial model shows promise (F1: 0.72), but there are issues:
> 1. Several categorical features weren't encoded
> 2. No handling of missing values in 'age' column
> 3. Consider feature scaling for tree-based models
> Please address these in the next iteration."

**Iteration 3:**
> "Good improvement (F1: 0.84). The preprocessing is solid, but:
> 1. Try feature engineering on 'date' column (extract month/day)
> 2. Consider interaction features between 'age' and 'income'
> Current solution is acceptable but can be better."

**Iteration 5:**
> "Excellent work (F1: 0.89). The solution is comprehensive:
> - Proper encoding and scaling
> - Good feature engineering
> - Well-tuned hyperparameters
> This solution is ready for production."

## Memory Integration

Both agents benefit from long-term memory:

### Actor Memory

Retrieves past solutions for similar tasks:

- Preprocessing strategies that worked
- Successful feature engineering patterns
- Algorithm choices for similar data
- Hyperparameter configurations

### Critic Memory

Recalls feedback patterns:

- Common pitfalls to watch for
- Quality standards for different task types
- Evaluation criteria that matter

## Convergence

The loop typically runs 5 iterations (configurable). Convergence happens when:

1. Critic accepts solution with high confidence
2. Performance plateaus across iterations
3. Maximum iterations reached

## Benefits

### Why Actor-Critic?

**vs. Single Agent:**
- Separation of concerns reduces errors
- Critic catches Actor's mistakes
- Iterative refinement improves quality

**vs. Hard-coded Pipeline:**
- Flexible adaptation to data characteristics
- Learning from experience
- Natural language feedback is interpretable

**vs. Other AutoML:**
- Fewer wasted iterations (guided by feedback)
- Transparent reasoning
- Transfer learning via memory

## Configuration

Control the Actor-Critic loop:

```python
scald = Scald(
    max_iterations=5,      # Number of refinement cycles
    # Other options...
)
```

## Next Steps

- [MCP Servers](mcp-servers.md) - Tools available to the Actor
- [Python API](usage/api.md) - Programmatic control
- [Configuration](usage/configuration.md) - Tuning parameters
