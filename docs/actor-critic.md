# Actor-Critic Pattern

Scald implements collaborative problem-solving through two specialized agents with complementary roles: the Actor proposes solutions while the Critic evaluates and guides refinement.

## Actor: The Solver

The Actor is an LLM-powered data scientist with access to six MCP servers. Each iteration begins by reviewing the Critic's previous feedback, then analyzing data characteristics to inform strategy. The Actor makes decisions about encoding methods, scaling approaches, feature engineering, and algorithm selection based on data properties rather than fixed rules.

Using MCP servers as tools, the Actor performs exploratory analysis, applies transformations, trains gradient boosting models (CatBoost, LightGBM, or XGBoost), and generates predictions. All work produces executable code saved as artifacts, ensuring reproducibility. The Actor leverages past experiences from memory to inform decisions about preprocessing and modeling strategies.

## Critic: The Reviewer

The Critic evaluates solutions without access to MCP servers, ensuring objective review based on code quality and methodology rather than direct data manipulation. Each evaluation examines preprocessing appropriateness, algorithm suitability for the task, potential issues or edge cases, and overall solution quality.

The Critic provides specific, actionable feedback rather than generic observations. Feedback might identify unhandled missing values, suggest feature engineering opportunities, recommend algorithm adjustments, or point out logical errors. Based on evaluation, the Critic either accepts the solution for the next iteration or requests specific improvements.

## Iteration Dynamics

The first iteration starts with the Actor analyzing data and creating an initial solution. The Critic reviews this baseline and provides targeted feedback. In subsequent iterations, the Actor incorporates feedback progressively, addressing issues while maintaining what worked. Each cycle refines the approach through specific adjustments rather than complete rebuilds.

Convergence occurs when the Critic accepts the solution with high confidence, performance plateaus across iterations, or maximum iterations are reached. Typically, 5 iterations provide good balance between quality and cost.

## Example Feedback Evolution

**Iteration 1:** "F1 score of 0.72 shows promise, but categorical features lack encoding and missing values in 'age' remain unhandled. Apply one-hot encoding and imputation in the next iteration."

**Iteration 3:** "Improvement to F1 0.84 with solid preprocessing. Consider extracting temporal features from 'date' column and exploring interaction terms between 'age' and 'income'. Current solution is acceptable but refinable."

**Iteration 5:** "Excellent F1 of 0.89 with comprehensive feature engineering, proper encoding, and well-tuned hyperparameters. Solution is production-ready."

## Memory Integration

Both agents benefit from ChromaDB-based long-term memory. The Actor retrieves preprocessing strategies, feature engineering patterns, and algorithm choices from similar past tasks. The Critic recalls common pitfalls, quality standards, and evaluation criteria relevant to the task type. This enables transfer learning across problems and faster convergence on novel datasets.

## Benefits

Separating solving from reviewing reduces errors through independent evaluation. Iterative refinement with targeted feedback produces higher quality outcomes than single-pass approaches. Natural language feedback provides interpretable insight into decision-making. Memory-based transfer learning reduces wasted iterations on similar problems.

## Configuration

Control the refinement process through iteration count:

```python
scald = Scald(max_iterations=5)  # Default balance
```

Increase for complex tasks requiring more refinement; decrease for simple problems or faster prototyping.

Continue to [MCP Servers](mcp-servers.md) to understand the Actor's toolkit.
