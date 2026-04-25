# Quickstart

This guide provides a minimal, complete example of how to use `citk` for a conditional independence test within the `causal-learn` framework.

## Example: Running PC with a CITK Test

The following example demonstrates how to:
1. Generate synthetic data.
2. Run the PC algorithm.
3. Use a conditional independence test from `citk`.

```python
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import citk.tests 

# 1. Generate some data
# Here, X2 is a function of X0 and X1, creating a dependency.
np.random.seed(42)
data = np.random.randn(200, 3)
data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 1] + 0.1 * np.random.randn(200)

# 2. Run the PC algorithm using a citk test
# You can swap 'spearman' with any other available test, like
# "fisherz_citk", "gsq", "kci", "gcm", etc.
cg = pc(data, alpha=0.05, indep_test='spearman')

# 3. View the learned graph edges
# The output should reflect the dependencies in the data.
print("Learned Graph Edges:")
print(cg.G.get_edges())
``` 