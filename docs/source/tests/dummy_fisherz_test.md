# Dummy-Encoded Fisher's Z (DummyFisherZ) Test

DummyFisherZ is an adapter that one-hot encodes discrete variables before running pairwise :doc:`/tests/fisher_z_test` calls across the encoded columns, then aggregates the per-pair p-values via Fisher's combined probability test (Fisher, 1925). It is a practical option when the discrete variables in a problem have moderate cardinality and a linear-on-the-encoded-space dependence structure is plausible.

## Mathematical Formulation

For each variable $V$ in the input data:

- If $V$ is continuous, it contributes a single column to the expanded design matrix.
- If $V$ is categorical with $K$ levels, it contributes $K - 1$ one-hot indicator columns (drop-first to keep the design matrix full-rank).

For a triple $(X, Y, Z)$, the adapter runs :doc:`/tests/fisher_z_test` on every $(x_\text{col}, y_\text{col})$ pair drawn from the expanded $X$ and $Y$ blocks, conditioning on the union of expanded $Z$ columns (excluding the test pair itself to avoid degeneracy). This yields a set of p-values $\{p_1, \ldots, p_m\}$.

The aggregated p-value uses Fisher's combined probability test (Fisher, 1925):

```{math}
T = -2 \sum_{i=1}^{m} \ln p_i \;\sim\; \chi^2_{2m} \quad \text{under the null}
```

A single p-value is then obtained from the upper tail of the $\chi^2_{2m}$ distribution.

## Assumptions

- **Linear partial correlation captures dependence in the encoded space**: As with FisherZ, the test has low power against alternatives that one-hot encoding cannot linearise.
- **Discrete cardinalities are moderate**: Each additional level adds an indicator column, so high-cardinality discrete variables blow up the per-pair count $m$ and reduce power per pair.
- **Independence of per-pair p-values is approximate**: Fisher's combination assumes independent inputs; the p-values from overlapping conditioning sets are not strictly independent, so the combined p-value is approximate.

## Code Example

```python
import numpy as np
from citk.tests import DummyFisherZ

# Discrete chain: X -> Z -> Y
n = 600
X = np.random.randint(0, 3, size=n)
Z = (X + np.random.randint(0, 2, size=n)) % 3
Y = (Z + np.random.randint(0, 2, size=n)) % 3
data = np.vstack([X, Y, Z]).T

# Initialize the test
dummy_fisherz_test = DummyFisherZ(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = dummy_fisherz_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = dummy_fisherz_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.adapter_tests.DummyFisherZ`.

## References

Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd, Edinburgh.
