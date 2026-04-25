# Nearest-Neighbor Conditional Mutual Information (CMIknn) Test

CMIknn is a non-parametric conditional independence test based on a $k$-nearest-neighbour estimator of conditional mutual information (CMI), combined with a local-permutation procedure for p-values (Runge, 2018). It avoids parametric model choice and detects general non-linear dependencies, making it a flexible default when the dependence structure is unknown.

## Mathematical Formulation

The conditional mutual information of $X$ and $Y$ given $Z$ is

```{math}
I(X; Y \mid Z) = \int p(x, y, z) \log \frac{p(x, y \mid z)}{p(x \mid z)\, p(y \mid z)} \, dx \, dy \, dz
```

and is zero if and only if $X \perp Y \mid Z$. CMIknn estimates $I(X; Y \mid Z)$ from the $k$-th nearest-neighbour distances in the joint $(X, Y, Z)$ space using a Kraskov-style estimator (Kraskov et al., 2004) generalised to the conditional setting.

P-values are computed via a **local permutation test**: for each sample $i$, the value $Y_i$ is shuffled with a $Y_j$ from a sample $j$ whose conditioning value $Z_j$ is in the neighbourhood of $Z_i$. The shuffled sample preserves the marginal $Z$-conditional distribution while breaking the $X$-$Y$ link, so the empirical distribution of the test statistic under the null is generated locally rather than globally (Runge, 2018).

## Assumptions

- **Sample size**: Density estimation via $k$NN improves with sample size; small samples yield noisy estimates and conservative p-values.
- **Choice of $k$**: A larger $k$ smooths the estimator at the cost of bias; the default $k$ is a reasonable starting point.
- **Continuous variables**: Vanilla CMIknn assumes continuous data; for mixed types use :doc:`/tests/cmiknn_mixed_test` or :doc:`/tests/mcmiknn_test`.

## Code Example

```python
import numpy as np
from citk.tests import CMIknn

# Non-linear chain: X -> Z -> Y
n = 300
X = np.random.randn(n)
Z = np.tanh(X) + 0.2 * np.random.randn(n)
Y = Z**2 + 0.2 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
cmiknn_test = CMIknn(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = cmiknn_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = cmiknn_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.nearest_neighbor_tests.CMIknn`.

## References

Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E, 69*(6), 066138.

Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS 2018)*, 938-947.
