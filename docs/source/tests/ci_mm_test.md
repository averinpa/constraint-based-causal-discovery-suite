# Symmetric Mixed-Model Conditional Independence (CiMM) Test

The CiMM test is a symmetric, regression-based conditional independence test for mixed continuous and discrete data, originating from the R `MXM` package (Tsagris et al., 2018). It runs two complementary likelihood-ratio tests in opposite directions and combines them, providing better small-sample stability than a single asymmetric regression test.

## Mathematical Formulation

For each pair $(X, Y)$ and conditioning set $Z$, CiMM fits two pairs of nested regressions, one with $X$ as the response and one with $Y$ as the response:

```{math}
\Lambda_{X \mid Y, Z} = -2 \left[ \ell(X \sim Z) - \ell(X \sim Y, Z) \right]
```

```{math}
\Lambda_{Y \mid X, Z} = -2 \left[ \ell(Y \sim Z) - \ell(Y \sim X, Z) \right]
```

Each direction yields a likelihood-ratio statistic with an asymptotic $\chi^2$ distribution. The link function for each regression is chosen automatically by the response type: linear regression for continuous responses, logistic regression for binary or categorical responses (Tsagris et al., 2018). The two p-values are combined into a final symmetric test statistic.

## Assumptions

- **Correctly specified link per variable type**: Linear or logistic must be reasonable approximations for each variable's conditional mean.
- **R + MXM available**: This wrapper requires `rpy2` and the R `MXM` package to be installed.
- **Variable type declarations**: A `data_type` array specifies which columns are continuous and which are discrete, so the right link is chosen per regression.
- **Asymptotic regime**: P-values use the chi-square approximation; small-sample reliability depends on category counts.

## Code Example

```python
import numpy as np
from citk.tests import CiMM

# Mixed chain: continuous X -> binary Z -> continuous Y
n = 400
X = np.random.randn(n)
Z_logits = 1.5 * X
Z = (np.random.rand(n) < 1 / (1 + np.exp(-Z_logits))).astype(int)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Declare per-variable types: 0 = continuous, 1 = discrete
data_type = np.array([[0, 0, 1]])

# Initialize the test
ci_mm_test = CiMM(data, data_type=data_type)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = ci_mm_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = ci_mm_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.regression_tests.CiMM`.

## References

Tsagris, M., Borboudakis, G., Lagani, V., & Tsamardinos, I. (2018). Constraint-based causal discovery with mixed data. *International Journal of Data Science and Analytics, 6*(1), 19-30.
