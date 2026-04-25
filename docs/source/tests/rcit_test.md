# Randomized Conditional Independence Test (RCIT)

RCIT is a kernel-based conditional independence test that uses random Fourier features to approximate the Gaussian RBF kernel, providing an order-of-magnitude speedup over exact KCI while preserving most of its statistical power on continuous data (Strobl et al., 2019). It is implemented in the R `RCIT` package and exposed via `rpy2`.

## Mathematical Formulation

Let $\phi_X$, $\phi_Y$, $\phi_Z$ denote $D$-dimensional random Fourier feature maps for $X$, $Y$, $Z$ respectively (Rahimi & Recht, 2007). RCIT residualises the feature maps of $X$ and $Y$ against $Z$ in this finite-dimensional space:

```{math}
\tilde{\phi}_X = \phi_X - \hat{C}_{XZ} \hat{C}_{ZZ}^{-1} \phi_Z, \qquad \tilde{\phi}_Y = \phi_Y - \hat{C}_{YZ} \hat{C}_{ZZ}^{-1} \phi_Z
```

where $\hat{C}$ are the empirical cross-covariance estimates. The test statistic is the squared Hilbert-Schmidt norm of the empirical cross-covariance of the residualised features:

```{math}
T_{\mathrm{RCIT}} = \| \hat{C}_{\tilde{X} \tilde{Y}} \|_{\mathrm{HS}}^2
```

Under the null $X \perp Y \mid Z$, $T_{\mathrm{RCIT}}$ is asymptotically a weighted sum of $\chi^2_1$ variables. P-values are computed by approximating this null distribution using the spectrum of the empirical cross-covariance matrix.

## Assumptions

- **Continuous data**: RCIT is designed for continuous variables.
- **R + RCIT available**: This wrapper requires `rpy2` and the R `RCIT` package (from `ericstrobl/RCIT` on GitHub).
- **Approximation quality**: The number of random features $D$ controls the approximation accuracy; the package defaults are appropriate for typical sample sizes.

## Code Example

```python
import numpy as np
from citk.tests import RCIT

# Non-linear chain: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = np.tanh(X) + 0.2 * np.random.randn(n)
Y = Z**2 + 0.2 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
rcit_test = RCIT(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = rcit_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = rcit_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.kernel_tests.RCIT`.

## References

Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems 20 (NIPS 2007)*, 1177-1184.

Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate kernel-based conditional independence tests for fast non-parametric causal discovery. *Journal of Causal Inference, 7*(1), 20180017.
