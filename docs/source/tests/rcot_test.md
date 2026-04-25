# Randomized Conditional Correlation Test (RCoT)

RCoT is a faster variant of :doc:`/tests/rcit_test` that conditions on a lower-dimensional projection of the conditioning set, trading a small amount of power for a substantial reduction in compute (Strobl et al., 2019). Like RCIT, it approximates the Gaussian RBF kernel via random Fourier features and lives in the R `RCIT` package.

## Mathematical Formulation

RCoT uses the same random Fourier feature maps $\phi_X, \phi_Y, \phi_Z$ as RCIT (Rahimi & Recht, 2007), but instead of residualising via the full empirical cross-covariance with respect to $\phi_Z$, it tests the **partial cross-covariance** of $\phi_X$ and $\phi_Y$ given a coarser projection of $\phi_Z$:

```{math}
T_{\mathrm{RCoT}} = \left\| \hat{C}_{XY} - \hat{C}_{XZ} \hat{C}_{ZZ}^{-1} \hat{C}_{ZY} \right\|_{\mathrm{HS}}^2
```

The lower-dimensional approximation reduces the cost of inverting $\hat{C}_{ZZ}$ from cubic to roughly linear in the number of features used, at the cost of slightly reduced power against alternatives where the dependence is concentrated in the discarded directions of $Z$. Under the null $X \perp Y \mid Z$, the asymptotic distribution is again a weighted sum of $\chi^2_1$ variables, evaluated from the spectrum of the partial cross-covariance matrix (Strobl et al., 2019).

## Assumptions

- **Continuous data**: RCoT is designed for continuous variables.
- **R + RCIT available**: This wrapper requires `rpy2` and the R `RCIT` package.
- **Acceptable approximation**: The conditioning-set projection is chosen by the upstream defaults; for small or strongly nonlinear conditioning sets, prefer :doc:`/tests/rcit_test` or :doc:`/tests/kci_test`.

## Code Example

```python
import numpy as np
from citk.tests import RCoT

# Non-linear chain: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = np.tanh(X) + 0.2 * np.random.randn(n)
Y = Z**2 + 0.2 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
rcot_test = RCoT(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = rcot_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = rcot_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.kernel_tests.RCoT`.

## References

Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems 20 (NIPS 2007)*, 1177-1184.

Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate kernel-based conditional independence tests for fast non-parametric causal discovery. *Journal of Causal Inference, 7*(1), 20180017.
