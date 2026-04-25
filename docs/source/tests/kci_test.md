# Kernel Conditional Independence (KCI) Test

The Kernel Conditional Independence (KCI) test is a non-parametric method for assessing conditional independence on continuous data. Unlike regression-based tests that assume linearity, KCI captures complex non-linear dependencies through kernel methods, making it a strong default for general-purpose CI testing in causal discovery and feature selection (Zhang et al., 2011). This implementation wraps the Python KCI test from the `causal-learn` library, which is the canonical kernel implementation in `citk`.

## Mathematical Formulation

KCI is built on the **Hilbert-Schmidt Independence Criterion (HSIC)**, a measure of dependence in a Reproducing Kernel Hilbert Space (RKHS). For variables $X$ and $Y$, HSIC is the squared Hilbert-Schmidt norm of the cross-covariance operator between their kernel embeddings (Gretton et al., 2005):

```{math}
\mathrm{HSIC}(X, Y) = \| C_{XY} \|_{\mathrm{HS}}^2
```

For a universal kernel (e.g., Gaussian RBF), $\mathrm{HSIC}(X, Y) = 0$ if and only if $X \perp Y$.

The KCI extension to the conditional case $X \perp Y \mid Z$ residualizes the kernel-feature embeddings of $X$ and $Y$ against $Z$ in the RKHS, then tests whether the residual cross-covariance is zero. The asymptotic null distribution of the resulting test statistic is a weighted sum of $\chi^2_1$ variables; p-values are computed from this distribution (Zhang et al., 2011).

## Assumptions

- **Continuous data**: KCI is designed for continuous variables.
- **Kernel choice**: Performance depends on the kernel; the default Gaussian RBF works well in most settings, with bandwidth selected by the median heuristic.
- **Computational cost**: At least quadratic in sample size due to the kernel Gram matrices, which limits practical sample sizes (rule of thumb: a few thousand observations).

## Code Example

```python
import numpy as np
from citk.tests import KCI

# Non-linear chain: X -> Z -> Y
n = 300
X = np.random.randn(n)
Z = np.cos(X) + 0.1 * np.random.randn(n)
Y = Z**2 + 0.1 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
kci_test = KCI(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = kci_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = kci_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.kernel_tests.KCI`.

## References

Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005). Measuring statistical dependence with Hilbert-Schmidt norms. *Proceedings of the 16th International Conference on Algorithmic Learning Theory (ALT 2005)*, 63-77.

Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011). Kernel-based conditional independence test and application in causal discovery. *Proceedings of the 27th Conference on Uncertainty in Artificial Intelligence (UAI 2011)*, 804-813.
