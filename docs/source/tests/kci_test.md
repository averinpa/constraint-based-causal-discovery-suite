# Kernel Conditional Independence (KCI) Test

The Kernel Conditional Independence (KCI) test is a non-parametric method for assessing conditional independence. Unlike regression-based tests that assume linear relationships, KCI can capture complex, non-linear dependencies between variables without making strong distributional assumptions. This makes it a powerful tool for causal discovery and feature selection in complex systems. This implementation is a wrapper around the KCI test provided in the `causal-learn` library.

## Mathematical Formulation

The KCI test is built upon the **Hilbert-Schmidt Independence Criterion (HSIC)**, a measure of dependence between two variables, *X* and *Y* (Gretton et al., 2005). HSIC operates by mapping the data into a high-dimensional feature space, known as a Reproducing Kernel Hilbert Space (RKHS), and then computing the Hilbert-Schmidt norm of the cross-covariance operator between them. In this space, even complex dependencies can manifest as linear correlations. The key property is that HSIC is zero if and only if the variables are independent (for a universal kernel, like the Gaussian RBF).

The KCI test extends this principle to the *conditional* case ($X \perp Y | Z$). It tests for independence by essentially performing a regression in the RKHS and then testing whether the residuals are independent. The final test statistic is derived from the normalized HSIC, and its distribution under the null hypothesis of conditional independence has been derived, allowing for the calculation of a p-value (Zhang et al., 2011).

## Properties and Assumptions

*   **Non-parametric**: The test does not assume linearity, normality, or any specific data distribution.
*   **Kernel Choice**: Its performance can be influenced by the choice of kernel function (e.g., Gaussian, Polynomial). While default kernels often perform well, the selection is an important parameter.
*   **Computational Cost**: KCI is more computationally intensive than linear tests, with a complexity that is at least quadratic in the sample size. This can make it slow for very large datasets.

## Code Example

```python
import numpy as np
from citk.tests import KCI

# Generate data with a non-linear relationship: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = np.cos(X) + np.random.randn(n) * 0.1
Y = Z**2 + np.random.randn(n) * 0.1
data = np.vstack([X, Y, Z]).T

# Initialize the test
kci_test = KCI(data)

# Test for unconditional independence (should be dependent)
p_unconditional = kci_test(0, 1)
print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

# Test for conditional independence given Z (should be independent)
p_conditional = kci_test(0, 1, [2])
print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.kernel_tests.KCI`.

## References

*   Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005). Measuring Statistical Dependence with Hilbert-Schmidt Norms. In *Proceedings of the 16th International Conference on Algorithmic Learning Theory (ALT 2005)*.
*   Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011). Kernel-based conditional independence test and application in causal discovery. In *Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence (UAI 2011)*. 