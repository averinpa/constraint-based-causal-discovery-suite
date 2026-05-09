# Randomized Conditional Correlation Test (RCoT)

RCoT is the second of the two random-Fourier-feature kernel CI relaxations introduced by Strobl et al. (2019). It uses the same RFF strategy as :doc:`/tests/rcit_test` but formulates the test through a covariance-based statistic that avoids the $d_f^2$ matrix product, reducing complexity to $O(n d_f)$ and trading a small amount of power for a substantial reduction in compute (Strobl et al., 2019).

**Intuition.** RCIT residualises both feature embeddings against $\phi_Z$ and tests the resulting cross-covariance; RCoT instead tests the **partial cross-covariance** of $\phi_X$ and $\phi_Y$ given $\phi_Z$ directly (Strobl et al., 2019). Both approaches share the same RFF approximation of the underlying RBF kernel (Strobl et al., 2019).

## Mathematical Formulation

Using the same feature maps as RCIT (Strobl et al., 2019), RCoT computes the partial cross-covariance:

```{math}
T_{\mathrm{RCoT}} = \left\| \hat{C}_{XY} - \hat{C}_{XZ} \hat{C}_{ZZ}^{-1} \hat{C}_{ZY} \right\|_{\mathrm{HS}}^2
```

(Strobl et al., 2019). Avoiding the $d_f^2$ matrix product reduces complexity from $O(n d_f^2)$ for RCIT to $O(n d_f)$ for RCoT (Strobl et al., 2019). Under the null $X \perp Y \mid Z$ the asymptotic distribution is again a weighted sum of $\chi^2_1$ variables, calibrated by moment-matching to a mixture of chi-squared distributions (Strobl et al., 2019).

## Assumptions

- **Continuous data.** RCoT is designed for continuous variables (Strobl et al., 2019).
- **R + RCIT available.** This wrapper requires `rpy2` and the R `RCIT` package (Strobl et al., 2019).
- **Acceptable approximation.** RCoT is faster but slightly less powerful than RCIT against alternatives where the dependence is concentrated in directions of $Z$ that the projection discards; for small or strongly nonlinear conditioning sets, prefer :doc:`/tests/rcit_test` or :doc:`/tests/kci_test` (Strobl et al., 2019; Zhang et al., 2011).
- **Linear-time complexity.** RCoT scales as $O(n d_f)$, even faster than RCIT (Strobl et al., 2019).
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

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

Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate kernel-based conditional independence tests for fast non-parametric causal discovery. *Journal of Causal Inference, 7*(1), 20180017.

Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011). Kernel-based conditional independence test and application in causal discovery. *Proceedings of the 27th Conference on Uncertainty in Artificial Intelligence (UAI 2011)*, 804-813.
