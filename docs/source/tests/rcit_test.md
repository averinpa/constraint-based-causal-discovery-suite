# Randomized Conditional Independence Test (RCIT)

RCIT is the random-Fourier-feature relaxation of the kernel CI test of Zhang et al. (2011), introduced by Strobl et al. (2019). RCIT approximates KCIT by working in a finite-dimensional random feature space and scales linearly in $n$ in practice, returning accurate p-values much faster than KCIT in the large-sample regime; constraint-based causal discovery run with RCIT recovers graphs at least as accurate as with KCIT but with large run-time reductions (Strobl et al., 2019).

**Intuition.** A shift-invariant kernel can be approximated by random cosine features drawn from its spectral distribution (Strobl et al., 2019). Replacing the $n \times n$ Gram matrix with a $d_f$-dimensional feature matrix avoids cubic kernel eigendecompositions while preserving most of KCIT's statistical power on continuous data (Strobl et al., 2019).

## Mathematical Formulation

Let $\phi_X$, $\phi_Y$, $\phi_Z$ denote $d_f$-dimensional random Fourier feature maps for $X$, $Y$, $Z$ (Strobl et al., 2019). RCIT residualises the feature maps of $X$ (or the extended variable $\ddot{X} = (X, Z)$) and $Y$ against $\phi_Z$ via empirical cross-covariance estimates:

```{math}
\tilde{\phi}_X = \phi_X - \hat{C}_{XZ} \hat{C}_{ZZ}^{-1} \phi_Z, \qquad \tilde{\phi}_Y = \phi_Y - \hat{C}_{YZ} \hat{C}_{ZZ}^{-1} \phi_Z
```

(Strobl et al., 2019). The test statistic is the squared Frobenius norm of the empirical cross-covariance of the residualised features:

```{math}
T_{\mathrm{RCIT}} = \| \hat{C}_{\tilde{X} \tilde{Y}} \|_{\mathrm{HS}}^2
```

Under the null $X \perp Y \mid Z$, $T_{\mathrm{RCIT}}$ converges to a weighted sum of $\chi^2_1$ variables; RCIT calibrates p-values by moment-matching to a mixture of chi-squared distributions, the same family of approximations used in KCIT (Strobl et al., 2019).

## Assumptions

- **Continuous data.** RCIT is designed for continuous variables (Strobl et al., 2019).
- **Shift-invariant kernel.** RFF approximate shift-invariant kernels (Gaussian RBF by default), which do not naturally represent delta kernels for categorical inputs (Strobl et al., 2019).
- **R + RCIT available.** This wrapper requires `rpy2` and the R `RCIT` package (from `ericstrobl/RCIT` on GitHub) (Strobl et al., 2019).
- **Approximation quality.** The number of random features $d_f$ trades approximation accuracy against speed; Strobl et al. (2019) report $d_f$ between 5 and 25 suffices for most settings, with sensitivity rising in conditioning-set dimensionality.
- **Linear-time complexity.** RCIT scales as $O(n d_f^2)$ in practice (Strobl et al., 2019).

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

Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate kernel-based conditional independence tests for fast non-parametric causal discovery. *Journal of Causal Inference, 7*(1), 20180017.

Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011). Kernel-based conditional independence test and application in causal discovery. *Proceedings of the 27th Conference on Uncertainty in Artificial Intelligence (UAI 2011)*, 804-813.
