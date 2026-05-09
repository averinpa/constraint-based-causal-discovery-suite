# Kernel Conditional Independence (KCI) Test

The Kernel Conditional Independence test of Zhang et al. (2011) is a non-parametric CI test for continuous data that constructs an appropriate test statistic in a reproducing kernel Hilbert space (RKHS) and derives its asymptotic null distribution. Zhang et al. (2011) report that KCI outperforms competing methods especially when the conditioning set is large or the sample size is moderate. `citk` wraps the Python implementation from `causal-learn` as the canonical kernel implementation.

**Intuition.** Embed each variable into an RKHS via a (universal) kernel; for a universal kernel the population cross-covariance operator vanishes if and only if the variables are independent (Gretton et al., 2005). KCI extends this to the conditional case by residualising the kernel embeddings of $X$ and $Y$ against $Z$ and testing whether the residual cross-covariance is zero (Zhang et al., 2011).

## Mathematical Formulation

For variables $X$ and $Y$ with kernel cross-covariance operator $C_{XY}$, the Hilbert--Schmidt Independence Criterion (Gretton et al., 2005) is

```{math}
\mathrm{HSIC}(X, Y) = \| C_{XY} \|_{\mathrm{HS}}^2
```

and equals zero, for universal kernels, if and only if $X \perp Y$ (Gretton et al., 2005). KCI extends this to $X \perp Y \mid Z$ by residualising the kernel-feature embeddings of $X$ and $Y$ against $Z$ in the RKHS and testing whether the residual cross-covariance is zero (Zhang et al., 2011). The asymptotic null distribution of the resulting statistic is a weighted sum of $\chi^2_1$ variables; KCI approximates this by a two-parameter Gamma distribution for computational efficiency (Zhang et al., 2011).

## Assumptions

- **Continuous data.** KCI is designed for continuous variables (Zhang et al., 2011).
- **Kernel choice.** Performance depends on the kernel; Zhang et al. (2011) use Gaussian RBF kernels and the median heuristic for bandwidth selection.
- **Computational cost.** KCI requires kernel-matrix eigendecompositions and scales at least quadratically in $n$, limiting practical sample sizes (Zhang et al., 2011; Strobl et al., 2019).
- **No distributional assumption beyond kernel regularity.** KCI is fully non-parametric (Zhang et al., 2011).
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

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

Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate kernel-based conditional independence tests for fast non-parametric causal discovery. *Journal of Causal Inference, 7*(1), 20180017.

Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011). Kernel-based conditional independence test and application in causal discovery. *Proceedings of the 27th Conference on Uncertainty in Artificial Intelligence (UAI 2011)*, 804-813.
