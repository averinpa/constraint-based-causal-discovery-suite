# Spearman's Rho Test

Spearman's rho is a rank-based partial-correlation test for continuous data, originally introduced by Spearman (1904) as a non-parametric measure of association between two variables. As a CI test it is a robust alternative to Fisher's Z when the relationship between variables is monotone but not linear (Kutner et al., 2005).

**Intuition.** Replacing the raw values with their ranks removes the linearity and Gaussianity assumptions of Pearson's correlation while still detecting monotone dependence (Spearman, 1904; Kutner et al., 2005). Applying the Fisher Z-transform to the resulting rank partial correlation gives an approximately Gaussian-calibrated statistic (Hotelling, 1953).

## Mathematical Formulation

The test computes the partial Pearson correlation on the ranked data, $r_s = \rho(R(X), R(Y) \mid R(Z))$, where $R(V)$ denotes the rank vector of variable $V$ (Spearman, 1904; Kutner et al., 2005). Ties are broken by midranks (Kutner et al., 2005). The variance-stabilising Z-transform of Hotelling (1953) is then applied to $r_s$:

```{math}
Z(r_s) = \frac{1}{2} \ln\!\left(\frac{1+r_s}{1-r_s}\right)
```

and the standardised statistic

```{math}
T = \sqrt{n - |Z| - 3} \cdot |Z(r_s)|
```

is referred to a standard normal distribution under the null (Hotelling, 1953; Anderson, 2003).

## Assumptions

- **Monotone dependence.** The test is sensitive to monotone (not necessarily linear) relationships; it does not require linearity or multivariate normality (Spearman, 1904; Kutner et al., 2005).
- **Approximate CI procedure.** Zero partial rank correlation is not equivalent to conditional independence in general; rank-based tests rely on additional structural assumptions and should be treated as approximate CI procedures (Baba et al., 2004).
- **Sample size.** The Z-transform calibration is asymptotic and effective degrees of freedom are $n - |Z| - 3$ (Hotelling, 1953; Anderson, 2003).

## Code Example

```python
import numpy as np
from citk.tests import Spearman

# Generate data with a non-linear, monotonic relationship
# X -> Z -> Y
n = 500
X = np.random.rand(n) * 5
Z = np.exp(X / 2) + np.random.randn(n) * 0.1
Y = np.log(Z**2) + np.random.randn(n) * 0.1
data = np.vstack([X, Y, Z]).T

# Initialize the test
spearman_test = Spearman(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = spearman_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = spearman_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.Spearman`.

## References

Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis* (3rd ed.). Wiley-Interscience.

Baba, K., Shibata, R., & Sibuya, M. (2004). Partial correlation and conditional correlation as measures of conditional independence. *Australian & New Zealand Journal of Statistics, 46*(4), 657-664.

Hotelling, H. (1953). New light on the correlation coefficient and its transforms. *Journal of the Royal Statistical Society: Series B (Methodological), 15*(2), 193-225.

Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). *Applied Linear Statistical Models* (5th ed.). McGraw-Hill Irwin.

Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology, 15*(1), 72-101.
