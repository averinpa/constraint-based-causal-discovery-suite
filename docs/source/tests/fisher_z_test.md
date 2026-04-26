# Fisher's Z Test

Fisher's Z is the canonical partial-correlation--based conditional independence test for continuous data. Under multivariate normality, conditional independence between $X$ and $Y$ given $Z$ is equivalent to zero partial correlation $\rho(X, Y \mid Z)$ (Anderson, 2003).

**Intuition.** A variance-stabilising transform converts the (partial) Pearson correlation into a statistic that is asymptotically standard normal under the null, yielding analytic p-values without resampling (Hotelling, 1953; Anderson, 2003).

## Mathematical Formulation

The statistical basis of the Z-transform of the (partial) Pearson correlation coefficient was established by Hotelling (1953); its application to partial correlations and CI testing in Gaussian models is developed in classical multivariate analysis (Anderson, 2003). For sample partial correlation $r = \rho(X, Y \mid Z)$, the transform is

```{math}
Z(r) = \frac{1}{2} \ln\!\left(\frac{1+r}{1-r}\right)
```

equivalently $\operatorname{artanh}(r)$ (Hotelling, 1953). Under the null $X \perp Y \mid Z$ in a Gaussian model, the standardised statistic

```{math}
T = \sqrt{n - |Z| - 3} \cdot |Z(r)|
```

is asymptotically $N(0, 1)$ (Anderson, 2003), where $n$ is the sample size and $|Z|$ is the cardinality of the conditioning set.

## Assumptions

- **Multivariate normality.** Zero partial correlation is equivalent to conditional independence only under restrictive distributional assumptions; Baba et al. (2004) formalised the precise conditions, showing that violations of Gaussianity or linearity can invalidate correlation-based CI decisions even asymptotically.
- **Linearity.** The test targets linear (partial) dependence. Outside the Gaussian / elliptical family the equivalence between zero partial correlation and conditional independence breaks down (Baba et al., 2004).
- **Sample size and conditioning-set size.** The effective degrees of freedom are $n - |Z| - 3$; as $|Z|$ grows, variance increases and power drops (Anderson, 2003).

## Code Example

```python
import numpy as np
from citk.tests import FisherZ

# Generate data where X and Y are independent given Z
# X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = 2 * X + np.random.randn(n)
Y = 3 * Z + np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
fisher_z_test = FisherZ(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = fisher_z_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = fisher_z_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.FisherZ`.

## References

Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis* (3rd ed.). Wiley-Interscience.

Baba, K., Shibata, R., & Sibuya, M. (2004). Partial correlation and conditional correlation as measures of conditional independence. *Australian & New Zealand Journal of Statistics, 46*(4), 657-664.

Hotelling, H. (1953). New light on the correlation coefficient and its transforms. *Journal of the Royal Statistical Society: Series B (Methodological), 15*(2), 193-225.
