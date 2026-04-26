# G-Squared Test

The G-squared ($G^2$) test, also known as the likelihood-ratio test for contingency tables, is the entropy-based companion of the Pearson $\chi^2$ test for categorical data (Agresti, 2013). Its theoretical foundation is Wilks's theorem, which gives the asymptotic $\chi^2$ distribution of the log-likelihood-ratio statistic (Wilks, 1938).

**Intuition.** $G^2$ is twice the Kullback--Leibler divergence between the empirical joint distribution and the conditional-independence factorisation, equivalently $2n$ times the empirical conditional mutual information between $X$ and $Y$ given $Z$ (Cover & Thomas, 2006). Conditional independence corresponds to zero conditional mutual information, so a small $G^2$ supports the null.

## Mathematical Formulation

For a hypothesis $X \perp Y \mid Z$, the statistic stratifies by each value of $Z$ and sums the per-stratum log-likelihood-ratio contributions (Agresti, 2013):

```{math}
G^2 = 2 \sum_{i} O_i \ln\!\left(\frac{O_i}{E_i}\right)
```

where the sum is over all non-empty cells $i$ and $E_i$ are the expected counts under conditional independence (Agresti, 2013). Under the null and standard regularity conditions, $G^2$ is asymptotically $\chi^2$-distributed by Wilks's theorem (Wilks, 1938) with degrees of freedom

```{math}
df = (|X| - 1)(|Y| - 1) \prod_{z \in Z} |z|
```

(Agresti, 2013). The information-theoretic identity $G^2 = 2n\, \widehat{I}(X; Y \mid Z)$ links the test to conditional mutual information (Cover & Thomas, 2006).

## Assumptions

- **Categorical data.** Variables and the conditioning set must be discrete (Agresti, 2013).
- **Independent observations.** Standard multinomial sampling assumptions apply (Agresti, 2013).
- **Adequate sample size.** $G^2$ is asymptotic; the same Cochran-style cell-count guidance as for Pearson $\chi^2$ applies (Cochran, 1954; Agresti, 2013).
- **Power decay with conditioning-set size.** Stratification by $Z$ multiplies the number of cells, leading to sparse-table effects analogous to those of Pearson $\chi^2$ (Agresti, 2013).

## Code Example

```python
import numpy as np
from citk.tests import GSq

# Generate discrete data for a chain: X -> Z -> Y
n = 500
X = np.random.randint(0, 3, size=n)
Z = (X + np.random.randint(0, 2, size=n)) % 3
Y = (Z + np.random.randint(0, 2, size=n)) % 3
data = np.vstack([X, Y, Z]).T

# Initialize the test
g_sq_test = GSq(data)

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = g_sq_test(0, 1)
print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = g_sq_test(0, 1, [2])
print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.GSq`.

## References

Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.

Cochran, W. G. (1954). Some methods for strengthening the common $\chi^2$ tests. *Biometrics, 10*(4), 417-451.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60-62.
