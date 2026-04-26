# Chi-Squared Test

The Pearson chi-squared ($\chi^2$) test is the classical contingency-table CI test for categorical data. It compares observed and expected cell counts in a contingency table built from $X$, $Y$, and the (categorical) conditioning set $Z$, and is calibrated against an asymptotic $\chi^2$ distribution (Agresti, 2013).

**Intuition.** Conditional independence in a contingency table corresponds to multiplicative factorisation of conditional cell probabilities; the Pearson statistic measures the squared, expected-frequency-normalised discrepancy between observed counts and the counts implied by that factorisation (Agresti, 2013).

## Mathematical Formulation

The test compares observed frequencies $O_i$ with the frequencies $E_i$ that would be expected if $X \perp Y \mid Z$ held, summed over all cells of the (stratified) contingency table (Agresti, 2013):

```{math}
\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}
```

Under the null and standard regularity conditions, this statistic is asymptotically $\chi^2$-distributed with degrees of freedom

```{math}
df = (|X| - 1)(|Y| - 1) \prod_{z \in Z} |z|
```

where $|V|$ is the number of distinct categories of variable $V$ (Agresti, 2013). The Pearson $\chi^2$ statistic is asymptotically equivalent to the likelihood-ratio $G^2$ statistic (Agresti, 2013).

## Assumptions

- **Categorical data.** Both the variables of interest and the conditioning set must be categorical (Agresti, 2013).
- **Independent observations.** Standard multinomial sampling with independent draws is assumed (Agresti, 2013).
- **Adequate cell counts.** Cochran (1954) recommends that the test may be inappropriate if more than 20% of cells have an expected count below 5 or any cell has expected count below 1; these rules of thumb remain the standard finite-sample diagnostic for the test.
- **Power decay with conditioning-set size.** As $|Z|$ grows the number of contingency-table cells grows multiplicatively; sparse-table effects degrade power well before computational limits are reached (Agresti, 2013).

## Code Example

```python
import numpy as np
from citk.tests import ChiSq

# Generate discrete data representing a collider: X -> Y <- Z
n = 500
X = np.random.randint(0, 2, size=n)
Z = np.random.randint(0, 2, size=n)
Y = (X + Z + np.random.randint(0, 2, size=n)) % 2
data = np.vstack([X, Y, Z]).T

# Initialize the test
chisq_test = ChiSq(data)

# Test for unconditional independence (X and Z are independent)
p_value_unconditional = chisq_test(0, 2)
print(f"P-value for X _||_ Z: {p_value_unconditional:.4f}")

# Test for conditional dependence on the collider Y
p_value_conditional = chisq_test(0, 2, [1])
print(f"P-value for X _||_ Z | Y: {p_value_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.ChiSq`.

## References

Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.

Cochran, W. G. (1954). Some methods for strengthening the common $\chi^2$ tests. *Biometrics, 10*(4), 417-451.
