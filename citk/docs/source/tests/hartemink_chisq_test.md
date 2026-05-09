# Hartemink Discretisation + Chi-Squared (HarteminkChiSq) Test

HarteminkChiSq is an adapter that applies the information-preserving discretisation of Hartemink (2001) to continuous variables before running the standard :doc:`/tests/chi_sq_test` (Agresti, 2013). Unlike equal-frequency binning (see :doc:`/tests/disc_chisq_test`), Hartemink's procedure iteratively merges initial bins so as to minimise the loss of pairwise mutual information across columns, producing discretisations that preserve the dependency and conditional independence structure relevant for Bayesian network learning rather than merely approximating marginal distributions (Hartemink, 2001).

The discretisation step is delegated to the R `bnlearn` package via `rpy2`.

**Intuition.** Most discretisation schemes are local — they bin one variable at a time, ignoring its relationships to the others (Hartemink, 2001). Hartemink (2001) instead frames discretisation as a global optimisation: at each step, merge the pair of adjacent bins (in any column) that loses the smallest amount of pairwise mutual information across the rest of the data. The result tends to keep cells where dependencies live and merges cells that contribute little signal.

## Mathematical Formulation

The procedure starts from a fine equal-frequency discretisation with $b_0$ bins per column (default $b_0 = 10$), as recommended by Hartemink (2001). It then performs a series of greedy bin-merging steps in each column until each column has the target number of bins $b$ (default $b = 4$). At each step, the merge chosen for column $j$ is the one that minimises the pairwise mutual information loss

```{math}
\Delta = \sum_{k \neq j} \bigl[ I(\widetilde{V}_j; V_k) - I(\widetilde{V}_j^{\,\text{merged}}; V_k) \bigr]
```

across the other columns $k$ (Hartemink, 2001). Mutual information is computed from the empirical joint distributions over current bin boundaries (Cover & Thomas, 2006). The merge is applied if its loss is the smallest available; ties are broken by an upstream rule. After the iterative procedure, all columns are integer-coded and passed to the standard chi-squared test (Agresti, 2013):

```{math}
\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}
```

where $O_i$ and $E_i$ are the observed and expected cell counts in the contingency table on the discretised data; see :doc:`/tests/chi_sq_test`.

## Assumptions

- **R + bnlearn available.** This wrapper requires `rpy2` and the R `bnlearn` package (Hartemink, 2001).
- **Discretisation preserves dependence.** Hartemink's procedure is targeted at this objective, but it is not infallible — sample size still matters, and very subtle dependences may be lost (Hartemink, 2001).
- **Adequate cell counts.** Same Cochran (1954) rule of thumb as :doc:`/tests/chi_sq_test`: the test may be unreliable if more than 20% of cells have an expected frequency below 5 (Agresti, 2013).
- **Independent observations.** Standard multinomial sampling (Agresti, 2013).
- **Approximate CI procedure.** As with all discretisation-based adapters, the test targets independence between binned representations, not the underlying continuous CI null (Agresti, 2013).
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

## Code Example

```python
import numpy as np
from citk.tests import HarteminkChiSq

# Continuous chain: X -> Z -> Y
n = 600
X = np.random.randn(n)
Z = 0.8 * X + 0.5 * np.random.randn(n)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test with 4 target bins per column (default)
hartemink_test = HarteminkChiSq(data, breaks=4, ibreaks=10)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = hartemink_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = hartemink_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.adapter_tests.HarteminkChiSq`.

## References

Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.

Cochran, W. G. (1954). Some methods for strengthening the common $\chi^2$ tests. *Biometrics, 10*(4), 417-451.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Hartemink, A. J. (2001). *Principled computational methods for the validation and discovery of genetic regulatory networks*. PhD thesis, Massachusetts Institute of Technology.
