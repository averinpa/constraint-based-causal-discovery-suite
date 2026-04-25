# Hartemink Discretisation + Chi-Squared (HarteminkChiSq) Test

HarteminkChiSq is an adapter that applies an information-preserving discretisation due to Hartemink (2001) to continuous variables before running the standard :doc:`/tests/chi_sq_test`. Unlike equal-frequency binning (see :doc:`/tests/disc_chisq_test`), Hartemink's procedure iteratively merges initial bins in a way that minimises the loss of pairwise mutual information across columns, producing discretisations that preserve the dependence structure more faithfully.

The discretisation step is delegated to the R `bnlearn` package (Scutari, 2010) via `rpy2`.

## Mathematical Formulation

The procedure starts from a fine equal-frequency discretisation with $b_0$ bins per column (default $b_0 = 10$). It then performs a series of greedy bin-merging steps in each column until each column has the target number of bins $b$ (default $b = 4$). At each step, the merge chosen for column $j$ is the one that minimises the pairwise mutual information loss

```{math}
\Delta = \sum_{k \neq j} \bigl[ I(\widetilde{V}_j; V_k) - I(\widetilde{V}_j^{\,\text{merged}}; V_k) \bigr]
```

across the other columns $k$. The merge is applied if its loss is the smallest available; ties are broken by an upstream rule. After the iterative procedure, all columns are integer-coded and passed to the standard chi-squared test:

```{math}
\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}
```

where $O_i$ and $E_i$ are the observed and expected cell counts in the contingency table on the discretised data (see :doc:`/tests/chi_sq_test`).

## Assumptions

- **R + bnlearn available**: This wrapper requires `rpy2` and the R `bnlearn` package (from CRAN).
- **Discretisation preserves dependence**: The Hartemink procedure is targeted at this, but it is not infallible — sample size still matters, and very subtle dependences may be lost.
- **Adequate cell counts**: Same Cochran rule of thumb as Chi-Squared (Cochran, 1954): the test may be unreliable if more than 20% of cells have an expected frequency below 5.

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

Cochran, W. G. (1954). Some methods for strengthening the common $\chi^2$ tests. *Biometrics, 10*(4), 417-451.

Hartemink, A. J. (2001). *Principled computational methods for the validation and discovery of genetic regulatory networks*. PhD thesis, Massachusetts Institute of Technology.

Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. *Journal of Statistical Software, 35*(3), 1-22.
