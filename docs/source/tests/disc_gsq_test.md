# Discretised G-Squared (DiscGSq) Test

DiscGSq is an adapter that applies equal-frequency discretisation to continuous variables before running the standard :doc:`/tests/g_sq_test`. It is the likelihood-ratio counterpart of :doc:`/tests/disc_chisq_test`, and is often preferred for moderate sample sizes where the G-statistic's likelihood-ratio interpretation gives more reliable behaviour.

## Mathematical Formulation

For each continuous column $V$ of the input data, the adapter produces a discrete version

```{math}
\widetilde{V}_i = \mathrm{quantile\_bin}(V_i; b)
```

where `quantile_bin` assigns each observation to one of $b$ equal-frequency bins. Columns that are already categorical are left unchanged. The G-statistic is then computed on the binned data:

```{math}
G = 2 \sum_{i} O_i \ln\!\left(\frac{O_i}{E_i}\right)
```

where $O_i$ and $E_i$ are the observed and expected cell counts in the binned contingency table under the null $X \perp Y \mid Z$. Under the null, $G$ is asymptotically $\chi^2$-distributed by Wilks's theorem (Wilks, 1938) with degrees of freedom matching the binned cardinalities (see :doc:`/tests/g_sq_test`).

## Assumptions

- **Discretisation preserves dependence**: The number of bins must be coarse enough to keep cells populated and fine enough to preserve the underlying dependence.
- **Independent observations**: Same as G-Squared.
- **Adequate cell counts**: G-Squared is somewhat more robust than Chi-Squared at small expected frequencies, but the same rule of thumb (no more than 20% of cells below an expected count of 5) still applies (Sokal & Rohlf, 1981).

## Code Example

```python
import numpy as np
from citk.tests import DiscGSq

# Continuous chain: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = 0.8 * X + 0.5 * np.random.randn(n)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test with 4 equal-frequency bins per continuous variable
disc_gsq_test = DiscGSq(data, n_bins=4)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = disc_gsq_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = disc_gsq_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.adapter_tests.DiscGSq`.

## References

Sokal, R. R., & Rohlf, F. J. (1981). *Biometry: The Principles and Practice of Statistics in Biological Research*. W. H. Freeman.

Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60-62.
