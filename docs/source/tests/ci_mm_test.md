# Symmetric Mixed-Model Conditional Independence (CiMM) Test

CiMM is the symmetric, regression-based CI test for mixed continuous--categorical data introduced by Tsagris et al. (2018) for constraint-based causal discovery with mixed variables. It runs two complementary likelihood-ratio regressions in opposite directions and combines them, addressing the fact that likelihood-ratio tests are not generally symmetric in the roles of $X$ and $Y$ (Tsagris et al., 2018).

**Intuition.** Likelihood-ratio CI tests of $X \perp Y \mid Z$ via "regress $Y$ on $(Z, X)$" need not match those via "regress $X$ on $(Z, Y)$" once links and error families differ across variable types; symmetrising over the two directions gives a single CI verdict that does not depend on the arbitrary choice of response (Tsagris et al., 2018).

## Mathematical Formulation

For each pair $(X, Y)$ and conditioning set $Z$, CiMM fits two pairs of nested regressions, one with $X$ as the response and one with $Y$ as the response (Tsagris et al., 2018):

```{math}
\Lambda_{X \mid Y, Z} = -2 \left[ \ell(X \sim Z) - \ell(X \sim Y, Z) \right]
```

```{math}
\Lambda_{Y \mid X, Z} = -2 \left[ \ell(Y \sim Z) - \ell(Y \sim X, Z) \right]
```

The link for each regression is chosen by the response type: linear regression for continuous responses (Kutner et al., 2005), logistic or multinomial regression for binary or categorical responses (Hosmer et al., 2013); ordinal responses use ordered-logit links (Tsagris et al., 2018). Each direction's statistic is asymptotically $\chi^2$ under the null (Wilks, 1938). The two p-values are combined into a single symmetric verdict using the procedure of Tsagris et al. (2018).

## Assumptions

- **Correctly specified link per variable type.** The chosen GLM family must be a reasonable approximation for each variable's conditional mean (Kutner et al., 2005; Hosmer et al., 2013); under misspecification the asymptotic calibration can fail (Tsagris et al., 2018).
- **R + MXM available.** The reference implementation lives in the R `MXM` package and is exposed via `rpy2` (Tsagris et al., 2018).
- **Per-variable type declarations.** A `data_type` array specifies which columns are continuous and which are discrete so the right link is chosen per regression (Tsagris et al., 2018).
- **Asymptotic regime.** Calibration uses Wilks's theorem (Wilks, 1938); small-sample reliability depends on category counts (Tsagris et al., 2018).

## Code Example

```python
import numpy as np
from citk.tests import CiMM

# Mixed chain: continuous X -> binary Z -> continuous Y
n = 400
X = np.random.randn(n)
Z_logits = 1.5 * X
Z = (np.random.rand(n) < 1 / (1 + np.exp(-Z_logits))).astype(int)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Declare per-variable types: 0 = continuous, 1 = discrete
data_type = np.array([[0, 0, 1]])

# Initialize the test
ci_mm_test = CiMM(data, data_type=data_type)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = ci_mm_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = ci_mm_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.regression_tests.CiMM`.

## References

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). *Applied Linear Statistical Models* (5th ed.). McGraw-Hill Irwin.

Tsagris, M., Borboudakis, G., Lagani, V., & Tsamardinos, I. (2018). Constraint-based causal discovery with mixed data. *International Journal of Data Science and Analytics, 6*(1), 19-30.

Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60-62.
