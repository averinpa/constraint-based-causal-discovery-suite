# Regression-Based Conditional Independence (RegressionCI) Test

RegressionCI is a likelihood-ratio CI test that fits nested parametric regressions of one variable on the conditioning set, and tests whether adding the other variable yields a significantly better fit. The link function (linear / logistic / multinomial) is selected automatically from the response variable's declared type (Tsagris et al., 2018).

**Intuition.** Under correct model specification, $X \perp Y \mid Z$ is equivalent to the regression coefficient on $X$ being zero in a model for $Y$ given $(Z, X)$ (Tsagris et al., 2018; Kutner et al., 2005). Wilks's theorem then gives the asymptotic $\chi^2$ distribution of the log-likelihood-ratio statistic (Wilks, 1938).

## Mathematical Formulation

For a hypothesis $X \perp Y \mid Z$, RegressionCI compares two nested regression models for the response $Y$:

```{math}
\mathcal{M}_0:\; Y = f(Z) + \epsilon_0, \qquad \mathcal{M}_1:\; Y = g(Z, X) + \epsilon_1
```

Continuous responses use a Gaussian linear model (Kutner et al., 2005); binary or multinomial responses use logistic regression (Hosmer et al., 2013). Conditional independence is tested by the likelihood-ratio statistic comparing the two fits (Tsagris et al., 2018):

```{math}
\Lambda = -2 \left[ \ell(\mathcal{M}_0) - \ell(\mathcal{M}_1) \right]
```

where $\ell(\cdot)$ is the maximised log-likelihood. Under the null and standard regularity conditions, $\Lambda$ is asymptotically $\chi^2$-distributed with degrees of freedom equal to the number of parameters added in $\mathcal{M}_1$ over $\mathcal{M}_0$ (Wilks, 1938).

## Assumptions

- **Correctly specified model class.** The chosen link (linear, logistic) and functional form must capture the true conditional mean of $Y$ given $(Z, X)$ (Kutner et al., 2005; Hosmer et al., 2013); under misspecification, the regression null and the CI null diverge (Tsagris et al., 2018).
- **Mixed-data support via type tagging.** Variables are tagged continuous or discrete so the appropriate link is chosen per regression (Tsagris et al., 2018).
- **Asymptotic regime.** P-values rely on the chi-square approximation justified by Wilks's theorem (Wilks, 1938); finite-sample reliability degrades when stratum-level sample sizes shrink (Tsagris et al., 2018).
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

## Code Example

```python
import numpy as np
from citk.tests import RegressionCI

# Continuous chain: X -> Z -> Y
n = 400
X = np.random.randn(n)
Z = 0.8 * X + 0.5 * np.random.randn(n)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test (continuous variables by default)
regci_test = RegressionCI(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = regci_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = regci_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.regression_tests.RegressionCI`.

## References

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). *Applied Linear Statistical Models* (5th ed.). McGraw-Hill Irwin.

Tsagris, M., Borboudakis, G., Lagani, V., & Tsamardinos, I. (2018). Constraint-based causal discovery with mixed data. *International Journal of Data Science and Analytics, 6*(1), 19-30.

Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60-62.
