# Generalised Kernel Covariance Measure (GKCM) Test

GKCM (Bergen, Sejdinovic & Didelez, CLeaR 2025) is a kernelised generalisation of the Generalised Covariance Measure. Where GCM tests whether the scalar residuals $X - \hat{E}[X\mid Z]$ and $Y - \hat{E}[Y\mid Z]$ are uncorrelated, GKCM maps the residuals into a reproducing-kernel Hilbert space and tests whether their kernel cross-covariance operator is zero, so it detects dependence in the full conditional distribution rather than only in the conditional mean (Bergen et al., 2025).

**Intuition.** Regress $X$ and $Y$ on $Z$ with a flexible nuisance, embed the residuals with characteristic kernels, and test the Hilbert–Schmidt norm of their cross-covariance. Under a characteristic kernel the statistic is zero exactly when the residual embeddings are independent, widening GCM's mean-zero detection class toward full conditional independence (Bergen et al., 2025).

## Assumptions

- **Continuous data.** GKCM is formulated for continuous variables (Bergen et al., 2025).
- **R + `comets` available.** This wrapper dispatches to the R `comets::kgcm` implementation and requires `rpy2` and the R `comets` package (`install.packages("comets")`).
- **Nuisance rate.** Like GCM, validity rests on the nuisance regressions estimating the conditional means at a sufficient rate; the regression method is the R `reg_YonZ`/`reg_XonZ` argument (default `"rf"`), set through the `reg` keyword.
- **Quadratic cost.** The kernel statistic is $O(n^2)$ in the sample size, so it does not scale to large $n$ the way the random-feature relaxations (`rcit`, `rcot`) do.

## Code Example

```python
import numpy as np
from citests.tests import GKCM

rng = np.random.default_rng(0)
n = 300
z = rng.normal(size=n)
x = z + 0.5 * rng.normal(size=n)
y = z + 0.5 * rng.normal(size=n)          # x _||_ y | z
data = np.column_stack([x, y, z])

p = GKCM(data)(0, 1, [2])                 # test X _||_ Y | Z
print(p)

# select the nuisance regression (R comets method name), default "rf":
p_lm = GKCM(data, reg="lasso")(0, 1, [2])
```

## References

- Bergen, L., Sejdinovic, D., & Didelez, V. (2025). The Generalised Kernel Covariance Measure. *Proceedings of the Conference on Causal Learning and Reasoning (CLeaR)*.
- Shah, R. D., & Peters, J. (2020). The Hardness of Conditional Independence Testing and the Generalised Covariance Measure. *Annals of Statistics*.
