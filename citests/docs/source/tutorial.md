# Tutorial: a first conditional-independence call

This tutorial walks through one full conditional-independence
calculation using `citests`, then plugs the same test into a
constraint-based discovery algorithm via the cross-package
`cbcd.CITest` Protocol. The reader is assumed to know what
conditional independence is and what role it plays in causal
discovery; for a primer see
[Explanation: what is conditional independence](explanation/what_is_ci.md).

## A direct CI call

Every test in `citests` exposes the same three-member surface:
`n_vars: int`, `__call__(X, Y, S)`, and `details(X, Y, S)`. Construct
the test once with the data array, then query as many CI relations
as you need:

```python
import numpy as np
from citests.tests.partial_correlation_tests import FisherZ

# X and Y are independent given Z under the chain X → Z → Y.
n = 500
X = np.random.randn(n)
Z = 2 * X + np.random.randn(n)
Y = 3 * Z + np.random.randn(n)
data = np.vstack([X, Y, Z]).T  # columns: X, Y, Z

test = FisherZ(data)
p = test(0, 1, [2])             # X ⫫ Y | Z
print(f"p(X ⫫ Y | Z) = {p:.4f}")  # large — fail to reject independence

p_marg = test(0, 1, [])         # marginally not independent
print(f"p(X ⫫ Y)     = {p_marg:.4f}")  # small — reject independence
```

The conditioning set is passed positionally as a list of integer
indices into the data columns. Returned values are p-values; the
caller is responsible for thresholding (typically against an $\alpha$
on the order of $0.05$).

For richer diagnostics — the test statistic, degrees of freedom,
effective sample size — call `test.details(X, Y, S)`, which returns
a `CITKResult` value object.

## Plugging citests into cbcd

Because every `citests` test satisfies the structural `cbcd.CITest`
Protocol, any constraint-based algorithm in the sister package
[`cbcd`](https://github.com/averinpa/cbcd) accepts a citests test
directly:

```python
from cbcd import pc
from citests.tests.partial_correlation_tests import FisherZ

cpdag = pc(data, ci_test=FisherZ(data), alpha=0.05)
```

No adapter or wrapping is required. The Protocol contract is
verified at runtime — `isinstance(FisherZ(data), cbcd.citest.CITest)`
returns `True` — but the duck-typed dispatch needs no inheritance,
so neither package imports the other.

```{seealso}
The full four-package end-to-end story (data simulation through
metric scoring) is in the suite tutorial at
`suite/docs/tutorial.md`.
```

## Choosing a different test

Switch to any other test by changing the import:

```python
from citests.tests.partial_correlation_tests import Spearman    # rank-based
from citests.tests.kernel_tests             import KCI          # non-parametric
from citests.tests.nearest_neighbor_tests   import CMIknn       # k-NN CMI
from citests.tests.regression_tests         import RegressionCI # mixed-type
```

A practical mapping from data type and modelling assumption to test
choice is given in
[How-to: choosing a CI test](howto/choosing_a_test.md). The deeper
theory — what makes each family of tests valid under which
assumptions — is in
[Explanation: taxonomy of CI tests](explanation/taxonomy_of_tests.md).

## What is next

- **Per-test deep dives** — every test has a dedicated page with
  Mathematical Formulation, Assumptions, and Code Example sections.
  Start at the [Tests index](tests/index.rst).
- **The Protocol contract** —
  [Explanation: API stability](explanation/api_stability.md).
- **Custom tests** — implement the `CITKTest` interface and the
  test slots into both the `citests` cache machinery and `cbcd`'s
  algorithm dispatch.
