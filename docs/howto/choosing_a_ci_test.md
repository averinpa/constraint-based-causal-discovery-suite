# Choosing a conditional-independence test

The `ci_test=` argument of `cbcd.pc`, `cbcd.fci`, and their
variants accepts any object satisfying the structural `cbcd.CITest`
Protocol. The bundled tests are listed below; sister packages
(`citk`) supply additional non-parametric and ML-based options.

## When to use which

Each test makes assumptions about the data-generating distribution.
The choice should be driven by the dominant assumption a researcher
is willing to make about the mechanism between variables.

| Test | Bundled in cbcd | Assumes | Use when |
|---|:---:|---|---|
| Fisher–Z | yes | linear-Gaussian SCM | continuous data, mostly-linear mechanisms |
| χ² | yes | discrete categorical | discrete-only data |
| G² | yes | discrete categorical | discrete-only data, prefer log-likelihood ratio |
| KCI | planned | none beyond i.i.d. | non-linear continuous data, computational budget allows |
| `citk.RegressionCI` | via `citk` | flexible | mixed-type data |

A more complete catalogue of the test landscape appears in
[Explanation: CI test taxonomy](../explanation/ci_test_taxonomy.md).

```{note}
This page is currently a stub. A worked example covering each test
on a benchmark fixture will land in v0.x.x.
```
