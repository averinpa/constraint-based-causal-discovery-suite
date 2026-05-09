# Compare two graphs

The fastest path to a full comparison is the `bnm.compare` façade,
which runs every comparative metric and packages the results into
a `Comparison` value object:

```python
import bnm
result = bnm.compare(g1, g2)
result.shd            # int
result.f1             # float
result.additions      # int
bnm.to_dataframe([result])  # tidy long-form pandas DataFrame
```

For interactive inspection of a single metric, the standalone
functions `bnm.shd`, `bnm.f1`, `bnm.precision`, `bnm.recall`,
`bnm.hd`, `bnm.count_additions`, `bnm.count_deletions`, and
`bnm.count_reversals` are equivalent and faster (no DataFrame
overhead).

## Variable-name alignment

`bnm` aligns variables by name when both inputs expose
`var_names`; otherwise it aligns positionally. A
`BNMDataError` is raised if the two graphs have different
variable counts or non-matching name sets, since silent
positional alignment of inconsistently-named graphs is the most
common source of subtle metric bugs.

```{note}
This page is currently a stub. A worked example with several
metrics on the canonical diamond fixture will land in v0.x.x.
```
