# Working with cbcd graphs

`cbcd` graph types — `cbcd.DAG`, `cbcd.CPDAG`, `cbcd.PAG`,
`cbcd.MAG` — satisfy the structural `bnm.GraphLike` Protocol
without modification. Every `bnm` metric and visualiser accepts
them directly:

```python
from cbcd import pc
import bnm

cpdag = pc(data, alpha=0.05)
bnm.shd(cpdag, true_cpdag)
bnm.plot_graph(cpdag)
```

No conversion or wrapping is required: `cbcd.CPDAG` exposes
`n_vars: int`, `endpoints: ndarray[int8]`, and
`var_names: tuple[str, ...]` — the three members `bnm.GraphLike`
requires.

```{note}
This page is currently a stub. The full Protocol contract and a
worked PC → bnm round-trip example will land in v0.x.x.
```
