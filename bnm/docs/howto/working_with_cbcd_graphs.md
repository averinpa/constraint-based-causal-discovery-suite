# Working with cbcd graphs

The `cbcd` graph types — `cbcd.DAG`, `cbcd.CPDAG`, `cbcd.PAG`,
`cbcd.MAG` — satisfy the structural `bnm.GraphLike` Protocol
without modification, so every `bnm` metric and visualiser accepts
them directly. No conversion is required: `cbcd.CPDAG` already
exposes `n_vars`, `endpoints`, and `var_names`.

## A worked PC → bnm round trip

The standard pattern is to run a constraint-based search in
`cbcd`, then score the recovered graph against a reference using
`bnm`. The reference graph below is itself the output of `pc()`
under a perfect d-separation oracle, which by construction returns
the true CPDAG of the generating DAG:

```python
from dagsampler import CausalDataGenerator
from cbcd import pc
import bnm

cfg = {
    "simulation_params": {
        "n_samples": 3000, "seed_structure": 1, "seed_data": 2,
        "binary_proportion": 0.0,
    },
    "graph_params": {
        "type": "custom",
        "nodes": ["A", "B", "C", "D"],
        "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"]],
    },
}
gen = CausalDataGenerator(cfg)
result = gen.simulate()

true_cpdag = pc(result["data"], ci_test=gen.as_ci_oracle(), alpha=0.05)
recovered  = pc(result["data"], ci_test="fisherz",         alpha=0.05)

bnm.shd(true_cpdag, recovered)        # 2
bnm.f1(true_cpdag, recovered)         # 0.5
bnm.plot_side_by_side(true_cpdag, recovered,
                      name1="true", name2="recovered",
                      direction="LR", save="comparison.svg")
```

Both `true_cpdag` and `recovered` are `cbcd.CPDAG` instances;
`bnm` does not import `cbcd`. The interop is mediated entirely by
the `GraphLike` Protocol, which `cbcd.CPDAG` satisfies because it
exposes the three required attributes:

```python
true_cpdag.n_vars        # int
true_cpdag.endpoints     # np.ndarray[int8], shape (n_vars, n_vars)
true_cpdag.var_names     # tuple[str, ...] | None
isinstance(true_cpdag, bnm.GraphLike)   # True
```

`isinstance` works because `GraphLike` is a `@runtime_checkable`
Protocol; conformance is structural (duck-typed) rather than
nominal, so no inheritance from a shared base class is required.
