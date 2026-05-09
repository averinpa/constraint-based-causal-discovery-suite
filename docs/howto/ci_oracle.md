# Working with the CI oracle

`dagsampler` exposes the d-separation truth of the generated DAG
through two complementary surfaces. Pick the one that matches your
downstream consumer.

## Option 1 — `as_ci_oracle()` (since v0.2.0)

For use with constraint-based algorithms expecting a
`cbcd.CITest`-conforming object. The oracle is **lazy**: every
query is answered by an on-demand d-separation check on the
generated DAG, with no precomputation.

```python
from dagsampler import CausalDataGenerator
from cbcd import pc

gen = CausalDataGenerator(cfg)
result = gen.simulate()

oracle = gen.as_ci_oracle()
true_cpdag = pc(result["data"], ci_test=oracle, alpha=0.05)
```

The returned `DSeparationOracle` exposes:

- `n_vars: int` — number of variables in the DAG.
- `var_names: tuple[str, ...]` — the alphabetically-sorted column
  order matching `result["data"]`.
- `__call__(x: int, y: int, S: Sequence[int]) -> float` — returns
  `1.0` if the two indices are d-separated given `S`, `0.0`
  otherwise. (cbcd's PC tests `p > alpha`, so this convention
  recovers the oracle answer for any `alpha ∈ (0, 1)`.)
- `details(x, y, S)` — returns a small `_CITestResult` value
  object exposing `.p_value`.

This surface is the recommended one for cbcd interop — it scales
to any conditioning-set size without precomputing a table.

## Option 2 — precomputed d-separation table

For workflows that prefer a static record of CI relations (e.g.
auditing, reproducible benchmark fixtures), the simulator can emit
a precomputed list at simulation time:

```python
config = {
    "simulation_params": {
        "n_samples": 200,
        "seed": 42,
        "store_ci_oracle": True,
        "ci_oracle_max_cond_set": 2,  # enumerate |S| up to 2
    },
    "graph_params": {
        "type": "custom",
        "nodes": ["X", "Y", "Z"],
        "edges": [["X", "Z"], ["Y", "Z"]],
    },
}
result = CausalDataGenerator(config).simulate()
ci_oracle = result["ci_oracle"]
# list of dicts: [{"x": "X", "y": "Y", "S": [], "is_independent": True}, ...]
```

Each entry is a dictionary with string variable names and a
boolean `is_independent`. The list covers every unordered pair
`(X, Y)` and every conditioning set `S ⊆ V \ {X, Y}` with
`|S| ≤ ci_oracle_max_cond_set`.

The precomputed table is *not* a `cbcd.CITest`-conforming object;
if you need cbcd interop, use `gen.as_ci_oracle()` instead.

## Index conventions

`as_ci_oracle()` uses **integer indices** matching the
alphabetically-sorted column order of `result["data"]`. The
precomputed table uses **string names** matching the original
config. The two are interchangeable through the
`var_names` attribute of `DSeparationOracle`:

```python
oracle = gen.as_ci_oracle()
i = oracle.var_names.index("X")
j = oracle.var_names.index("Y")
oracle(i, j, [])
```
