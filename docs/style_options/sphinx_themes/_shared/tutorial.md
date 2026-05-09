# Tutorial: a first PC run

The PC algorithm recovers a **CPDAG** — the Markov equivalence class
of a directed acyclic graph — from observational data. Constraint-
based discovery proceeds in two phases: skeleton recovery via
repeated conditional-independence tests, followed by orientation of
v-structures and propagation through Meek's rules.

The motivating idea is simple. If two variables `X` and `Y` are
directly causally linked, no conditioning set renders them
independent; if they're linked only through a third variable `Z`,
conditioning on `Z` should break the dependence. PC turns this
observation into an algorithm.

## A 4-node diamond

We'll generate data from a diamond DAG `A → B, A → C, B → D, C → D`
and watch PC recover its CPDAG.

```python
from dagsampler import CausalDataGenerator
from cbcd import pc

cfg = {
    "simulation_params": {"n_samples": 3000, "seed_structure": 1,
                          "seed_data": 2, "binary_proportion": 0.0},
    "graph_params": {"type": "custom",
                     "nodes": ["A", "B", "C", "D"],
                     "edges": [["A", "B"], ["A", "C"],
                               ["B", "D"], ["C", "D"]]},
}
result = CausalDataGenerator(cfg).simulate()
cpdag = pc(result["data"], alpha=0.05)
```

`cpdag.endpoints` is an `int8` matrix in cbcd's mark convention:
`0` for no edge, `1` for TAIL, `2` for ARROW. Reading the row for
`D` reveals the v-structure — both `B→D` and `C→D` arrive as ARROW
— exactly what the collider rule oriented.

## Choosing alpha

`alpha` is the significance level for the CI tests. The standard
default `0.05` works well for moderate sample sizes. With small
`n` you'll see *spurious* edges; with very large `n` the type-I
rate dominates and you may want to tighten `alpha` to `0.01`.

## What's next

- Different CI test → see [How-to: choosing a CI test](howto.md).
- Algorithmic theory → see [Explanation: PC theory](explanation.md).
