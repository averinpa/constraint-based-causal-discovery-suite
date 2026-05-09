# Tutorial: a first PC run

The PC algorithm recovers a **CPDAG** — the Markov equivalence class
of a directed acyclic graph — from observational data. Constraint-
based discovery proceeds in two phases: skeleton recovery via
repeated conditional-independence (CI) tests, followed by orientation
of v-structures and propagation through Meek's rules.

The motivating idea is simple. If two variables `X` and `Y` are
directly causally linked, no conditioning set can render them
independent; if they're linked only through a third variable `Z`,
then conditioning on `Z` should break the dependence. PC turns this
observation into an algorithm: start with a fully connected graph,
test for conditional independence at growing conditioning-set sizes,
and remove edges where independence is found.

## A 4-node diamond

We'll generate data from a diamond DAG `A → B, A → C, B → D, C → D`
and watch PC recover its CPDAG. Under faithfulness, the only
v-structure is the collider into `D`; the upper edges (`A→B`, `A→C`)
share their equivalence class with their reversals, and PC will
return them undirected.

```python
from dagsampler import CausalDataGenerator
from cbcd import pc

cfg = {
    "simulation_params": {"n_samples": 3000, "seed_structure": 1, "seed_data": 2,
                          "binary_proportion": 0.0},
    "graph_params": {"type": "custom",
                     "nodes": ["A", "B", "C", "D"],
                     "edges": [["A", "B"], ["A", "C"],
                               ["B", "D"], ["C", "D"]]},
}
result = CausalDataGenerator(cfg).simulate()
cpdag = pc(result["data"], alpha=0.05)
```

`cpdag.endpoints` is a square `int8` matrix in cbcd's mark
convention: `0` for no edge, `1` for TAIL, `2` for ARROW. Reading
the row for `D` reveals the v-structure — both `B→D` and `C→D`
arrive as ARROW, with `D`'s side as TAIL — exactly what the
collider rule oriented. Reading the row for `A` reveals undirected
edges to `B` and `C`: TAIL on both ends, the visual signature of
"the data isn't decisive about this orientation".

## Choosing alpha

`alpha` is the significance level for the CI tests. The standard
default `0.05` works well for moderate sample sizes. With small
`n` you'll see *spurious* edges (false positives) because individual
CI tests don't have enough power to detect independence. With very
large `n` the type-I rate dominates and you may want to tighten
`alpha` to `0.01` or below.

The empirical recovery is a function of three things together: data
quality, the choice of CI test (`"fisherz"` assumes linear-Gaussian
mechanisms), and `alpha`. The how-to guide *Choosing a CI test*
walks through the tradeoff in detail.

## What's next

- A different CI test — kernel-based for non-linear data, χ²/G² for
  discrete — see [How-to: choosing a CI test](../howto/choosing_a_ci_test.md).
- The full algorithmic theory, with the reasoning behind Meek's
  rules, lives in [Explanation: PC theory](../explanation/pc_theory.md).
- For time-series data, the analogous algorithm is `pcmci`; see
  [Tutorial: pcmci](pcmci.md).
