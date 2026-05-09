# Run PC

PC is the workhorse for constraint-based causal discovery. Give it
data, an `alpha`, and a CI test — out comes a CPDAG.

```python
from cbcd import pc

cpdag = pc(my_data, alpha=0.05)
```

That's it!

!!! tip
    Not sure what `alpha` to pick? Start with `0.05`. We'll cover
    the tradeoffs in [How-to: tuning alpha](../howto/tuning_alpha.md).

## What just happened?

PC builds the graph in two phases:

1. **Skeleton.** Test each pair of variables for independence,
   conditioning on growing subsets of the others. Remove edges where
   independence is found.
2. **Orientation.** Identify v-structures (`X → Z ← Y` patterns) and
   propagate through Meek's rules to direct as many edges as the
   data allows.

The result is a CPDAG: directed edges where the data is decisive,
undirected edges where multiple equivalent DAGs are compatible.

!!! info "What's a CPDAG?"
    A CPDAG is the **Markov equivalence class** of a DAG. Two DAGs
    are equivalent if they encode the same conditional-independence
    relations. The CPDAG captures what's identifiable from data
    alone — the orientations the algorithm is *sure* about, plus
    the ones it has to leave undirected.

## Reading the output

```python
print(cpdag.endpoints)
# array([[0, 1, 1, 0],
#        [2, 0, 0, 2],
#        [2, 0, 0, 2],
#        [0, 1, 1, 0]], dtype=int8)
```

cbcd uses a small int8 endpoint matrix:

- `0` — no edge
- `1` — TAIL (the `o` end of `o→`)
- `2` — ARROW (the `→` end)

So `endpoints[i, j] == 2 and endpoints[j, i] == 1` means **`i → j`**,
and `endpoints[i, j] == endpoints[j, i] == 1` means **`i — j`** (an
undirected edge in the CPDAG).

!!! note
    Same endpoint convention is used by [`bnm`](https://github.com/...),
    so `bnm.shd(cpdag1, cpdag2)` works on cbcd outputs with no
    conversion.

## Where next?

- 🧪 **Different data type?** [How-to: pick a CI test](../howto/choosing_a_ci_test.md)
- 🔬 **Want the full algorithm?** [Explanation: PC theory](../explanation/pc_theory.md)
- 🌐 **Time series?** [Tutorial: pcmci](pcmci.md)

!!! warning
    PC assumes **causal sufficiency** — no unobserved common causes.
    If that's a stretch for your data, use [`fci`](fci.md) instead.
