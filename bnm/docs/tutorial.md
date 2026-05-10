# Tutorial: a first comparison

This tutorial walks through one full structural comparison between
two graphs using `bnm`, from constructing graph inputs through
metric computation to visualisation. The reader is assumed to be
familiar with directed acyclic graphs, conditional independence,
and the standard Markov equivalence vocabulary; for the
mathematical background see
[Explanation: comparative metrics](explanation/metrics_theory.md).

## Setting

We will compare a recovered CPDAG against the *true* CPDAG of a
known generating DAG, using the suite-wide pattern from the cbcd
tutorial: PC under a perfect d-separation oracle returns the true
CPDAG by construction (the gold standard), while PC under a
finite-sample CI test returns the empirical recovery (the object
being scored).

```python
from dagsampler import CausalDataGenerator
from cbcd import pc
import bnm

cfg = {
    "simulation_params": {
        "n_samples": 3000,
        "seed_structure": 1,
        "seed_data": 2,
        "binary_proportion": 0.0,
    },
    "graph_params": {
        "type": "custom",
        "nodes": ["A", "B", "C"],
        "edges": [["A", "C"], ["B", "C"]],
    },
}
gen = CausalDataGenerator(cfg)
result = gen.simulate()

true_cpdag = pc(result["data"], ci_test=gen.as_ci_oracle(), alpha=0.05)
recovered  = pc(result["data"], ci_test="fisherz", alpha=0.05)
```

Note that `bnm` does not import `cbcd` or `dagsampler`. Both
`true_cpdag` and `recovered` are instances of `cbcd.CPDAG`, which
satisfies the structural `bnm.GraphLike` Protocol; `bnm`'s
`to_graphlike` adapter normalises the input at the function
boundary.

## Comparative metrics

The structural Hamming distance counts the number of edges where
the two graphs disagree, partitioned into additions, deletions,
and reversals (Tsamardinos et al., 2006):

```python
bnm.shd(true_cpdag, recovered)        # 0
bnm.hd(true_cpdag, recovered)         # 0  — skeleton-only
bnm.f1(true_cpdag, recovered)         # 1.0
bnm.precision(true_cpdag, recovered)  # 1.0
bnm.recall(true_cpdag, recovered)     # 1.0
```

For the collider fixture above at $n = 3000$, the empirical
recovery matches the gold standard exactly. A more interesting
case is the diamond DAG $A \to B, A \to C, B \to D, C \to D$,
where Fisher–Z at the same sample size flips the orientation of
the upper edges; running the same comparison yields
$\mathrm{SHD} = 2$, $F_1 = 0.50$, with the two reversals localised
on `A — B` and `A — C`.

```{seealso}
Fine-grained reversal accounting (additions vs deletions vs
reversals) is exposed by `bnm.count_additions`,
`bnm.count_deletions`, `bnm.count_reversals`. The full multi-
metric façade `bnm.compare` runs every comparative metric and
returns a `Comparison` value object.
```

## Structural intervention distance

The structural intervention distance (Peters & Bühlmann, 2015)
measures the size of the set of pairs $(X, Y)$ for which the
recovered graph predicts a different *interventional* distribution
$P(Y \mid \mathrm{do}(X))$ than the true graph. Unlike SHD it is
sensitive to which mistakes matter for downstream causal
estimation.

```python
sid_result = bnm.sid(true_cpdag, recovered)
print(sid_result)
```

When the recovered graph is a CPDAG rather than a DAG, the SID is
no longer a single integer; `bnm.sid` returns both lower and upper
bounds over the equivalence class (Peters & Bühlmann, 2015,
§3.3). See [SID theory](explanation/sid_theory.md) for the
algorithm and the bound semantics.

## Visualisation

Graph comparison is most easily understood visually. The
side-by-side renderer paints matching edges in pastel red so true
positives are immediately identifiable:

```python
bnm.plot_side_by_side(
    true_cpdag, recovered,
    name1="true_cpdag", name2="recovered",
    direction="LR",
    save="comparison.svg",
)
```

The viz extras (`graphviz`, `plotly`, `ipython`) are gated behind
the optional `[viz]` extra; calling these functions without the
extra installed raises `ImportError` with an actionable hint.

## What is next

- The full metrics taxonomy and definitions —
  [Explanation: comparative metrics](explanation/metrics_theory.md).
- The SID algorithm and bound construction —
  [Explanation: SID](explanation/sid_theory.md).
- Working with `networkx.DiGraph` inputs —
  [How-to: networkx interop](howto/working_with_networkx.md).
- The `GraphLike` Protocol and design rationale —
  [Explanation: the GraphLike Protocol](explanation/graphlike_protocol.md).

## References

- Peters, J., & Bühlmann, P. (2015). Structural intervention
  distance for evaluating causal graphs. *Neural Computation*,
  27(3), 771–799.
- Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The
  max-min hill-climbing Bayesian network structure learning
  algorithm. *Machine Learning*, 65(1), 31–78.
