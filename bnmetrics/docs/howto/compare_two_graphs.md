# Compare two graphs

The fastest path to a full comparison is `bnmetrics.compare`, which runs
every comparative metric and packages the results into a
`Comparison` value object.

## Worked example: the diamond fixture

The canonical pedagogical fixture is the four-node diamond DAG
$A \to B,\; A \to C,\; B \to D,\; C \to D$. Its CPDAG keeps the
v-structure into $D$ directed but leaves the upper edges $A - B$
and $A - C$ undirected (they lie in the Markov-equivalence class).
A common recovery error at moderate sample sizes is to flip these
upper edges, returning $B \to A$ and $C \to A$ — two reversals.

```python
import numpy as np
import bnmetrics

# True CPDAG of the diamond.
true_cpdag = bnmetrics.to_graphlike(
    np.array([
        # cols A  B  C  D
        [0, 1, 1, 0],   # A
        [1, 0, 0, 2],   # B
        [1, 0, 0, 2],   # C
        [0, 1, 1, 0],   # D
    ], dtype=np.int8),
    var_names=("A", "B", "C", "D"),
)

# Recovered CPDAG with both upper edges reversed.
recovered = bnmetrics.to_graphlike(
    np.array([
        # cols A  B  C  D
        [0, 1, 1, 0],   # A
        [2, 0, 0, 2],   # B
        [2, 0, 0, 2],   # C
        [0, 1, 1, 0],   # D
    ], dtype=np.int8),
    var_names=("A", "B", "C", "D"),
)
```

## Single-metric calls

For interactive inspection of one metric, the standalone functions
are equivalent to (and faster than) extracting the same field from
a `Comparison`:

```python
bnmetrics.shd(true_cpdag, recovered)               # 2
bnmetrics.hd(true_cpdag, recovered)                # 0   — skeleton-only
bnmetrics.f1(true_cpdag, recovered)                # 0.5
bnmetrics.precision(true_cpdag, recovered)         # 0.5
bnmetrics.recall(true_cpdag, recovered)            # 0.5
bnmetrics.count_reversals(true_cpdag, recovered)   # 2
```

The Hamming distance is zero because both graphs share the same
skeleton; all error is concentrated in orientation. SHD = 2 and
$F_1 = 0.5$ together localise the discrepancy to two reversals.

## Multi-metric Comparison

```python
comp = bnmetrics.compare(true_cpdag, recovered)
comp.comparative["shd"]        # 2.0
comp.comparative["f1"]         # 0.5
comp.comparative["reversals"]  # 2.0
```

`comp.comparative` is a dictionary keyed by metric name; the
available keys are `additions`, `deletions`, `reversals`, `shd`,
`hd`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`. Descriptive
metrics for each input graph are exposed on `comp.g1_descriptive`
and `comp.g2_descriptive`.

## Tidy DataFrame export

`bnmetrics.to_dataframe` flattens a `Comparison` into a single-row
`pandas.DataFrame` combining descriptive and comparative metrics
(requires the `pandas` extra):

```python
df = bnmetrics.to_dataframe(comp)
df[["shd", "hd", "f1", "precision", "recall",
    "additions", "deletions", "reversals"]]
```

```text
 shd  hd   f1  precision  recall  additions  deletions  reversals
 2.0 0.0  0.5        0.5     0.5        0.0        0.0        2.0
```

## Variable-name alignment

`bnmetrics` aligns variables by name when both inputs expose
`var_names`; otherwise it aligns positionally. A `BNMDataError`
is raised if the two graphs have different variable counts or
non-matching name sets, since silent positional alignment of
inconsistently-named graphs is the most common source of subtle
metric bugs.
