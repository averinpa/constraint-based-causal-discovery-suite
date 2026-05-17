# Tutorial: a first PC run

This tutorial walks through one full invocation of the PC algorithm
on observational data, from data simulation through CPDAG recovery
to structural evaluation. The reader is assumed to be familiar with
directed acyclic graphs, conditional independence, and the standard
faithfulness / causal Markov assumptions; for the underlying theory
see [Explanation: PC theory](explanation/pc_theory.md).

## Setup

We use the sister package [`dagsampler`](https://github.com/averinpa/dagsampler)
to simulate data from a known DAG so that the recovered CPDAG can
be compared against ground truth. cbcd does not depend on
`dagsampler`; the two packages communicate through the structural
`cbcd.CITest` Protocol.

```python
from dagsampler import CausalDataGenerator
from cbcd import pc

cfg = {
    "simulation_params": {
        "n_samples": 3000,
        "seed_structure": 1,
        "seed_data": 2,
        "binary_proportion": 0.0,
    },
    "graph_params": {
        "type": "custom",
        "nodes": ["A", "B", "C", "D"],
        "edges": [["A", "B"], ["A", "C"],
                  ["B", "D"], ["C", "D"]],
    },
}
result = CausalDataGenerator(cfg).simulate()
```

The DAG `A → B, A → C, B → D, C → D` has a single v-structure
(`B → D ← C`). Under faithfulness the only orientation that PC can
recover from observational data alone is the collider into `D`;
the upper edges (`A — B`, `A — C`) belong to the same Markov
equivalence class as their reversals and are returned undirected
in the CPDAG.

## Recovery

The default invocation uses cbcd's bundled Fisher–Z partial
correlation test:

```python
cpdag = pc(result["data"], alpha=0.05)
```

Under the `cbcd.CITest` Protocol any conforming object may be
substituted; for example,

```python
oracle = result_gen.as_ci_oracle()  # dagsampler.DSeparationOracle
cpdag_oracle = pc(result["data"], ci_test=oracle, alpha=0.05)
```

returns the CPDAG that PC would recover under a perfect d-separation
oracle. Because PC is sound and complete under faithfulness with a
perfect oracle, `cpdag_oracle` is the *true* CPDAG of the generating
DAG; it provides a principled benchmark against which the empirical
recovery `cpdag` can be scored. See
[Explanation: PC theory](explanation/pc_theory.md) for the soundness
and completeness statements.

## Reading the output

`cpdag` is an instance of `cbcd.CPDAG`, internally backed by an
`int8` endpoint matrix following the convention in
**D5** (`docs/design/api_v0.py`):

| value | meaning |
|:--:|:--|
| `0` | no edge |
| `1` | TAIL — the `o` end of `o→` or `o—` |
| `2` | ARROW — the `→` end |

A directed edge $i \to j$ is encoded as `endpoints[i, j] == ARROW`
and `endpoints[j, i] == TAIL`; an undirected edge as both ends
`TAIL`. For the diamond fixture above, `cpdag.endpoints` will show
`B → D` and `C → D` oriented (the v-structure) with `A — B` and
`A — C` left undirected.

```python
print(cpdag.endpoints)
# array([[0, 1, 1, 0],
#        [1, 0, 0, 2],
#        [1, 0, 0, 2],
#        [0, 1, 1, 0]], dtype=int8)
```

## Choosing α

The significance level $\alpha$ controls the type-I error rate of
each individual CI test. Under independent CI tests, the family-wise
type-I rate of the skeleton phase grows with the number of
performed tests; in practice cbcd does not apply a multiple-testing
correction, and the user is expected to choose $\alpha$ in light of
sample size and graph density. The default $\alpha = 0.05$ is
common in the literature (Spirtes et al., 2000). For large $n$,
tighter values (e.g. $\alpha = 0.01$) reduce spurious adjacencies;
for small $n$, looser values may be required to retain power.

A more thorough treatment of this tradeoff appears in
[How-to: choosing a CI test](howto/choosing_a_ci_test.md).

## Evaluation against the oracle

Endpoint-by-endpoint structural distance between the empirical and
oracle CPDAGs is reported by the `bnmetrics` package via its
[Structural Hamming Distance](https://en.wikipedia.org/wiki/Structural_Hamming_distance)
function. Like the dagsampler ↔ cbcd connection, cbcd ↔ bnmetrics passes
through a structural Protocol (`bnmetrics.GraphLike`) — neither package
imports the other.

```python
import bnmetrics
shd = bnmetrics.shd(cpdag_oracle, cpdag)
f1 = bnmetrics.f1(cpdag_oracle, cpdag)
```

For the diamond fixture at $n = 3000$ with default Fisher–Z and
$\alpha = 0.05$, this yields $\mathrm{SHD} = 2$ and $F_1 = 0.50$.
The two-edge gap is in the upper edges (`A — B`, `A — C`): the
oracle leaves them undirected, while Fisher–Z at this sample size
spuriously orients them in one direction. This kind of orientation
drift on undirected CPDAG edges is documented and expected; it is
not a regression in PC's logic but a power limitation of the
finite-sample CI test.

## What is next

- A different CI test for non-Gaussian or discrete data —
  [How-to: choosing a CI test](howto/choosing_a_ci_test.md).
- The full algorithmic theory and assumptions —
  [Explanation: PC theory](explanation/pc_theory.md).
- Time-series data — `cbcd.pcmci`, treated under the same Diátaxis
  structure in the time-series tutorial.

## References

- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation,
  Prediction, and Search* (2nd ed.). MIT Press.
