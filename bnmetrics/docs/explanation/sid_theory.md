# Structural intervention distance

The structural intervention distance (SID) is a graph-distance
metric introduced by Peters & Bühlmann (2015) that, unlike the
structural Hamming distance, weights structural mistakes by their
consequences for downstream causal inference. It counts the number
of variable pairs $(X, Y)$ for which the recovered graph would
support an *incorrect* interventional distribution
$P(Y \mid \mathrm{do}(X))$ relative to the true graph.

## Setting

Let $G^* = (V, E^*)$ be the true DAG over a vertex set $V$ with
$|V| = d$, and let $\hat G$ be a recovered structure (a DAG, a
CPDAG, or a more general mixed graph). For each ordered pair
$(X, Y)$ with $X \neq Y$, let $\mathrm{Pa}_G(X)$ denote the parents
of $X$ in $G$. The interventional distribution under the
graphical do-calculus (Pearl, 2009) is identifiable from $G$ via
the *adjustment formula*:

$$
P(Y \mid \mathrm{do}(X = x))
= \sum_{\mathbf{z}} P(Y \mid X = x, \mathrm{Pa}_G(X) = \mathbf{z}) \,
  P(\mathrm{Pa}_G(X) = \mathbf{z}),
$$

provided that the parent set $\mathrm{Pa}_G(X)$ in $G$ satisfies
the back-door criterion. SID compares the parent sets recommended
by $G^*$ and $\hat G$.

## Definition (DAG vs DAG)

For two DAGs $G^*$ and $\hat G$ over the same vertex set,

$$
\mathrm{SID}(G^*, \hat G)
= \sum_{X \neq Y} \mathbf{1}\!\left[
  \mathrm{Pa}_{\hat G}(X)
  \text{ is not a valid adjustment set for } (X, Y) \text{ in } G^*
\right].
$$

The SID is zero iff every parent set in $\hat G$ is also a valid
adjustment set in $G^*$ — i.e. iff $\hat G$ supports the same
interventional distributions as $G^*$ for every ordered pair.

## Definition (DAG vs CPDAG)

When $\hat G$ is a CPDAG (e.g. the output of PC), it represents
the Markov equivalence class $[\hat G] = \{D_1, \dots, D_K\}$ of
DAGs consistent with the same skeleton + v-structure pattern.
Following Peters & Bühlmann (2015) §3.3, `bnmetrics.sid` returns both
bounds:

$$
\mathrm{SID}_{\mathrm{lower}}(G^*, \hat G)
= \min_{D_k \in [\hat G]} \mathrm{SID}(G^*, D_k),
\quad
\mathrm{SID}_{\mathrm{upper}}(G^*, \hat G)
= \max_{D_k \in [\hat G]} \mathrm{SID}(G^*, D_k).
$$

The lower bound corresponds to the *best-case* DAG extension, the
upper bound to the *worst-case*. Computing these bounds reduces
to enumerating the equivalence class via Dor–Tarsi extension, then
evaluating the DAG-vs-DAG SID for each member.

```{admonition} Implementation note
:class: note
Naïve enumeration of the equivalence class is exponential in the
number of undirected edges. `bnmetrics.sid` uses the Chickering (2002)
enumeration algorithm, which prunes at each step using the Meek-
rule completeness theorem, keeping practical runtime acceptable
on graphs with up to ~12 undirected edges per connected
undirected component.
```

## Interpretation

The number returned by `bnmetrics.sid` is in $\{0, 1, \dots, d(d-1)\}$.
Values should be reported alongside the maximum,
$\mathrm{SID}_{\max} = d(d-1)$, since SID is not normalised. For
$d = 10$, $\mathrm{SID}_{\max} = 90$; an SID of 5 means that 5 of
the 90 ordered pairs are mis-adjusted by $\hat G$.

```{admonition} Comparison with SHD
:class: tip
SHD and SID are not directly comparable. A graph that disagrees on
*one* edge incident to a high-fanout hub will produce many
mis-adjusted pairs (large SID) while differing in only one edge
(small SHD). Conversely, a graph that disagrees on multiple edges
incident to peripheral leaves may have large SHD but small SID.
For applied causal effect estimation, SID is the more directly
relevant metric.
```

## What SID is not

SID does *not* assess the *magnitude* of the adjustment error: a
mis-adjusted pair is counted regardless of whether the adjustment
formula returns a slightly biased estimate or a wildly wrong one.
For applications where the magnitude matters, simulate the
interventional distributions explicitly (e.g. via the suite's
`dagsampler`) rather than relying on SID alone.

## References

- Chickering, D. M. (2002). Learning equivalence classes of
  Bayesian network structures. *Journal of Machine Learning
  Research*, 2, 445–498.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
  (2nd ed.). Cambridge University Press.
- Peters, J., & Bühlmann, P. (2015). Structural intervention
  distance for evaluating causal graphs. *Neural Computation*,
  27(3), 771–799.
