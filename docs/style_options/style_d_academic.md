# `cbcd.pc()`

```text
cbcd.pc(data, ci_test="fisherz", alpha=0.05, ...)
```

Constraint-based recovery of the Markov equivalence class of a
faithful directed acyclic graph from i.i.d. observational data.

## Background

The PC algorithm (Spirtes & Glymour, 1991; Spirtes et al., 2000)
recovers the Markov equivalence class of a directed acyclic graph
$G$ over an observed variable set $V$, given access to a stream of
conditional independence judgements over $V$.

Under the **causal Markov** condition, every variable is independent
of its non-descendants given its parents in $G$. Under
**faithfulness**, the converse also holds: every conditional
independence in the data-generating distribution is implied by
d-separation in $G$. Together, these conditions imply

$$
X \perp\!\!\!\perp Y \mid \mathbf{S}
\;\;\Longleftrightarrow\;\;
X \text{ d-separated from } Y \text{ by } \mathbf{S} \text{ in } G.
$$

PC exploits this equivalence. It queries CI relations (statistically
or oracularly) to first recover the *skeleton* of $G$ — the
undirected graph with the same adjacencies — and then orients as
many edges as the equivalence class permits, returning the resulting
**completed partially directed acyclic graph** (CPDAG).

## Assumptions

1. **Faithfulness.** Every conditional independence in $P$ is
   implied by d-separation in $G$.
2. **Causal Markov.** Every variable is independent of its
   non-descendants given its parents in $G$.
3. **Causal sufficiency.** No unobserved common cause exists for
   any pair of variables in $V$.

When (3) is suspect, the user should consult `cbcd.fci`, whose FCI
algorithm relaxes causal sufficiency and returns a partial ancestral
graph (PAG) over the observed variables.

## Algorithm

The skeleton phase iteratively removes edges $(X, Y)$ when there
exists a separating set $\mathbf{S} \subseteq \mathrm{Adj}(X) \setminus \{Y\}$
of cardinality $\leq \ell$ such that
$X \perp\!\!\!\perp Y \mid \mathbf{S}$ under the chosen CI test.
The conditioning-set size $\ell$ grows from $0$ until no further
removals are possible.

The orientation phase identifies *unshielded v-structures*
$X \to Z \leftarrow Y$ on triples $X - Z - Y$ where $X$ and $Y$
are non-adjacent and $Z$ is not a member of the separating set
recorded for $(X, Y)$ during skeleton recovery. The CPDAG is then
completed via Meek's rules R1–R4 (Meek, 1995).

The implementation in `cbcd/cbcd/algorithms/pc.py` factors these
phases through the `SkeletonAlgorithm`, `ColliderOrienter`, and
`CPDAGRules` Protocols (see `docs/design/api_v0.py` §C–§F).

## Returns

`cbcd.CPDAG` — the recovered CPDAG, encoded as an int8 endpoint
matrix following the convention in §D of the design document.

## References

- Meek, C. (1995). Causal inference and causal explanation with
  background knowledge. In *Proceedings of the Eleventh Conference
  on Uncertainty in Artificial Intelligence* (UAI '95), pp. 403–410.
- Spirtes, P., Glymour, C. (1991). An algorithm for fast recovery
  of sparse causal graphs. *Social Science Computer Review*, 9(1),
  62–72.
- Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation,
  Prediction, and Search* (2nd ed.). MIT Press.
