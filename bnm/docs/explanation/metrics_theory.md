# Comparative metrics

This page summarises the mathematical content of the comparative
metrics implemented in `bnm`. The treatment follows Tsamardinos,
Brown, & Aliferis (2006) and the survey in de Jongh & Druzdzel
(2009); readers who need full proofs should consult those sources.

## Setting

Let $G_1, G_2$ be two mixed graphs over the same vertex set $V$
with $|V| = d$. Each graph is encoded by a pair of $d \times d$
endpoint-mark matrices following the convention of decision **D5**
(`docs/design/api_v0.py` §D): `endpoints[i, j]` is the mark at
vertex $j$ of the edge $\{i, j\}$, taking values in
$\{0, 1, 2, 3\}$ (no edge / TAIL / ARROW / CIRCLE).

For comparative metrics it is convenient to *classify* each
unordered pair $\{i, j\}$ with $i < j$ into one of five disjoint
edge kinds:

| kind | endpoint $(i, j)$ | endpoint $(j, i)$ |
|---|:---:|:---:|
| no edge | NO_EDGE | NO_EDGE |
| directed forward $i \to j$ | ARROW | TAIL |
| directed backward $i \leftarrow j$ | TAIL | ARROW |
| undirected $i - j$ | TAIL | TAIL |
| bidirected $i \leftrightarrow j$ | ARROW | ARROW |

(CIRCLE-marked edges, used by PAGs, are classified as a sixth
kind and treated below.)

## Adjacency mask and edge classification

Let $A_k(i, j) = 1$ iff the unordered pair $\{i, j\}$ is adjacent
(edge kind $\neq$ no edge) in $G_k$. Let $C_k(i, j)$ be the kind
above (one of five for CPDAGs, one of six for PAGs). The
*skeleton-level* metrics depend only on $A_k$; the
*orientation-aware* metrics depend on the full $C_k$.

## Hamming distance

The (skeleton-only) Hamming distance counts pairs where adjacency
differs:

$$
\mathrm{HD}(G_1, G_2)
= \sum_{i < j} \mathbf{1}\!\left[A_1(i, j) \neq A_2(i, j)\right].
$$

It ignores edge orientation and edge kind. Implemented as
`bnm.hd`.

## Structural Hamming distance

The structural Hamming distance counts pairs where the *kind*
differs (Tsamardinos et al., 2006):

$$
\mathrm{SHD}(G_1, G_2)
= \sum_{i < j} \mathbf{1}\!\left[C_1(i, j) \neq C_2(i, j)\right]
= \mathrm{additions} + \mathrm{deletions} + \mathrm{reversals},
$$

where

- *additions* count pairs adjacent in $G_2$ but not in $G_1$;
- *deletions* count pairs adjacent in $G_1$ but not in $G_2$;
- *reversals* count pairs adjacent in both but with different
  edge kind.

The decomposition is exposed by `bnm.count_additions`,
`bnm.count_deletions`, `bnm.count_reversals` for fine-grained
diagnostics. Implemented as `bnm.shd`.

```{admonition} Convention
:class: note
A directed-vs-undirected mismatch (e.g. $i \to j$ in $G_1$ versus
$i - j$ in $G_2$) is counted as a single reversal — equivalently,
SHD treats every distinct edge kind as equally far apart.
de Jongh & Druzdzel (2009) discuss alternative conventions; bnm
v0.2 follows the Tsamardinos et al. (2006) definition.
```

## Precision, recall, F1

For exact-match orientation accounting, define:

$$
\begin{aligned}
\mathrm{TP}(G_1, G_2) &= \#\{(i, j) : A_1 = A_2 = 1 \text{ and } C_1 = C_2\}, \\
\mathrm{FP}(G_1, G_2) &= \#\{(i, j) : A_2 = 1\} - \mathrm{TP}, \\
\mathrm{FN}(G_1, G_2) &= \#\{(i, j) : A_1 = 1\} - \mathrm{TP}.
\end{aligned}
$$

Then

$$
\mathrm{precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}},
\qquad
\mathrm{recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}},
\qquad
F_1 = \frac{2 \cdot \mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}.
$$

By convention, $\mathrm{precision} = \mathrm{recall} = F_1 = 0$
when both graphs have no adjacencies. Implemented as
`bnm.precision`, `bnm.recall`, `bnm.f1`.

```{admonition} On the strictness of TP
:class: warning
`bnm` defines a true positive as an *exact-mark* match — same
adjacency *and* same edge kind. Some references count adjacency-
only TPs (i.e. either edge kind matches) and report SHD-by-skeleton
separately. Practitioners citing F1 from `bnm` should be explicit
about the orientation-aware definition.
```

## Audit findings preserved from v0.1.x

Several metric bugs in `bnm` v0.1.x were diagnosed during the
v0.2 rewrite (see `docs/audit.md` in the repo for the full list
with file:line references). The four most consequential:

1. **§1 — SID empty-parents crash.** v0.1.x's SID raised a
   `KeyError` on graphs with isolated nodes; v0.2 returns the
   correct integer SID.
2. **§6 — SID hash-seed non-determinism.** v0.1.x's SID iterated
   over a Python `set` of v-structure components, producing
   different outputs across `PYTHONHASHSEED` values; v0.2's
   iteration is deterministic.
3. **§7 — Reversal under-counting.** v0.1.x stored undirected
   edges asymmetrically in its `nx.DiGraph` representation; the
   reversal count under-reported by the count of asymmetric
   storage events. v0.2 stores both directions and counts
   correctly.
4. **§8 — SID upper-bound under-counting on CPDAG inputs.** The
   v0.1.x upper bound enumerated only a subset of the equivalence
   class; v0.2 enumerates the full class as required by Peters &
   Bühlmann (2015) Theorem 3.

19 fixtures across the 87 v0.1.x snapshots have intentional
v0.2-vs-v0.1.x semantic divergences; these are catalogued in
`tests/fixtures_legacy_v02_overrides.json`.

## References

- de Jongh, M., & Druzdzel, M. J. (2009). A comparison of
  structural distance measures for causal Bayesian network models.
  In *Recent Advances in Intelligent Information Systems*,
  443–456.
- Peters, J., & Bühlmann, P. (2015). Structural intervention
  distance for evaluating causal graphs. *Neural Computation*,
  27(3), 771–799.
- Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The
  max-min hill-climbing Bayesian network structure learning
  algorithm. *Machine Learning*, 65(1), 31–78.
