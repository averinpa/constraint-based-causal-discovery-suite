# Markov blanket comparison

## Definition

The Markov blanket of a vertex $X$ in a DAG $G$ is the set

$$\mathrm{MB}_G(X) = \mathrm{Pa}_G(X) \cup \mathrm{Ch}_G(X) \cup
\mathrm{Sp}_G(X),$$

where $\mathrm{Sp}_G(X)$ denotes the *spouses* of $X$ — the parents
of $X$'s children, excluding $X$ itself. Under the causal Markov
assumption (Pearl, 1988), $\mathrm{MB}_G(X)$ is the smallest set
$\mathbf{S} \subseteq V$ for which

$$X \perp\!\!\!\perp V \setminus (\mathbf{S} \cup \{X\}) \mid \mathbf{S}.$$

That is, conditioning on $\mathrm{MB}_G(X)$ renders $X$ independent
of every other variable in the graph. In a Bayesian-network sense
$\mathrm{MB}_G(X)$ contains *exactly* the information needed to
predict $X$, and no more.

## Why scope a comparison to the Markov blanket?

Many downstream uses of a recovered graph depend on $\hat{G}$ only
through the per-vertex Markov blankets, not through the global
structure:

- **Feature selection / prediction of $X$.** The Bayes-optimal
  predictor of $X$ from observational data depends only on
  $\mathrm{MB}(X)$. Errors in $\mathrm{MB}_{\hat{G}}(X)$ translate
  directly into prediction bias for $X$.
- **Conditional independence testing.** A CI test
  $X \perp\!\!\!\perp Y \mid \mathbf{S}$ is consistent with $G$ only
  if $\mathbf{S}$ separates $X$ and $Y$ in $G$; the relevant
  separating sets are subsets of $\mathrm{MB}_G(X)$ or
  $\mathrm{MB}_G(Y)$.
- **Local-search learners.** Algorithms in the Max-Min Parents and
  Children / IAMB family (Tsamardinos, Brown, & Aliferis, 2003)
  recover one Markov blanket at a time and never assemble the
  global structure.

For these workflows the per-vertex MB quality of $\hat{G}$ is more
informative than a single global SHD. The latter aggregates errors
across the whole graph; the former localises them to the variables
they affect.

## Two notions of Markov-blanket error

A pair $(\hat{G}, G^*)$ can disagree about the Markov blanket of
$X$ in two distinct ways:

### Structural error

The two Markov-blanket *sub-graphs* differ — same vertices, but
different edges or orientations. Quantified by applying any
standard comparative metric to the sub-graphs:

$$\mathrm{SHD}_{\mathrm{MB}(X)}(\hat{G}, G^*) =
\mathrm{SHD}\bigl(\mathrm{MB}_{\hat{G}}(X),\, \mathrm{MB}_{G^*}(X)\bigr).$$

This is the natural local analogue of the global SHD. By
construction $\mathrm{SHD}_{\mathrm{MB}(X)} \le \mathrm{SHD}$, and
the gap quantifies the share of structural error that lies outside
$\mathrm{MB}(X)$.

### Membership error

The two blankets differ as *sets of vertices*, irrespective of how
the within-blanket edges are oriented. Quantified by the Jaccard
distance

$$d_{\mathrm{J}}\bigl(\mathrm{MB}_{\hat{G}}(X),\, \mathrm{MB}_{G^*}(X)\bigr)
= 1 -
\frac{\lvert \mathrm{MB}_{\hat{G}}(X) \cap \mathrm{MB}_{G^*}(X) \rvert}
     {\lvert \mathrm{MB}_{\hat{G}}(X) \cup \mathrm{MB}_{G^*}(X) \rvert}.$$

Membership error captures the question "which variables enter the
blanket?" — relevant when the downstream task is feature selection
rather than structure learning. The two notions are complementary:
the same MB membership can support very different sub-graph
structures, and conversely two identical sub-graphs have zero
Jaccard distance trivially.

## API surface in `bnm`

| call | returns | use |
|---|---|---|
| `bnm.markov_blanket_indices(g, var)` | tuple of int indices for $\{X\} \cup \mathrm{MB}_G(X)$ | feed into a custom membership metric (e.g. Jaccard) |
| `bnm.markov_blanket(g, var)` | sub-`GraphLike` over $\{X\} \cup \mathrm{MB}_G(X)$, with every endpoint mark incident to those vertices preserved | pass to any standard comparative metric for the *structural* error |
| `bnm.compare(g1, g2, per_node=True)` | a `Comparison` whose `.per_node` field maps each vertex name to its descriptive + comparative metrics, computed on the MB sub-graph | survey MB-scoped error across all vertices in one call |

Jaccard distance over MB membership is not currently a built-in
metric; it can be computed in two lines from
`markov_blanket_indices` if needed.

## References

- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent
  Systems*. Morgan Kaufmann.
- Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2003).
  Algorithms for large scale Markov blanket discovery.
  *Proceedings of FLAIRS-16*, 376–380.
