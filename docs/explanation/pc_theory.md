# PC theory

This page summarises the theoretical foundations of the PC algorithm
as implemented in `cbcd.pc`. The treatment follows Spirtes,
Glymour, and Scheines (2000); readers who need the full proofs
should consult the textbook.

## Setting

Let $V = \{X_1, \dots, X_d\}$ be a finite set of random variables
with joint distribution $P$, and let $G = (V, E)$ be a directed
acyclic graph (DAG) over $V$. Write $\mathrm{Pa}_G(X)$ for the
parent set of $X$ in $G$ and $\mathrm{Nd}_G(X)$ for the
non-descendants of $X$ in $G$. Conditional independence is denoted
$X \perp\!\!\!\perp Y \mid \mathbf{S}$.

## Assumptions

The PC algorithm recovers the Markov equivalence class of $G$ from
i.i.d. samples of $P$ under the following conditions.

```{admonition} Assumption (Causal Markov)
:class: note
For every $X \in V$,
$X \perp\!\!\!\perp \mathrm{Nd}_G(X) \setminus \mathrm{Pa}_G(X) \mid \mathrm{Pa}_G(X)$
in $P$.
```

```{admonition} Assumption (Faithfulness)
:class: note
Every conditional independence $X \perp\!\!\!\perp Y \mid \mathbf{S}$
that holds in $P$ is implied by d-separation of $X$ and $Y$ given
$\mathbf{S}$ in $G$.
```

```{admonition} Assumption (Causal sufficiency)
:class: note
For every pair $X, Y \in V$, no variable outside $V$ is a common
cause of $X$ and $Y$.
```

When (3) is violated, FCI (Spirtes et al., 2000, Ch. 6) replaces
PC; see [FCI theory](fci_theory.md).

## D-separation and the CI oracle

Under the conjunction of causal Markov and faithfulness, a graphical
criterion (d-separation) and a probabilistic criterion (conditional
independence) coincide:

$$
X \perp\!\!\!\perp_{P} Y \mid \mathbf{S}
\;\Longleftrightarrow\;
X \perp\!\!\!\perp_{G,\,\text{d-sep}} Y \mid \mathbf{S}.
$$

This equivalence is what PC exploits: a stream of CI judgements over
$V$ — supplied by either a statistical test on samples of $P$ or
a d-separation oracle on $G$ — uniquely determines the Markov
equivalence class of $G$ (Verma & Pearl, 1990).

## Algorithm

PC operates in two phases.

### Phase 1 — skeleton recovery

Initialise the working graph as the complete undirected graph on
$V$. For $\ell = 0, 1, 2, \dots$, iterate over each adjacent pair
$(X, Y)$ in the working graph. If there exists a separating set
$\mathbf{S} \subseteq \mathrm{Adj}(X) \setminus \{Y\}$ with
$|\mathbf{S}| = \ell$ such that
$X \perp\!\!\!\perp Y \mid \mathbf{S}$ under the chosen CI test,
remove the edge $(X, Y)$ and record $\mathbf{S}$ as the
*separating set* of $X, Y$, denoted $\mathrm{Sep}(X, Y)$. Halt
when no further removals occur.

The implementation in `cbcd.skeleton.PCStable` uses the order-
independent variant due to Colombo & Maathuis (2014): the
adjacency list used to enumerate candidate separating sets at
level $\ell$ is fixed at the start of each level, so the result
does not depend on the iteration order of CI tests.

### Phase 2 — orientation

Identify *unshielded triples* $(X, Z, Y)$ such that $(X, Z)$ and
$(Z, Y)$ are adjacent but $(X, Y)$ is not. If
$Z \notin \mathrm{Sep}(X, Y)$, orient $X \to Z \leftarrow Y$
(v-structure). Then apply Meek's rules R1–R4 (Meek, 1995) to
propagate orientations until no further edges can be directed
without introducing a cycle or a new v-structure.

The implementation factors these phases through the `cbcd.collider`
and `cbcd.rules` modules; the precise correspondence to the
literature is documented in the source.

## Soundness and completeness

```{admonition} Theorem (Spirtes et al., 2000, §5.4)
:class: important
Under causal Markov, faithfulness, causal sufficiency, and access
to a perfect CI oracle for $P$, the output of PC is the unique
CPDAG of $G$.
```

In the presence of statistical noise (i.e. when the CI oracle is
replaced by a finite-sample test), the recovered CPDAG converges
in probability to the CPDAG of $G$ as $n \to \infty$ at a rate
determined by the consistency of the chosen test. Convergence of
Fisher–Z under linear-Gaussian SCMs is given by Kalisch & Bühlmann
(2007).

## Practical caveats

1. **Faithfulness violations.** Linear cancellations of the form
   $\sum_k \alpha_k \beta_k = 0$ along multiple paths from $X$ to
   $Y$ can produce CI relations not implied by d-separation in $G$.
   The set of parameter values exhibiting this is Lebesgue measure
   zero in the linear-Gaussian model (Spirtes et al., 2000,
   Theorem 3.2), but power is reduced near the cancellation
   boundary.
2. **Conservative variants.** When v-structure orientation
   decisions disagree across overlapping unshielded triples,
   `cbcd` exposes the conservative (Ramsey et al., 2006) and
   majority (Colombo & Maathuis, 2014) collider rules in addition
   to the default sepset-based rule. See
   `cbcd.ColliderOrienter`.
3. **Multiple testing.** `cbcd` does not apply a multiple-testing
   correction across the CI tests in the skeleton phase. The
   user-facing $\alpha$ governs each individual test.

## References

- Colombo, D., & Maathuis, M. H. (2014). Order-independent
  constraint-based causal structure learning. *Journal of Machine
  Learning Research*, 15(1), 3741–3782.
- Kalisch, M., & Bühlmann, P. (2007). Estimating high-dimensional
  directed acyclic graphs with the PC-algorithm. *Journal of
  Machine Learning Research*, 8, 613–636.
- Meek, C. (1995). Causal inference and causal explanation with
  background knowledge. In *Proceedings of UAI '95*, 403–410.
- Ramsey, J., Zhang, J., & Spirtes, P. (2006). Adjacency-faithfulness
  and conservative causal inference. In *Proceedings of UAI '06*,
  401–408.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation,
  Prediction, and Search* (2nd ed.). MIT Press.
- Verma, T., & Pearl, J. (1990). Equivalence and synthesis of
  causal models. In *Proceedings of UAI '90*, 255–270.
