# Markov blanket comparison

The Markov blanket of a vertex $X$ in a DAG $G$ is the set
$\mathrm{MB}_G(X) = \mathrm{Pa}_G(X) \cup \mathrm{Ch}_G(X) \cup
\mathrm{Sp}_G(X)$, where $\mathrm{Sp}_G(X)$ denotes the *spouses*
of $X$ (the parents of $X$'s children, excluding $X$ itself).
It is the smallest set $\mathbf{S}$ such that
$X \perp\!\!\!\perp V \setminus (\mathbf{S} \cup \{X\}) \mid \mathbf{S}$
under the causal Markov assumption (Pearl, 1988).

```{note}
This page is currently a stub. The full treatment, including
Markov-blanket Jaccard distance and per-node comparison, will
land in v0.x.x.
```

## References

- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent
  Systems*. Morgan Kaufmann.
