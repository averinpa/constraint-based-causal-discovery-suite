# FCI theory

The FCI algorithm (Spirtes et al., 2000, Ch. 6) drops the causal
sufficiency assumption required by PC, returning a partial ancestral
graph (PAG) over the observed variables. The PAG is a compact
representation of the Markov equivalence class of *maximal ancestral
graphs* (MAGs) compatible with the observed CI relations.

```{note}
This page is currently a stub. The full Background / Assumptions /
Algorithm / References treatment, parallel to
[PC theory](pc_theory.md), will land in v0.x.x.
```

## References

- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation,
  Prediction, and Search* (2nd ed.). MIT Press, Chapter 6.
- Zhang, J. (2008). On the completeness of orientation rules for
  causal discovery in the presence of latent confounders and
  selection bias. *Artificial Intelligence*, 172(16-17), 1873–1896.
