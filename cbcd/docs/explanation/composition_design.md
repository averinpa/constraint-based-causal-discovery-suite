# Composition-first design

`cbcd` is structured around a small set of structural Protocols
rather than a single monolithic `pc(...)` function. Each algorithm
in the package is wired from these pieces:

- `CITest` — the conditional-independence oracle / statistical
  test (§A of `docs/design/api_v0.py`).
- `SkeletonAlgorithm` — the skeleton-discovery phase (§C).
- `ColliderOrienter` — the v-structure orientation strategy (§E).
- `CPDAGRules`, `PAGRules`, `PAGSkeletonRefinement` — the
  edge-orientation rule families (§F).

Top-level functions (`cbcd.pc`, `cbcd.fci`, `cbcd.rfci`,
`cbcd.anytime_fci`, `cbcd.pcmci`) compose these pieces in §G
(i.i.d.) and §H (time-series). The precise composition for each
algorithm is documented in the source.

## Why this matters

A new algorithm — for example, CDNOD or J-PCMCI — should mostly be
plumbing existing pieces: choose the appropriate skeleton phase,
collider rule, and edge-orientation set, then assemble them in a
two- or three-line top-level function. This shape mirrors the
algorithmic structure of the published constraint-based literature,
in which most "new" algorithms are recombinations of existing
phases for a specific setting.

## Why no `causal-learn` dependency

`causal-learn` (Wang et al., 2024) is the most widely-used
constraint-based discovery library. cbcd does **not** depend on
it. The reasons are catalogued in `cbcd/docs/audit_causal_learn.md`
(developer-facing) and summarised here:

- **API stability.** `causal-learn`'s public surface mixes pure
  algorithmic functions with classes that hold mutable state and
  internal caches. Refactoring across versions has historically
  broken downstream code.
- **Algorithmic auditing.** Several CI tests and orientation
  routines have known bugs (documented in the audit) that have
  produced incorrect parity numbers in published comparisons.
- **Dependency surface.** `causal-learn` transitively pulls in
  several heavy ML dependencies that are not required for
  constraint-based discovery proper.

cbcd ships its own minimal CI test layer (Fisher–Z, χ², G², KCI
planned) and accepts external CI tests through the structural
`CITest` Protocol. Users who prefer `causal-learn`'s tests can
wrap them in a Protocol-conforming class without modification to
either library.

## References

- Wang, Y., Wang, B., Wei, Y., Yang, J., Pratt, L. Y., Hwang, S.,
  & Zhang, K. (2024). Causal-learn: Causal discovery in Python.
  *Journal of Machine Learning Research*, 25.
