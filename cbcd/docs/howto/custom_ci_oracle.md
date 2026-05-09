# Wiring a custom CI test or oracle

Any object satisfying the structural `cbcd.CITest` Protocol —
i.e. exposing `n_vars: int`, `__call__(x, y, S) -> float`, and
`details(x, y, S)` returning an object with a `.p_value` attribute —
can be passed to `cbcd.pc`, `cbcd.fci`, etc. as `ci_test=`.

```{note}
This page is currently a stub. A worked example wrapping a
user-supplied callable into a Protocol-conforming object will land
in v0.x.x.
```
