# Encoding background knowledge

`cbcd.BackgroundKnowledge` lets the practitioner pin required and
forbidden adjacencies and orientations before the algorithm runs.
The constraints are validated for internal consistency at
construction time; inconsistent constraints raise
`CBCDInputError` rather than silently degrading recovery quality.

```{note}
This page is currently a stub. A worked example with required and
forbidden edges plus orientation constraints will land in v0.x.x.
```
