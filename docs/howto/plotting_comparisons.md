# Plotting comparisons

`bnm.plot_side_by_side(g1, g2, name1=..., name2=...)` renders two
graphs as paired graphviz diagrams. Edges that match between the
two panels (same skeleton, same orientation) are highlighted in
pastel red to make true positives visually distinguishable from
errors:

```python
import bnm
bnm.plot_side_by_side(
    g1, g2,
    name1="truth", name2="recovered",
    direction="LR",
    save="comparison.svg",
)
```

The viz functions are gated behind the optional `[viz]` extra
(`graphviz`, `plotly`, `ipython`); they raise `ImportError` with
an actionable hint when called without the extra installed.

```{note}
This page is currently a stub. A worked example covering edge
highlighting, layout choices, and SID heatmap rendering will land
in v0.x.x.
```
