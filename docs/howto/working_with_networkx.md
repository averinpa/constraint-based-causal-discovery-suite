# Working with networkx graphs

`networkx.DiGraph` instances are accepted by `bnm.to_graphlike`,
which converts them to the canonical `(n_vars, endpoints, var_names)`
representation. Edge orientation is taken from the directed-edge
convention; undirected (CPDAG-style) edges are encoded by setting
`graph[u][v]["type"] = "undirected"` on the networkx side.

```python
import networkx as nx
import bnm

g = nx.DiGraph()
g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
gl = bnm.to_graphlike(g)
bnm.shd(gl, other_graph)
```

```{note}
This page is currently a stub. A worked example covering both
DAG and CPDAG-style networkx inputs will land in v0.x.x.
```
