# Working with networkx graphs

`bnm.to_graphlike` accepts a `networkx.DiGraph` and converts it to
the canonical `(n_vars, endpoints, var_names)` representation.
Directed edges follow the DiGraph orientation convention; undirected
(CPDAG-style) edges are encoded by setting the edge attribute
`type="undirected"`, in which case the adapter also requires the
reverse edge to be present.

## DAG input

```python
import networkx as nx
import bnm

g = nx.DiGraph()
g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
gl = bnm.to_graphlike(g)

gl.var_names      # ('A', 'B', 'C', 'D')
gl.n_vars         # 4
```

The adapter infers `var_names` from the node iteration order of the
DiGraph. Pass an explicit `var_names=(...)` to `to_graphlike` to
override.

## CPDAG-style input with undirected edges

The CPDAG of the diamond DAG above leaves the upper edges $A - B$
and $A - C$ undirected. Encode each undirected edge as a pair of
directed edges marked `type="undirected"`:

```python
g_cpdag = nx.DiGraph()
# Undirected upper edges.
g_cpdag.add_edge("A", "B", type="undirected")
g_cpdag.add_edge("B", "A", type="undirected")
g_cpdag.add_edge("A", "C", type="undirected")
g_cpdag.add_edge("C", "A", type="undirected")
# Directed v-structure into D.
g_cpdag.add_edge("B", "D")
g_cpdag.add_edge("C", "D")
gl_cpdag = bnm.to_graphlike(g_cpdag)

bnm.shd(gl, gl_cpdag)   # 2 — the two upper edges differ in orientation
```

A `BNMInputError` is raised if an `type="undirected"` edge is added
in only one direction, since silent half-undirected inputs are the
most common encoding mistake.
