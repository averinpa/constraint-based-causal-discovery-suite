# Configuration cookbook

Practical config templates for all major simulator options. Each
example below is a complete `dict` (JSON-equivalent) that can be
passed directly to
`CausalDataGenerator(config).simulate()`.

## Seeding

The simulator uses two independent random streams:

- `rng_structure` controls the **data-generating process** itself —
  random DAG topology, sampled structural weights, intercepts,
  thresholds, and stratum means.
- `rng_data` controls the **per-sample draws** given that DGP —
  exogenous variable values, noise draws, and Bernoulli /
  categorical sampling.

You can seed them in two ways:

- **Single `seed`** (convenience): `seed_structure` is set to
  `seed` and `seed_data` is derived as `seed + 1` so the two
  streams stay independent while remaining fully reproducible.
  Use this for one-off examples and quickstarts.
- **Explicit `seed_structure` and `seed_data`**: seed each stream
  independently. This is the recommended form for benchmarks
  because it lets you decouple structure from data — for example,
  hold `seed_structure` fixed and vary `seed_data` to measure how
  a CI test behaves on different finite samples from the *same*
  DGP.

The minimal custom-DAG example below uses the single-seed form;
the random-DAG example uses the explicit pair.

## Minimal custom DAG

```json
{
  "simulation_params": {
    "n_samples": 200,
    "seed": 42
  },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "Y", "Z1"],
    "edges": [["X", "Z1"], ["Y", "Z1"]]
  }
}
```

## Random DAG

```json
{
  "simulation_params": {
    "n_samples": 300,
    "seed_structure": 123,
    "seed_data": 124
  },
  "graph_params": {
    "type": "random",
    "n_nodes": 6,
    "edge_prob": 0.35
  }
}
```

When `type = "random"` and node types are not pinned in
`node_params`, the simulator samples a type per node using
`binary_proportion` (default `0.4`) and `categorical_proportion`
(default `0.0`); the remainder become continuous. Override either
to control the type mix:

```json
{
  "simulation_params": {
    "n_samples": 300,
    "seed_structure": 123,
    "seed_data": 124,
    "binary_proportion": 0.2,
    "categorical_proportion": 0.3
  },
  "graph_params": { "type": "random", "n_nodes": 6, "edge_prob": 0.35 }
}
```

## Random weights with near-zero exclusion (signal-strength control)

```json
{
  "simulation_params": {
    "n_samples": 500,
    "seed_structure": 201,
    "seed_data": 202,
    "random_weight_low": -1.5,
    "random_weight_high": 1.5,
    "random_weight_min_abs": 0.1
  },
  "graph_params": {
    "type": "custom",
    "nodes": ["X1", "X2", "X3", "Y"],
    "edges": [["X1", "Y"], ["X2", "Y"], ["X3", "Y"]]
  },
  "node_params": {
    "X1": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "X2": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "X3": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Y": {
      "type": "continuous",
      "functional_form": { "name": "linear" },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.2 }
    }
  }
}
```

In this setup, omitted linear weights are sampled from
$[-1.5, -0.1] \cup [0.1, 1.5]$, guaranteeing every edge contributes
a minimum amount of signal rather than being effectively muted by
a near-zero draw.

## Categorical parent with metric form policy override

By default, categorical parents with
`linear` / `polynomial` / `interaction` raise an error. To
auto-redirect to `stratum_means` (including mixed-parent cases):

```json
{
  "simulation_params": {
    "n_samples": 300,
    "seed": 303,
    "categorical_parent_metric_form_policy": "stratum_means"
  },
  "graph_params": {
    "type": "custom",
    "nodes": ["C", "Y"],
    "edges": [["C", "Y"]]
  },
  "node_params": {
    "C": { "type": "categorical", "cardinality": 4 },
    "Y": {
      "type": "continuous",
      "functional_form": { "name": "linear" },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.2 }
    }
  }
}
```

## Exogenous node distributions

```json
{
  "simulation_params": {
    "n_samples": 500,
    "seed": 1
  },
  "graph_params": {
    "type": "custom",
    "nodes": ["G", "T", "Ga", "E", "B", "C"],
    "edges": []
  },
  "node_params": {
    "G": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0.0, "std": 1.0 } },
    "T": { "type": "continuous", "distribution": { "name": "student_t", "df": 4 } },
    "Ga": { "type": "continuous", "distribution": { "name": "gamma", "shape": 2.0, "scale": 1.0 } },
    "E": { "type": "continuous", "distribution": { "name": "exponential", "scale": 1.2 } },
    "B": { "type": "binary", "distribution": { "name": "bernoulli", "p": 0.35 } },
    "C": {
      "type": "categorical",
      "cardinality": 5,
      "distribution": { "probs": [0.1, 0.2, 0.3, 0.2, 0.2] }
    }
  }
}
```

## Continuous child with linear / polynomial / interaction

```json
{
  "simulation_params": { "n_samples": 300, "seed": 10 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X1", "X2", "Y_lin", "Y_poly", "Y_int"],
    "edges": [["X1", "Y_lin"], ["X2", "Y_lin"], ["X1", "Y_poly"], ["X2", "Y_poly"], ["X1", "Y_int"], ["X2", "Y_int"]]
  },
  "node_params": {
    "X1": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "X2": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Y_lin": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X1": 1.2, "X2": -0.7 } },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.5 }
    },
    "Y_poly": {
      "type": "continuous",
      "functional_form": { "name": "polynomial", "weights": { "X1": 1.0, "X2": 0.6 }, "degrees": { "X1": 3, "X2": 2 } },
      "noise_model": { "name": "additive", "dist": "student_t", "df": 5, "scale": 0.3 }
    },
    "Y_int": {
      "type": "continuous",
      "functional_form": { "name": "interaction", "weights": { "interaction": 0.8 } },
      "noise_model": { "name": "multiplicative", "dist": "gaussian", "std": 0.2 }
    }
  }
}
```

## Continuous child with sigmoid / cos / sin

```json
{
  "simulation_params": { "n_samples": 300, "seed": 11 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X1", "X2", "Y_sig", "Y_cos", "Y_sin"],
    "edges": [["X1", "Y_sig"], ["X2", "Y_sig"], ["X1", "Y_cos"], ["X2", "Y_cos"], ["X1", "Y_sin"], ["X2", "Y_sin"]]
  },
  "node_params": {
    "X1": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "X2": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Y_sig": {
      "type": "continuous",
      "functional_form": { "name": "sigmoid", "weights": { "X1": 1.0, "X2": -0.5 }, "output_weight": 1.5 },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.3 }
    },
    "Y_cos": {
      "type": "continuous",
      "functional_form": { "name": "cos", "weights": { "X1": 1.0, "X2": 0.5 } },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.2 }
    },
    "Y_sin": {
      "type": "continuous",
      "functional_form": { "name": "sin", "weights": { "X1": 0.8, "X2": 1.1 } },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.2 }
    }
  }
}
```

For `sigmoid`, `output_weight` (the post-tanh scaling $w_j$) and
the per-parent `weights` are sampled from the random-weight
distribution if omitted.

## Post-nonlinear transform

```json
{
  "simulation_params": { "n_samples": 300, "seed": 12 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "Y"],
    "edges": [["X", "Y"]]
  },
  "node_params": {
    "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Y": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X": 1.5 } },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.4 },
      "post_transform": { "name": "tanh" }
    }
  }
}
```

Replace `"tanh"` with any of `sin`, `cos`, `exp_neg_abs`,
`sqrt_abs`, `relu`, `sign`. The transform is applied element-wise
after the structural function and noise have been combined.

## Noise model variants

```json
{
  "simulation_params": { "n_samples": 250, "seed": 22 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "Y_add", "Y_mult", "Y_hetero"],
    "edges": [["X", "Y_add"], ["X", "Y_mult"], ["X", "Y_hetero"]]
  },
  "node_params": {
    "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Y_add": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
      "noise_model": { "name": "additive", "dist": "gamma", "shape": 2.0, "scale": 0.6 }
    },
    "Y_mult": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
      "noise_model": { "name": "multiplicative", "dist": "exponential", "scale": 1.0 }
    },
    "Y_hetero": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
      "noise_model": { "name": "heteroskedastic", "func": "abs_parent_plus_const" }
    }
  }
}
```

## Heavy-tailed and uniform additive noise

In addition to `gaussian`, `student_t`, `gamma`, and
`exponential`, the additive noise model accepts `laplace`,
`cauchy`, and `uniform`. All three are zero-centred and
parameterised by `scale`:

```json
{
  "simulation_params": { "n_samples": 400, "seed": 23 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "Y_lap", "Y_cau", "Y_uni"],
    "edges": [["X", "Y_lap"], ["X", "Y_cau"], ["X", "Y_uni"]]
  },
  "node_params": {
    "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Y_lap": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
      "noise_model": { "name": "additive", "dist": "laplace", "scale": 0.7 }
    },
    "Y_cau": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
      "noise_model": { "name": "additive", "dist": "cauchy", "scale": 0.3 }
    },
    "Y_uni": {
      "type": "continuous",
      "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
      "noise_model": { "name": "additive", "dist": "uniform", "scale": 1.0 }
    }
  }
}
```

Multiplicative noise also accepts `student_t`, `gamma`, and
`exponential` in addition to `gaussian`; gamma and exponential
factors are normalised to mean 1 to avoid biasing the structural
signal.

## Forced uniform marginals

Set `force_uniform_marginals` to make exogenous binary nodes draw
an exact 50/50 split and exogenous categorical nodes use exactly
equal counts per class (when their `p` / `probs` are not
explicitly set):

```json
{
  "simulation_params": {
    "n_samples": 200,
    "seed": 24,
    "force_uniform_marginals": true
  },
  "graph_params": {
    "type": "custom",
    "nodes": ["B", "C", "Y"],
    "edges": [["B", "Y"], ["C", "Y"]]
  },
  "node_params": {
    "B": { "type": "binary" },
    "C": { "type": "categorical", "cardinality": 4 },
    "Y": {
      "type": "continuous",
      "functional_form": { "name": "stratum_means" },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.3 }
    }
  }
}
```

This is convenient for constructing balanced benchmark scenarios
without worrying about small-sample fluctuations in the exogenous
strata.

## Binary child

```json
{
  "simulation_params": { "n_samples": 300, "seed": 33 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "Z", "B"],
    "edges": [["X", "B"], ["Z", "B"]]
  },
  "node_params": {
    "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Z": { "type": "binary", "distribution": { "name": "bernoulli", "p": 0.4 } },
    "B": {
      "type": "binary",
      "functional_form": { "name": "linear", "weights": { "X": 1.3, "Z": 0.9 } },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.5 }
    }
  }
}
```

## Categorical child (logistic softmax)

```json
{
  "simulation_params": { "n_samples": 400, "seed_structure": 40, "seed_data": 41 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "B", "C"],
    "edges": [["X", "C"], ["B", "C"]]
  },
  "node_params": {
    "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "B": { "type": "binary", "distribution": { "name": "bernoulli", "p": 0.5 } },
    "C": {
      "type": "categorical",
      "cardinality": 3,
      "categorical_model": {
        "name": "logistic",
        "intercepts": [0.0, 0.0, 0.0],
        "weights": {
          "X": [0.9, -0.2, -0.7],
          "B": [-0.4, 0.8, -0.3]
        }
      }
    }
  }
}
```

## Continuous to categorical (threshold)

```json
{
  "simulation_params": { "n_samples": 350, "seed": 50 },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "C"],
    "edges": [["X", "C"]]
  },
  "node_params": {
    "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "C": {
      "type": "categorical",
      "cardinality": 5,
      "categorical_model": {
        "name": "threshold",
        "weights": { "X": 1.0 },
        "thresholds": [-1.0, -0.2, 0.4, 1.1]
      }
    }
  }
}
```

To use fixed theoretical threshold placement:

```json
{
  "node_params": {
    "C": {
      "type": "categorical",
      "cardinality": 5,
      "categorical_model": {
        "name": "threshold",
        "threshold_loc": 0.0,
        "threshold_scale": 1.0
      }
    }
  }
}
```

## Categorical to continuous (stratum-specific means)

```json
{
  "simulation_params": { "n_samples": 300, "seed": 60 },
  "graph_params": {
    "type": "custom",
    "nodes": ["C1", "C2", "Y"],
    "edges": [["C1", "Y"], ["C2", "Y"]]
  },
  "node_params": {
    "C1": { "type": "categorical", "cardinality": 3 },
    "C2": { "type": "categorical", "cardinality": 2 },
    "Y": {
      "type": "continuous",
      "functional_form": {
        "name": "stratum_means",
        "default_mean": 0.0,
        "strata_means": {
          "C1=0|C2=0": -1.5,
          "C1=1|C2=0": 0.2,
          "C1=2|C2=1": 1.8
        }
      },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.15 }
    }
  }
}
```

## Mixed parents under `stratum_means`

When `stratum_means` has both categorical and metric parents, you
can supply `metric_weights` (a per-parent dict or a single number)
for the metric contribution. Omit it to have weights sampled from
the random-weight distribution.

```json
{
  "simulation_params": {
    "n_samples": 300,
    "seed": 61,
    "categorical_parent_metric_form_policy": "stratum_means"
  },
  "graph_params": {
    "type": "custom",
    "nodes": ["C", "X", "Y"],
    "edges": [["C", "Y"], ["X", "Y"]]
  },
  "node_params": {
    "C": { "type": "categorical", "cardinality": 3 },
    "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
    "Y": {
      "type": "continuous",
      "functional_form": {
        "name": "stratum_means",
        "strata_means": { "C=0": -1.0, "C=1": 0.0, "C=2": 1.5 },
        "metric_weights": { "X": 0.8 }
      },
      "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.2 }
    }
  }
}
```

## CI oracle output

```json
{
  "simulation_params": {
    "n_samples": 250,
    "seed": 70,
    "store_ci_oracle": true,
    "ci_oracle_max_cond_set": 2
  },
  "graph_params": {
    "type": "custom",
    "nodes": ["X", "Y", "Z"],
    "edges": [["X", "Z"], ["Y", "Z"]]
  }
}
```

When `store_ci_oracle` is enabled, `simulate()` also returns a
`ci_oracle` list with entries of the form:

```json
{
  "x": "X",
  "y": "Y",
  "conditioning_set": ["Z"],
  "is_independent": false
}
```

The oracle iterates over every ordered pair $(X, Y)$ and every
conditioning subset $S$ of size $\le$ `ci_oracle_max_cond_set`
(default `2`); both independent and dependent triples are
recorded.

```{seealso}
For the lazy alternative that satisfies the `cbcd.CITest`
Protocol, see [How-to: working with the CI oracle](ci_oracle.md).
```

## `simulate()` return value

`CausalDataGenerator(config).simulate()` returns a dict with the
following keys:

- `data` — a `pandas.DataFrame` of shape `(n_samples, n_nodes)`
  containing the simulated values.
- `dag` — a `networkx.DiGraph` representing the realised DAG.
- `parametrization` — a deep copy of the input config with every
  randomly-sampled value (weights, intercepts, thresholds, stratum
  means, noise parameters, marginals, derived `seed_structure` /
  `seed_data`, inferred node types) filled in. Suitable for
  round-tripping to JSON to reproduce the exact DGP.
- `ci_oracle` (only present when `store_ci_oracle = true`) — the
  list of oracle entries described above.
