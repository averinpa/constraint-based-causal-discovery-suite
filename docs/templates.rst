Template Configurations
=======================

The ``dagsampler.templates`` module provides helper functions that build
``CausalDataGenerator`` configs for common DAG shapes — chains, forks,
colliders, and independent (no-edge) sets. They handle node-spec normalization,
seeding, and reasonable defaults so you can spin up a benchmark scenario in a
few lines.

All templates are re-exported from the package root:

.. code-block:: python

   from dagsampler import (
       CausalDataGenerator,
       chain_config,
       fork_config,
       collider_config,
       indep_config,
       independence_config,
   )

Common arguments
----------------

Most templates accept a ``mechanism`` argument selecting the structural form
applied to endogenous continuous/binary nodes:

* ``"linear"`` — weighted sum of parents with additive Gaussian noise.
* ``"sigmoid"`` — ``tanh`` of a weighted parent sum, plus additive Gaussian
  noise.
* ``"stratum_means"`` — required when any parent is categorical; the simulator
  falls back to ``"linear"`` automatically when no categorical parent is
  present.

Categorical endogenous nodes always use
``categorical_model = {"name": "logistic"}`` regardless of the ``mechanism``
argument.

The ``seed`` argument can be:

* an ``int`` — sets both ``seed_structure`` and ``seed_data`` to the same value;
* a ``dict`` like ``{"structure": 1, "data": 2}`` — sets each stream
  independently;
* ``None`` — leaves both streams unseeded.

The optional ``post_transform`` argument, when set, applies the named
post-nonlinear transform (any key from the post-transform registry — see
:doc:`formulations`) to every endogenous continuous node.

independence_config / indep_config
----------------------------------

A no-edge config: every node is exogenous. Useful for null scenarios in CI
benchmarks.

.. code-block:: python

   cfg = independence_config(
       var_specs=[
           {"name": "X", "type": "continuous"},
           {"name": "B", "type": "binary"},
           {"name": "C", "type": "categorical", "cardinality": 4},
       ],
       n_samples=300,
       seed=7,
       force_uniform=True,
   )

``indep_config`` is a backward-compatible shorthand: if ``var_specs`` is
omitted, it generates ``n_vars`` variables of a single ``node_type`` named
``{prefix}0..{prefix}{n_vars-1}``.

.. code-block:: python

   cfg = indep_config(n_vars=5, node_type="binary", n_samples=300, seed=7)

When ``force_uniform=True`` (default), the config sets
``simulation_params.force_uniform_marginals = true`` so binary nodes get an
exact 50/50 split and categorical nodes get equal class counts.

chain_config
------------

A chain ``var_specs[0] -> var_specs[1] -> ... -> var_specs[-1]``. The first
node is exogenous; each subsequent node is endogenous with the previous as its
single parent.

.. code-block:: python

   cfg = chain_config(
       var_specs=[
           {"name": "X", "type": "continuous"},
           {"name": "M", "type": "continuous"},
           {"name": "Y", "type": "continuous"},
       ],
       mechanism="linear",
       n_samples=400,
       seed={"structure": 11, "data": 12},
       post_transform="tanh",
   )

fork_config
-----------

A fork ``root -> left``, ``root -> right`` (a common cause / confounder
pattern). ``var_specs`` is a dict with keys ``"root"``, ``"left"``,
``"right"``.

.. code-block:: python

   cfg = fork_config(
       var_specs={
           "root":  {"name": "Z", "type": "continuous"},
           "left":  {"name": "X", "type": "continuous"},
           "right": {"name": "Y", "type": "continuous"},
       },
       mechanism="linear",
       n_samples=300,
       seed=42,
   )

collider_config
---------------

A v-structure ``left -> collider``, ``right -> collider``. ``left`` and
``right`` are exogenous; ``collider`` has both as parents.

.. code-block:: python

   cfg = collider_config(
       var_specs={
           "left":     {"name": "X", "type": "continuous"},
           "right":    {"name": "Y", "type": "continuous"},
           "collider": {"name": "Z", "type": "continuous"},
       },
       mechanism="linear",
       n_samples=300,
       seed=42,
   )

Putting it together
-------------------

Each template returns a plain ``dict`` ready to hand to
``CausalDataGenerator``:

.. code-block:: python

   from dagsampler import CausalDataGenerator, chain_config

   cfg = chain_config(
       var_specs=[
           {"name": "X", "type": "continuous"},
           {"name": "Y", "type": "continuous"},
       ],
       mechanism="linear",
       n_samples=200,
       seed=0,
   )
   result = CausalDataGenerator(cfg).simulate()
