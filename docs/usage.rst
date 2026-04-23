Usage
=====

Install
-------

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install "dagsampler @ git+https://github.com/averinpa/dagsampler.git"

Python API
----------

.. code-block:: python

   from dagsampler import CausalDataGenerator

   config = {
       "simulation_params": {"n_samples": 200, "seed": 42},
       "graph_params": {
           "type": "custom",
           "nodes": ["X", "Y", "Z1"],
           "edges": [["X", "Z1"], ["Y", "Z1"]],
       },
   }

   result = CausalDataGenerator(config).simulate()

Random weights away from zero
-----------------------------

To control the minimum signal strength on every edge — preventing randomly
sampled weights from being effectively zero:

.. code-block:: python

   config = {
       "simulation_params": {
           "n_samples": 500,
           "seed_structure": 11,
           "seed_data": 12,
           "random_weight_low": -1.5,
           "random_weight_high": 1.5,
           "random_weight_min_abs": 0.1,
       },
       "graph_params": {"type": "random", "n_nodes": 8, "edge_prob": 0.25},
   }

This samples random structural weights from
``[-1.5, -0.1] U [0.1, 1.5]``.

Template configurations
-----------------------

For common DAG shapes, the package ships helper functions that build a config
dict for you. See :doc:`templates` for the full reference.

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

CLI
---

.. code-block:: bash

   dagsampler-generate \
     --config config.json \
     --output dataset.csv \
     --params-out params.json \
     --edges-out edges.json
