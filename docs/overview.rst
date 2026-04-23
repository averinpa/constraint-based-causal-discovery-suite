Overview
========

The ``dagsampler`` package provides a configurable causal data generator.

Main features:

* ``custom`` and ``random`` DAG generation
* Continuous, binary, and categorical variables
* Structural forms: ``linear``, ``polynomial``, ``interaction``, ``sigmoid``,
  ``cos``, ``sin``, ``stratum_means``
* Optional element-wise ``post_transform`` (``tanh``, ``sin``, ``cos``,
  ``exp_neg_abs``, ``sqrt_abs``, ``relu``, ``sign``)
* Noise models: ``additive`` (``gaussian``, ``student_t``, ``gamma``,
  ``exponential``, ``laplace``, ``cauchy``, ``uniform``), ``multiplicative``,
  and ``heteroskedastic``
* Cross-type mechanisms: continuous → categorical (``threshold``) and
  categorical → continuous (``stratum_means``, with optional metric parents)
* Random structural weight controls:

  * ``random_weight_low``
  * ``random_weight_high``
  * ``random_weight_min_abs`` (excludes near-zero coefficients)
* ``force_uniform_marginals`` for balanced exogenous binary / categorical draws
* Template helpers for chain, fork, collider, and independence configurations
* Reproducible sampling with separate structure/data seeds
* Optional d-separation CI oracle output
