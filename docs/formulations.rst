Model Formulations
==================

This page describes the mathematical structure implemented by the simulator and
the valid combinations of node types, structural equations, and noise models.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Symbol
     - Meaning
   * - :math:`G = (V, E)`
     - Directed acyclic graph with node set :math:`V` and edge set :math:`E`.
   * - :math:`j \in V`
     - A node (variable) in the graph.
   * - :math:`\mathrm{Pa}(j)`
     - Set of parent nodes of :math:`j` in :math:`G`.
   * - :math:`X_j`
     - Random variable associated with node :math:`j`.
   * - :math:`X_{\mathrm{Pa}(j)}`
     - The vector of parent values for node :math:`j`.
   * - :math:`K`
     - Cardinality of a categorical variable (number of classes).
   * - :math:`\mathcal{D}_j`
     - Marginal distribution of an exogenous continuous node :math:`j`
       (Gaussian, Student-t, Gamma, or Exponential).
   * - :math:`p_j`
     - Success probability of an exogenous Bernoulli node :math:`j`.
   * - :math:`\pi_{j,k}`
     - Class probability for category :math:`k` of an exogenous categorical
       node :math:`j`; satisfies :math:`\sum_k \pi_{j,k} = 1`.
   * - :math:`f_j(\cdot)`
     - Structural function mapping parents of :math:`j` to its mean signal.
   * - :math:`\epsilon_j`
     - Noise term for node :math:`j` (additive, multiplicative, or
       heteroskedastic).
   * - :math:`w_{jp}`
     - Structural weight from parent :math:`p` to child :math:`j`.
   * - :math:`d_{jp}`
     - Polynomial degree applied to parent :math:`p` in the structural form
       for child :math:`j`.
   * - :math:`w_j`
     - Single interaction weight in the ``interaction`` form.
   * - :math:`\mu_s`
     - Mean assigned to categorical-parent stratum :math:`s` in the
       ``stratum_means`` form.
   * - :math:`s(\mathbf{x}_{\mathrm{Pa}(j)})`
     - Stratum index determined by the categorical parent values.
   * - :math:`L, H`
     - Lower / upper bounds for random structural weight sampling
       (``random_weight_low``, ``random_weight_high``).
   * - :math:`m`
     - Near-zero exclusion radius for random weights
       (``random_weight_min_abs``).
   * - :math:`\sigma_j(\cdot)`
     - Heteroskedastic noise scale as a function of parents.
   * - :math:`z`
     - Standard normal draw, :math:`z \sim \mathcal{N}(0, 1)`.
   * - :math:`\eta_j`
     - Latent signal for an endogenous binary node before the logistic link.
   * - :math:`\sigma(t)`
     - Logistic sigmoid, :math:`\sigma(t) = 1 / (1 + e^{-t})`.
   * - :math:`\ell_{jk}`
     - Logit for class :math:`k` of an endogenous categorical node :math:`j`.
   * - :math:`b_{jk}`
     - Intercept for class :math:`k` in the logistic categorical model.
   * - :math:`g_{jpk}(X_p)`
     - Contribution of parent :math:`p` to logit :math:`\ell_{jk}`.
   * - :math:`\tau_{j1}, \dots, \tau_{j(K-1)}`
     - Cut-points used by the threshold categorical model for node :math:`j`.
   * - :math:`\perp\!\!\!\perp`
     - Conditional independence (used in the CI oracle section).

The simulator draws from two independent random streams: one seeds the
**data-generating process** (DAG topology, structural weights, intercepts,
thresholds, stratum means) and the other seeds the **per-sample draws**
(exogenous values, noise, Bernoulli/categorical sampling). They are configured
via ``seed_structure`` and ``seed_data`` respectively, or jointly via a single
``seed`` (see the Seeding section in :doc:`config_examples`).

Graph Model
-----------

The simulator generates a DAG :math:`G = (V, E)` using one of:

* ``custom``: user-defined node and edge sets
* ``random``: random acyclic edges over ordered nodes

Node Types
----------

Supported node types:

* Continuous
* Binary (values in :math:`\{0, 1\}`)
* Categorical (values in :math:`\{0, \dots, K-1\}`, configurable cardinality :math:`K`)

Exogenous Nodes (:math:`\mathrm{Pa}(j)=\varnothing`)
----------------------------------------------------

Continuous exogenous node:

.. math::

   X_j \sim \mathcal{D}_j

where :math:`\mathcal{D}_j` is one of Gaussian, Student-t, Gamma, or Exponential.
*Intuition:* draw each value of :math:`X_j` independently from the chosen
marginal distribution.

Binary exogenous node:

.. math::

   X_j \sim \mathrm{Bernoulli}(p_j)

*Intuition:* a coin flip that returns 1 with probability :math:`p_j` and 0
otherwise.

Categorical exogenous node:

.. math::

   X_j \sim \mathrm{Categorical}(\pi_{j,0}, \dots, \pi_{j,K-1}), \quad \sum_k \pi_{j,k}=1

*Intuition:* a weighted dice roll that returns class :math:`k` with
probability :math:`\pi_{j,k}`.

Endogenous Continuous Nodes
---------------------------

General form:

.. math::

   X_j = f_j(X_{\mathrm{Pa}(j)}) + \epsilon_j

*Intuition:* the value of :math:`X_j` is a deterministic function of its
parents plus an independent noise draw.

Supported structural forms :math:`f_j`:

Linear:

.. math::

   f_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p

*Intuition:* a weighted sum of the parent values.

Polynomial:

.. math::

   f_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p^{d_{jp}}

*Intuition:* a weighted sum where each parent is first raised to its own
fixed power.

Interaction:

.. math::

   f_j = w_j \prod_{p \in \mathrm{Pa}(j)} X_p

*Intuition:* the product of all parent values, scaled by a single weight.

Sigmoid (tanh):

.. math::

   f_j = w_j \cdot \tanh\!\left( \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p \right)

*Intuition:* a smooth saturating nonlinearity — the weighted parent sum is
squashed by ``tanh`` and rescaled by an output weight :math:`w_j`.

Cosine:

.. math::

   f_j = \cos\!\left( \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p \right)

Sine:

.. math::

   f_j = \sin\!\left( \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p \right)

*Intuition:* the parent values are first combined linearly, then passed through
a periodic nonlinearity. Useful for stress-testing kernel-based CI tests on
oscillatory dependencies.

Stratum-specific means (categorical parents to continuous child):

.. math::

   f_j = \mu_{s(\mathbf{x}_{\mathrm{Pa}(j)})}

where :math:`s(\cdot)` indexes the categorical parent stratum.
*Intuition:* look up a pre-assigned mean for the combination of categorical
parent values observed at this row.

When ``stratum_means`` is used with **mixed parents** (at least one categorical
parent plus one or more metric parents), the structural function combines a
stratum mean with a linear contribution from the metric parents:

.. math::

   f_j = \mu_{s(\mathbf{x}_{\mathrm{cat}})}
       + \sum_{p \in \text{metric parents}} w_{jp} X_p

The metric weights can be set explicitly via ``functional_form.metric_weights``
(a dict per parent or a single number applied to all metric parents), or
sampled from the random-weight distribution if omitted.

Random structural weights
-------------------------

When ``weights`` are omitted for ``linear``, ``polynomial``, or ``interaction``,
the simulator samples weights from a configurable interval:

.. math::

   w \sim \mathrm{Uniform}(L, H)

where ``L=random_weight_low`` and ``H=random_weight_high``.
*Intuition:* when you don't pin a weight, it's drawn uniformly between
:math:`L` and :math:`H`.

If ``random_weight_min_abs = m > 0``, values in :math:`(-m, m)` are excluded
and weights are sampled from:

.. math::

   [L, -m] \cup [m, H]

This guarantees a minimum signal strength on every edge, giving you direct
control over how strongly each parent influences its child rather than letting
random sampling produce effectively-zero coefficients.
*Intuition:* every edge contributes at least :math:`m` worth of signal, so
no parent ends up silently muted by the random draw.

Noise models:

Additive:

.. math::

   X_j = f_j + \epsilon_j

*Intuition:* the noise is added on top of the structural signal.

Additive noise distributions accepted under ``noise_model.dist``:

* ``gaussian`` (parameter ``std``)
* ``student_t`` (parameters ``df``, ``scale``)
* ``gamma`` (parameters ``shape``, ``scale``; centered to zero mean)
* ``exponential`` (parameter ``scale``; centered to zero mean)
* ``laplace`` (parameter ``scale``; zero-centered)
* ``cauchy`` (parameter ``scale``; zero-centered, heavy-tailed)
* ``uniform`` (parameter ``scale``; symmetric on :math:`[-\text{scale}, \text{scale}]`)

Multiplicative:

.. math::

   X_j = f_j \cdot (1 + \epsilon_j')

*Intuition:* the noise scales the structural signal, so the spread grows
with the magnitude of :math:`f_j`.

Multiplicative noise also supports ``gaussian``, ``student_t``, ``gamma``,
and ``exponential`` distributions for :math:`\epsilon_j'`. Gamma and
exponential factors are normalized to mean 1 so the structural signal is not
biased; all factors are clipped to a small positive minimum for numerical
safety.

Heteroskedastic:

.. math::

   X_j = f_j + \sigma_j(X_{\mathrm{Pa}(j)}) z, \quad z \sim \mathcal{N}(0,1)

*Intuition:* additive Gaussian noise whose standard deviation depends on
the parent values.

with registered :math:`\sigma_j(\cdot)` choices:

* ``abs_first_parent`` (default when ``func`` is omitted)
* ``abs_parent_plus_const``
* ``mean_abs_plus_const``

Post-nonlinear transform
------------------------

Any continuous endogenous node may apply a final element-wise nonlinearity to
its output after the structural function and noise have been combined:

.. math::

   X_j \leftarrow g(X_j)

where :math:`g` is selected by ``post_transform.name`` from the registry:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Name
     - Function
   * - ``tanh``
     - :math:`\tanh(x)`
   * - ``sin``
     - :math:`\sin(x)`
   * - ``cos``
     - :math:`\cos(x)`
   * - ``exp_neg_abs``
     - :math:`\exp(-|x|)`
   * - ``sqrt_abs``
     - :math:`\sqrt{|x|}`
   * - ``relu``
     - :math:`\max(0, x)`
   * - ``sign``
     - :math:`\mathrm{sign}(x)`

*Intuition:* the structural function and noise model determine the *signal*;
``post_transform`` warps that signal afterwards. This is how the literature
typically realizes "post-nonlinear" DGPs (e.g., :math:`Y = \tanh(\text{linear}(X) + \epsilon)`).

Endogenous Binary Nodes
-----------------------

Binary children use a logistic link on the latent signal:

.. math::

   \eta_j = f_j(X_{\mathrm{Pa}(j)}) + \epsilon_j

*Intuition:* build a continuous latent score from the parents and a noise
term.

.. math::

   \Pr(X_j=1 \mid X_{\mathrm{Pa}(j)}) = \sigma(\eta_j), \quad
   \sigma(t)=\frac{1}{1+e^{-t}}

*Intuition:* squash the latent score into a probability between 0 and 1.

.. math::

   X_j \sim \mathrm{Bernoulli}\!\left(\sigma(\eta_j)\right)

*Intuition:* flip a biased coin with that probability to decide whether
:math:`X_j` is 0 or 1.

Endogenous Categorical Nodes
----------------------------

Two models are supported.

1. Logistic (multinomial softmax)

.. math::

   \ell_{jk} = b_{jk} + \sum_{p \in \mathrm{Pa}(j)} g_{jpk}(X_p)

*Intuition:* compute one logit per class as an intercept plus parent
contributions.

.. math::

   \Pr(X_j=k \mid X_{\mathrm{Pa}(j)}) =
   \frac{\exp(\ell_{jk})}{\sum_{m=0}^{K-1} \exp(\ell_{jm})}

*Intuition:* convert the logits into class probabilities via softmax, then
sample a class from that distribution.

where :math:`g_{jpk}` depends on parent type:

* continuous/binary parent: linear contribution per class — ``weights[parent]``
  is a length-:math:`K` vector, one coefficient per child class.
* categorical parent: class-specific lookup via a parent-category weight matrix
  of shape :math:`(K_{\text{parent}}, K)` — one row per parent class, one column
  per child class.

2. Threshold (continuous-to-categorical)

.. math::

   s_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p

*Intuition:* form a continuous score from a weighted sum of parents.

.. math::

   X_j = \mathrm{digitize}(s_j; \tau_{j1}, \dots, \tau_{j(K-1)})

*Intuition:* assign a class based on which bin the score falls into,
defined by the cut-points :math:`\tau_{j1}, \dots, \tau_{j(K-1)}`.

If thresholds are not provided, defaults are set from a theoretical Gaussian
quantile grid, not from realized sample quantiles. By default:

* ``threshold_loc = 0.0``
* ``threshold_scale`` is sampled from ``Uniform(0.5, 2.0)``

You can override both explicitly in config.

Compatibility Matrix
--------------------

.. list-table:: Supported combinations
   :header-rows: 1
   :widths: 16 24 28 32

   * - Child type
     - Parent types
     - Structural model
     - Noise / link
   * - Continuous
     - Continuous, binary, categorical, or mixed
     - ``linear``, ``polynomial``, ``interaction``, ``sigmoid``, ``cos``,
       ``sin``, ``stratum_means`` (+ optional ``post_transform``)
     - ``additive``, ``multiplicative``, ``heteroskedastic``
   * - Binary
     - Continuous, binary, categorical, or mixed
     - ``linear``, ``polynomial``, ``interaction``, ``sigmoid``, ``cos``,
       ``sin``, ``stratum_means``
     - Latent signal + noise, then logistic link and Bernoulli draw
   * - Categorical
     - Continuous, binary, categorical, or mixed
     - ``categorical_model = logistic`` or ``categorical_model = threshold``
     - Softmax sampling (logistic) or threshold digitization

For random structural weights, additional controls are:
``random_weight_low``, ``random_weight_high``, and ``random_weight_min_abs``.
The same ``random_weight_min_abs`` exclusion is applied to auto-sampled
categorical logistic weights as well.

Forced uniform marginals
------------------------

Set ``simulation_params.force_uniform_marginals = true`` to override the
default randomized marginals on exogenous nodes:

* **Exogenous binary** (no explicit ``p``): the simulator uses :math:`p = 0.5`
  *and* generates an exact balanced 0/1 split rather than sampling
  :math:`X_j \sim \mathrm{Bernoulli}(0.5)`, eliminating small-sample
  fluctuations.
* **Exogenous categorical** (no explicit ``probs``): the simulator uses
  uniform :math:`\pi_{j,k} = 1/K` *and* enforces equal counts per class
  (with a small remainder distributed at random).
* **Exogenous continuous**: unchanged — distributional parameters are still
  sampled or read from the config.

If ``p`` (binary) or ``probs`` (categorical) is explicitly provided, the flag
is ignored for that node and your config wins.

Random node-type assignment
---------------------------

When ``graph_params.type = "random"`` and a node's ``type`` is not pinned in
``node_params``, the simulator samples a type per node according to:

* ``simulation_params.binary_proportion`` (default ``0.4``)
* ``simulation_params.categorical_proportion`` (default ``0.0``)
* the remainder becomes continuous

Categorical parents in metric forms
-----------------------------------

Using categorical parents with ``linear``, ``polynomial``, or ``interaction``
is blocked by default (``categorical_parent_metric_form_policy = "error"``),
because treating category codes as metric values can distort the intended DGP.

Set ``categorical_parent_metric_form_policy = "stratum_means"`` to auto-redirect
such cases to ``stratum_means``.

For mixed parents (categorical + continuous/binary), redirected ``stratum_means``
uses:

.. math::

   f_j = \mu_{\text{cat-stratum}} + \sum_{p \in \text{metric parents}} w_p X_p

where categorical parents select the stratum mean and metric parents contribute
an additive linear term.

Stratum means reproducibility
-----------------------------

For ``stratum_means`` with multiple categorical parents, all strata are
pre-enumerated and assigned means upfront, ensuring stable DGP parameters even
for rare/unseen strata in a particular sample.

CI Oracle (Ground Truth)
------------------------

If ``simulation_params.store_ci_oracle = true``, the simulator stores conditional
independence truth values from DAG d-separation:

.. math::

   X \perp\!\!\!\perp Y \mid S \iff S \text{ is a d-separator of } X \text{ and } Y \text{ in } G

for conditioning sets up to ``ci_oracle_max_cond_set``.
*Intuition:* the oracle records, for every triple :math:`(X, Y, S)`, whether
the DAG structure forces :math:`X` and :math:`Y` to be independent given
:math:`S` — useful as ground truth for evaluating CI tests.
