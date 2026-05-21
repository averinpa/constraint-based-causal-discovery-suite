# SCM formulation

This page describes the mathematical structure that `dagsampler`
instantiates and the valid combinations of node types, structural
equations, and noise models.

## Notation

| Symbol | Meaning |
|---|---|
| $G = (V, E)$ | Directed acyclic graph with node set $V$ and edge set $E$. |
| $j \in V$ | A node (variable) in the graph. |
| $\mathrm{Pa}(j)$ | Set of parent nodes of $j$ in $G$. |
| $X_j$ | Random variable associated with node $j$. |
| $X_{\mathrm{Pa}(j)}$ | The vector of parent values for node $j$. |
| $K$ | Cardinality of a categorical variable (number of classes). |
| $\mathcal{D}_j$ | Marginal distribution of an exogenous continuous node $j$ (Gaussian, Student-*t*, gamma, or exponential). |
| $p_j$ | Success probability of an exogenous Bernoulli node $j$. |
| $\pi_{j,k}$ | Class probability for category $k$ of an exogenous categorical node $j$; satisfies $\sum_k \pi_{j,k} = 1$. |
| $f_j(\cdot)$ | Structural function mapping parents of $j$ to its mean signal. |
| $\epsilon_j$ | Noise term for node $j$ (additive, multiplicative, or heteroskedastic). |
| $\sigma_j(\cdot)$ | Heteroskedastic noise scale as a function of parents. |
| $z$ | Standard normal draw, $z \sim \mathcal{N}(0, 1)$. |
| $\eta_j$ | Latent signal for an endogenous binary node before the logistic link. |
| $\sigma(t)$ | Logistic sigmoid, $\sigma(t) = 1 / (1 + e^{-t})$. |
| $\ell_{jk}$ | Logit for class $k$ of an endogenous categorical node $j$. |
| $b_{jk}$ | Intercept for class $k$ in the logistic categorical model. |
| $g_{jpk}(X_p)$ | Contribution of parent $p$ to logit $\ell_{jk}$. |
| $\tau_{j1}, \dots, \tau_{j(K-1)}$ | Cut-points used by the threshold categorical model for node $j$. |
| $\perp\!\!\!\perp$ | Conditional independence (used in the CI oracle section). |

The simulator draws from two independent random streams: one seeds
the **data-generating process** (DAG topology, structural weights,
intercepts, thresholds, stratum means) and the other seeds the
**per-sample draws** (exogenous values, noise, Bernoulli /
categorical sampling). They are configured via `seed_structure`
and `seed_data` respectively, or jointly via a single `seed` (see
the seeding section in the
[configuration cookbook](../howto/config_cookbook.md)).

## Graph model

The simulator generates a DAG $G = (V, E)$ using one of:

- `custom` — user-defined node and edge sets.
- `random` — random acyclic edges over ordered nodes.

## Node types

Supported node types:

- Continuous.
- Binary (values in $\{0, 1\}$).
- Categorical (values in $\{0, \dots, K-1\}$, configurable
  cardinality $K$).

## Exogenous nodes ($\mathrm{Pa}(j) = \varnothing$)

Continuous exogenous node:

$$
X_j \sim \mathcal{D}_j
$$

where $\mathcal{D}_j$ is one of Gaussian, Student-*t*, gamma, or
exponential.

Binary exogenous node:

$$
X_j \sim \mathrm{Bernoulli}(p_j).
$$

Categorical exogenous node:

$$
X_j \sim \mathrm{Categorical}(\pi_{j,0}, \dots, \pi_{j,K-1}),
\quad \sum_k \pi_{j,k} = 1.
$$

## Endogenous continuous nodes

General form:

$$
X_j = f_j(X_{\mathrm{Pa}(j)}) + \epsilon_j.
$$

Supported structural forms $f_j$ are listed below.

**Linear.**

$$
f_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p.
$$

**Polynomial.**

$$
f_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p^{d_{jp}}.
$$

**Interaction.**

$$
f_j = w_j \prod_{p \in \mathrm{Pa}(j)} X_p.
$$

**Sigmoid (`tanh`).**

$$
f_j = w_j \cdot \tanh\!\left(
  \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p
\right).
$$

A smooth saturating nonlinearity — the weighted parent sum is
squashed by `tanh` and rescaled by an output weight $w_j$.

**Cosine.**

$$
f_j = \cos\!\left( \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p \right).
$$

**Sine.**

$$
f_j = \sin\!\left( \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p \right).
$$

The parent values are first combined linearly, then passed through
a periodic nonlinearity. Useful for stress-testing kernel-based
CI tests on oscillatory dependencies.

**Stratum-specific means** (categorical parents to continuous
child).

$$
f_j = \mu_{s(\mathbf{x}_{\mathrm{Pa}(j)})},
$$

where $s(\cdot)$ indexes the categorical parent stratum.

When `stratum_means` is used with **mixed parents** (at least one
categorical parent plus one or more metric parents), the structural
function combines a stratum mean with a linear contribution from
the metric parents:

$$
f_j = \mu_{s(\mathbf{x}_{\mathrm{cat}})}
    + \sum_{p \in \text{metric parents}} w_{jp} X_p.
$$

The metric weights can be set explicitly via
`functional_form.metric_weights` (a dict per parent or a single
number applied to all metric parents), or sampled from the
random-weight distribution if omitted.

### Random structural weights

When `weights` are omitted for `linear`, `polynomial`, or
`interaction`, the simulator samples weights from a configurable
interval:

$$
w \sim \mathrm{Uniform}(L, H),
$$

where $L =$ `random_weight_low` and $H =$ `random_weight_high`.

If `random_weight_min_abs = m > 0`, values in $(-m, m)$ are
excluded and weights are sampled from:

$$
[L, -m] \cup [m, H].
$$

This guarantees a minimum signal strength on every edge, giving
direct control over how strongly each parent influences its child
rather than letting random sampling produce effectively-zero
coefficients.

### Noise models

**Additive.**

$$
X_j = f_j + \epsilon_j.
$$

Additive noise distributions accepted under `noise_model.dist`:

- `gaussian` (parameter `std`)
- `student_t` (parameters `df`, `scale`)
- `gamma` (parameters `shape`, `scale`; centred to zero mean)
- `exponential` (parameter `scale`; centred to zero mean)
- `laplace` (parameter `scale`; zero-centred)
- `cauchy` (parameter `scale`; zero-centred, heavy-tailed)
- `uniform` (parameter `scale`; symmetric on
  $[-\text{scale}, \text{scale}]$)

**Multiplicative.**

$$
X_j = f_j \cdot (1 + \epsilon_j').
$$

The noise scales the structural signal, so the spread grows with
the magnitude of $f_j$. Multiplicative noise also supports
`gaussian`, `student_t`, `gamma`, and `exponential` distributions
for $\epsilon_j'$. Gamma and exponential factors are normalised to
mean 1 so the structural signal is not biased; all factors are
clipped to a small positive minimum for numerical safety.

**Heteroskedastic.**

$$
X_j = f_j + \sigma_j(X_{\mathrm{Pa}(j)})\, z,
\quad z \sim \mathcal{N}(0, 1).
$$

Additive Gaussian noise whose standard deviation depends on the
parent values. Registered $\sigma_j(\cdot)$ choices:

- `abs_first_parent` (default when `func` is omitted)
- `abs_parent_plus_const`
- `mean_abs_plus_const`

### Post-nonlinear transform

Any continuous endogenous node may apply a final element-wise
nonlinearity to its output after the structural function and noise
have been combined:

$$
X_j \leftarrow g(X_j),
$$

where $g$ is selected by `post_transform.name` from the registry:

| Name | Function |
|---|---|
| `tanh` | $\tanh(x)$ |
| `sin` | $\sin(x)$ |
| `cos` | $\cos(x)$ |
| `exp_neg_abs` | $\exp(-|x|)$ |
| `sqrt_abs` | $\sqrt{|x|}$ |
| `relu` | $\max(0, x)$ |
| `sign` | $\mathrm{sign}(x)$ |

The structural function and noise model determine the *signal*;
`post_transform` warps that signal afterwards. This is how the
literature typically realises *post-nonlinear* DGPs (Zhang &
Hyvärinen, 2009): e.g. $Y = \tanh(\text{linear}(X) + \epsilon)$.

## Endogenous binary nodes

Binary children use a logistic link on the latent signal:

$$
\eta_j = f_j(X_{\mathrm{Pa}(j)}) + \epsilon_j,
$$

$$
\Pr(X_j = 1 \mid X_{\mathrm{Pa}(j)}) = \sigma(\eta_j),
\quad \sigma(t) = \frac{1}{1 + e^{-t}},
$$

$$
X_j \sim \mathrm{Bernoulli}\!\left(\sigma(\eta_j)\right).
$$

## Endogenous categorical nodes

Two models are supported.

### Logistic (multinomial softmax)

$$
\ell_{jk} = b_{jk} + \sum_{p \in \mathrm{Pa}(j)} g_{jpk}(X_p),
$$

$$
\Pr(X_j = k \mid X_{\mathrm{Pa}(j)}) =
\frac{\exp(\ell_{jk})}{\sum_{m=0}^{K-1} \exp(\ell_{jm})}.
$$

The parent contribution $g_{jpk}$ depends on the parent type:

- continuous / binary parent: linear contribution per class —
  `weights[parent]` is a length-$K$ vector, one coefficient per
  child class.
- categorical parent: class-specific lookup via a parent-category
  weight matrix of shape $(K_{\text{parent}}, K)$ — one row per
  parent class, one column per child class.

### Threshold (continuous-to-categorical)

$$
s_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p,
$$

$$
X_j = \mathrm{digitize}\!\left(
  s_j + \varepsilon_j;\ \tau_{j1}, \dots, \tau_{j(K-1)}
\right),
\qquad
\varepsilon_j \sim \mathcal{N}\!\left(0,\ (\sigma_j \cdot \mathrm{sd}(s_j))^2\right).
$$

The latent noise $\varepsilon_j$ is governed by `noise_scale`
($\sigma_j$, **default `0.0`**). With $\sigma_j = 0$ the model is the
classic deterministic discretisation, $X_j = \mathrm{digitize}(s_j)$,
so a thresholded child is a pure function of its parents. With
$\sigma_j > 0$ it becomes an **ordered-probit** model: the child
retains idiosyncratic variation given its parents, which is required
for *faithful* alternatives when a thresholded variable is an effect in
a conditional-independence test. Because the noise SD is scaled by
$\mathrm{sd}(s_j)$, `noise_scale` is a **noise-to-signal ratio** — the
strength of the conditional dependence $X_j \mid \mathrm{Pa}(j)$ is held
constant across cardinalities $K$ and parent mechanisms (e.g.
$\sigma_j = 0.5 \Rightarrow$ signal-to-noise ratio $2$ in SD). The
default `0.0` keeps behaviour identical to releases $\le 0.1.0$.

If thresholds are not provided, defaults are set from a
theoretical Gaussian quantile grid, not from realised sample
quantiles. By default:

- `threshold_loc = 0.0`
- `threshold_scale` is sampled from $\mathrm{Uniform}(0.5, 2.0)$

Both can be overridden in config.

## Compatibility matrix

| Child type | Parent types | Structural model | Noise / link |
|---|---|---|---|
| Continuous | Any | `linear`, `polynomial`, `interaction`, `sigmoid`, `cos`, `sin`, `stratum_means` (+ optional `post_transform`) | `additive`, `multiplicative`, `heteroskedastic` |
| Binary | Any | `linear`, `polynomial`, `interaction`, `sigmoid`, `cos`, `sin`, `stratum_means` | Latent signal + noise, then logistic link and Bernoulli draw |
| Categorical | Any | `categorical_model = logistic` or `categorical_model = threshold` | Softmax sampling (logistic); threshold digitisation, deterministic or ordered-probit via `noise_scale` |

For random structural weights, additional controls are
`random_weight_low`, `random_weight_high`, and
`random_weight_min_abs`. The same `random_weight_min_abs`
exclusion is applied to auto-sampled categorical logistic weights
as well.

## Forced uniform marginals

Setting `simulation_params.force_uniform_marginals = true`
overrides the default randomised marginals on exogenous nodes:

- **Exogenous binary** (no explicit `p`): the simulator uses
  $p = 0.5$ *and* generates an exact balanced 0/1 split rather
  than sampling $X_j \sim \mathrm{Bernoulli}(0.5)$, eliminating
  small-sample fluctuations.
- **Exogenous categorical** (no explicit `probs`): the simulator
  uses uniform $\pi_{j,k} = 1/K$ *and* enforces equal counts per
  class (with a small remainder distributed at random).
- **Exogenous continuous**: unchanged — distributional parameters
  are still sampled or read from the config.

If `p` (binary) or `probs` (categorical) is explicitly provided,
the flag is ignored for that node and the config wins.

## Random node-type assignment

When `graph_params.type = "random"` and a node's `type` is not
pinned in `node_params`, the simulator samples a type per node
according to:

- `simulation_params.binary_proportion` (default $0.4$).
- `simulation_params.categorical_proportion` (default $0.0$).
- The remainder becomes continuous.

## Categorical parents in metric forms

Using categorical parents with `linear`, `polynomial`, or
`interaction` is blocked by default
(`categorical_parent_metric_form_policy = "error"`), because
treating category codes as metric values can distort the intended
DGP.

Setting `categorical_parent_metric_form_policy = "stratum_means"`
auto-redirects such cases to `stratum_means`. For mixed parents
(categorical + continuous / binary), the redirected
`stratum_means` uses

$$
f_j = \mu_{\text{cat-stratum}}
    + \sum_{p \in \text{metric parents}} w_p X_p,
$$

where categorical parents select the stratum mean and metric
parents contribute an additive linear term.

## Stratum-means reproducibility

For `stratum_means` with multiple categorical parents, all strata
are pre-enumerated and assigned means upfront, ensuring stable DGP
parameters even for rare or unseen strata in a particular sample.

## CI oracle (ground truth)

If `simulation_params.store_ci_oracle = true`, the simulator
stores conditional independence truth values from DAG d-separation:

$$
X \perp\!\!\!\perp Y \mid S
\;\;\iff\;\;
S \text{ is a d-separator of } X \text{ and } Y \text{ in } G,
$$

for conditioning sets up to `ci_oracle_max_cond_set`. The oracle
records, for every triple $(X, Y, S)$, whether the DAG structure
forces $X$ and $Y$ to be independent given $S$ — useful as ground
truth for evaluating CI tests.

The lazy alternative —
`CausalDataGenerator.as_ci_oracle()` — returns a
`DSeparationOracle` satisfying the `cbcd.CITest` Protocol, suitable
for direct use inside constraint-based algorithms; see
[How-to: working with the CI oracle](../howto/ci_oracle.md).

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
  (2nd ed.). Cambridge University Press.
- Zhang, K., & Hyvärinen, A. (2009). On the identifiability of the
  post-nonlinear causal model. In *Proceedings of UAI '09*,
  647–655.
