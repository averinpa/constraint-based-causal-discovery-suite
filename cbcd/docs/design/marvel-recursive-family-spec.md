# Implementation spec: MARVEL and the recursive-causal-discovery family

Native cbcd implementation of **MARVEL** (cell 1 speed champion) and its family (RCD line,
Mokhtarian/Akbari et al.). Sources in corpus: `algo81` (MARVEL, KDD-CD 2021), `algo80` (extended
framework + the `RCD` Python package, §8). Pure constraint-based, so in-scope for cbcd.

## Why: two cells from one core
The recursive Markov-boundary elimination core is shared across the family, so one implementation
effort yields best-in-class **CI-test-efficiency champions** for two cells:
- **MARVEL** -> cell 1 (full - iid - sufficient -> CPDAG). Speed champion vs the PC-stable anchor.
- **L-MARVEL** (Akbari et al. 2021, NeurIPS) -> cell 2 (full - iid - non-sufficient -> PAG). Recursive
  FCI-analog with latents + selection bias. Speed champion vs the FCI anchor.
- **RSL** (2022, AAAI; RSL-D diamond-free / RSL-W bounded-clique) -> cell 1 under structural side
  info, even fewer tests. Optional.
- **ROL** (2023, AAAI) -> cell 2, ordering/RL-based. Skip for now (RL machinery is out of the clean
  constraint-based spirit; revisit only if a benchmark needs it).

## The MARVEL algorithm (algo81, Algorithm 1)
**Input:** variables V, a swappable CI test, and the Markov boundaries `Mb_X` for all X.
**Output:** the CPDAG (essential graph).

Recursively eliminate a *removable* variable (one whose removal preserves Markov + faithfulness of the
remaining subgraph; Def 3 / Remark 4). Per iteration:

1. Sort remaining V ascending by `|Mb_X|` -> ordering I. Scan for the **first removable** `X`.
   Removability (Thm 5) is tested via CI tests on subsets of `Mb_X`:
   - **Neighbors + co-parents** (Lemma 7): for each `Y in Mb_X`, `Y` is a neighbor iff
     `X not-indep Y | S` for all `S subset Mb_X\{Y}`; otherwise a co-parent (record the sepset).
   - **Condition 1** (Lemma 9): `Z not-indep W | S ∪ {X}` for all `W,Z in N_X`, `S ⊆ Mb_X\{Z,W}`.
   - If Cond 1 holds: find v-structures `V_X^pa` (Lemma 8), then **Condition 2** (Lemma 10):
     `Z not-indep T | S ∪ {X,Y}` for each `(X->Y<-T) in V_X^pa`. Both hold -> `X` removable.
2. For the removable `X`: add undirected edges `X—N_X`; orient v-structures `X->Y<-T`; orient
   remaining incident edges as in-going toward `X`. (Thm 13: yields the true skeleton + v-structures.)
3. Remove `X`; **update Markov boundaries** of `N_X` (Eq 8: drop `Y,Z` from each other's Mb if
   `Y indep Z | Mb_W\{X,Y,Z}`, `W` the smaller-Mb endpoint).
4. After all removed: keep skeleton + v-structures, apply **Meek rules** -> CPDAG.

**Why it's cheaper than PC (the whole point):**
- Removable vars have `|Mb_X| <= Δ_in` (Lemma 12), so conditioning sets stay small; we never touch a
  var with `Mb > Δ_in`. Fewer *and* more powerful CI tests than PC's global subset search.
- Sort-by-Mb + eliminate; duplicate-CI-test avoidance (§3.4) via caching.
- Cost to test one var: `O(|Mb_X|^2 · 2^|Mb_X|)` unique CI tests (Prop 11), but bounded by `Δ_in`.

## cbcd integration (reuse, do not rebuild)
Everything MARVEL needs already exists in cbcd:
- **Markov boundaries** -> `cbcd/mb.py` (`grow_shrink` for the initial all-variable Mb pass; also
  `iamb`/`inter_iamb`). This is the initialization input.
- **CI test** -> the swappable `CITest` protocol (`cbcd/citest/`), string or instance.
- **Meek rules** -> `cbcd/rules.py` (as used by `pc`).
- **CPDAG** -> `cbcd/graph/cpdag.py`; collider/orientation helpers `cbcd/collider.py`.
- **Instrumentation** -> `cbcd/recording.py` (`RunRecorder`) + `cbcd/citest/cached.py`. Critical: this
  is how we *measure* the efficiency win (n_ci_total / n_ci_unique / max_depth).

## Proposed API
`cbcd/algorithms/marvel.py`:
```
def marvel(data_or_citest, *, ci_test="fisherz", mb_algo="grow_shrink",
           recorder=None) -> CPDAG
```
Matches existing top-level algo conventions (`pc`, `fci`). Export from `cbcd/algorithms/__init__.py`
and `cbcd/__init__.py`. Internal helpers: `_removable`, `_neighbors_coparents` (Lemma 7),
`_vstructures` (Lemma 8), `_cond1`/`_cond2` (Lemmas 9/10), `_update_mb` (Eq 8), reusing `rules.meek`.

## Validation / parity gate
Two independent checks (extend `parity/` with `parity/marvel/`):
1. **Oracle correctness = PC-stable.** On random DAGs with a d-sep oracle, `marvel()` must return the
   **identical CPDAG** to `cbcd.pc()`. (Same equivalence class by Thm 13.) This is the sound+complete
   gate; reuse the oracle machinery in `tests/oracle.py`.
2. **Efficiency win, recorded.** Same runs, compare `RunRecorder` CI-test counts: MARVEL n_ci should
   be substantially below PC-stable n_ci for the same CPDAG. This *is* the reason to add MARVEL, so it
   must be measured, not asserted.
3. **External reference (optional but ideal):** the **RCD** Python package (algo80 §8) implements
   MARVEL and the family — parity `cbcd.marvel()` vs `RCD` endpoint matrices, same pattern as the
   causal-learn/tigramite harnesses (dev-only dep, not pinned).

## Assumptions / caveats
- Requires Markov + faithfulness (same as PC) and *correct* Markov boundaries. A finite-sample MB
  error propagates; the initial `grow_shrink` pass is the sensitive step. Note this in the benchmark.
- The `2^|Mb|` factor is bounded by `Δ_in` for removable vars, so MARVEL wins on **sparse / bounded
  in-degree** graphs (its target) and degrades toward PC on dense ones. State the regime.

## Build order (family)
1. **MARVEL** native (cell 1) + parity vs PC-stable + CI-count benchmark. <- start here.
2. **L-MARVEL** (cell 2): reuse the recursive core; add latent/selection handling -> PAG. Parity vs
   `cbcd.fci()` (oracle m-sep) + RCD. This is the cell-2 speed champion, ~free once MARVEL exists.
3. **RSL** (cell 1, side-info) optional; **ROL** deferred.

Related: [[algorithm-cell-map]] (cell 1 = PC-stable anchor + MARVEL champion; cell 2 = FCI anchor +
L-MARVEL champion).
