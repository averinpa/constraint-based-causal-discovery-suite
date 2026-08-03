# Implementation spec: L-MARVEL (cell 2 champion)

Native cbcd implementation of **L-MARVEL** (Akbari, Mokhtarian, Ghassami, Kiyavash, NeurIPS 2021;
corpus `algo147`; framework `algo80` §4.3/§5.2). Cell 2 = full · iid · non-sufficient -> **PAG**.
The recursive-elimination CI-test-efficiency champion vs the FCI anchor. Pure constraint-based.

## Relationship to MARVEL
Same recursive Markov-boundary elimination core as `marvel.py`, but:
- **Removability test is the MAG version** (Theorem 2 below), not MARVEL's Meek conditions.
- **Output is a PAG oriented by FCI rules R0-R10** (reuse `cbcd/fci.py` orientation), not a Meek CPDAG.
- Handles **latent confounders**. (Selection bias is an extension, see Scope.)
Refactor the shared pieces (`FindAdjacent`, `UpdateMb`, the sort-by-MB elimination loop) out of
`marvel.py` into helpers both can call; L-MARVEL swaps in its own `IsRemovable` + PAG orientation.

## Algorithm (algo147, Alg 1 + Alg 2 + Thm 2)
**Input:** observed variables O, a CI test. **Output:** PAG.
Maintain `A` = adjacency + separating-set store (exactly the object `fci`'s skeleton phase produces).

1. `Mb_O <- ComputeMb(O)` (grow_shrink/iamb; the TC characterization `X not-indep Y | rest` is correct
   for MAGs). Initialize `A`: for any `X` and `Y not in Mb(X)`, record `Mb(X)` as a sepset of `X,Y`.
2. Recurse `L-MARVEL(V, Mb_V, A)`:
   - if `|V|==1` return `A`.
   - sort `V` ascending by `|Mb_V(X)|`.
   - for each `X_i` in that order until the first removable one:
     - **FindAdjacent(X_i):** for each `Y in Mb_V(X_i)`, `Y in Adj(X_i)` iff `X_i not-indep Y | W` for
       *every* `W subset Mb_V(X_i)\{Y}`; otherwise record the separating `W` into `A`. (brute force,
       <= `|Mb|*2^(|Mb|-1)` CI tests; short-circuit on the first separator.)
     - **IsRemovable(X_i)** (Theorem 2 / Alg 2): for every `Y in Adj(X_i)` and `Z in Mb_V(X_i)`, at
       least one must hold, else NOT removable:
       - **C1:** `exists W subset Mb(X)\{Y,Z}: Y indep Z | W`.
       - **C2:** `forall W subset Mb(X)\{Y,Z}: Y not-indep Z | W ∪ {X}`.
       (Check C1 then C2, in that order — the paper found it more accurate.)
     - if removable: **UpdateMb** (remove `X` from all MBs; for `Y,Z in Mb(X)`, drop them from each
       other's MB iff `Y indep Z | Mb(Z)\{X,Y,Z}` — one CI test, smaller-MB conditioning set), then
       recurse on `V\{X}`.
3. Build PAG from `A`'s adjacencies; **orient maximally with FCI rules R0-R10** using the sepsets in
   `A` (reuse `cbcd/fci.py`'s orientation path — this substitution is *correct* here, unlike MARVEL,
   because FCI orientation is sepset-driven and `A` supplies the sepsets).

**Soundness/completeness (Thm 3):** given the CI oracle, output is the PAG of the MEC. Near-optimal
CI count: `O(n^2 + n·Δ⁺²·2^Δ⁺)` vs the proven lower bound `Ω(n^2 + n·Δ⁺·2^Δ⁺)` (factor <= n).

## cbcd integration (reuse, don't rebuild)
- MBs: `cbcd/mb.py` (`grow_shrink`/`iamb`).
- Recursive core: shared helpers refactored from `cbcd/algorithms/marvel.py`.
- Orientation + PAG: `cbcd/fci.py` orientation (R0-R10) + `cbcd/graph/pag.py`.
- CI test / caching / instrumentation: `CITest` protocol, `CachedCITest`, `RunRecorder` (measure the win).
- Background knowledge: thread `background` into the FCI orientation exactly as `fci` does (do this
  from the start — we just added BK to MARVEL; keep parity of features across the family).

## API
`cbcd/algorithms/lmarvel.py`:
```
def lmarvel(data, *, ci_test="fisherz", alpha=0.05, mb_algo="grow_shrink",
            max_cond_set=None, background=None, var_names=None,
            recorder=None, run_id=None) -> PAG
```
Match the `marvel`/`fci` signature conventions. Export from `cbcd/algorithms/__init__.py` + `cbcd/__init__.py`.

## Validation / parity gate
1. **Oracle correctness = FCI.** On random MAGs/DAGs-with-latents using the **m-separation oracle**
   (see `tests/oracle_pag.py`, `tests/fixtures_pag.py` — how `fci` is oracle-tested), `lmarvel()` must
   return the **identical PAG** to `cbcd.fci()` (endpoint matrix, SHD 0). Sound+complete gate (Thm 3).
2. **Efficiency win, recorded:** same runs, `RunRecorder` CI-test count for L-MARVEL << FCI.
3. **BK parity:** `lmarvel(background=bg)` == `fci(background=bg)` on the oracle, BK consistent with truth.
4. **External (optional):** the RCD Python package (algo80 §8) implements L-MARVEL — parity check.

## Scope
- **Build latents-only first (no selection bias, S=∅).** Theorem 2 requires the selection-induced
  undirected-edge subgraph to be chordal; with S=∅ there are no undirected edges, so it holds trivially,
  and it matches `cbcd.fci`'s default. This is the common case and gives a clean fci parity.
- **Selection bias = extension** (later): needs the chordality condition + undirected-edge handling.

## Caveats
- MB-discovery quality: finite-sample MB errors propagate (same as MARVEL). State the regime.
- `2^|Mb|` bounded by `Δ⁺_in` for removable vars -> wins on sparse/bounded-degree MAGs, degrades on dense.

Related: [[marvel-recursive-family-spec]] [[algorithm-cell-map]] (cell 2 = FCI anchor + L-MARVEL champion).
