# Phase 4G: The Uncertainty-Calibration Arc, and What Actually Bounds Offline Go2

*Engineering / validation log. Continues Phase 4F. Work spans 2026-06-21 → 2026-06-22 on the lab
workstation. This phase replaces the offline walker as the project's headline result: it isolates
**why** the offline Go2 policy does not robustly walk, and locates the bottleneck in the
**calibration of the epistemic uncertainty signal**, not in dataset size.*

---

## 0. Scope and the one-line result

Phase 4F left the contribution as "a demonstrated but seed-fragile offline walker at offline-ANYmal
parity," with the diverse-data rebuild and the ANYmal-WM diagnostic named as open. This phase ran those
experiments. The result reframes the whole project:

> On a light, contact-dominated quadruped, broadening offline data **improves next-state prediction**
> while **flattening the ensemble disagreement** that MOPO uses as its conservative penalty. The world
> model becomes more accurate and *less* informative as a trust signal, measured directly as a
> decoupling between prediction error and disagreement (≈25× MSE elevation near failures against ≈1×
> disagreement elevation), replicated across two independent policy-failure traces. The bottleneck is
> **uncertainty calibration, not data volume.**

This is a clean, mechanistically-explained negative-with-cause, and it is the thesis-bearing result.

---

## 1. The ANYmal fitter gate: the fitter is not broken

The first question was whether the standalone `fit_world_model.py` is fundamentally broken (failure
cause A), before blaming Go2 data (B) or morphology/reward (C).

A strict architecture check forced the issue: the shipped ANYmal reference checkpoint
(`pretrain_rnn_ens.pt`, and the local `…budget5000_fresh/model_5000.pt`) is **ensemble_size = 1**.
Building ens5 and strict-loading the ens1 reference fails; ens1 loads bit-identically. With ens1, the
fit converged cleanly on the shipped 10k ANYmal CSV: state loss 0.62 → 0.015, contact → 0.000,
architecture VERIFIED by strict load, normalizer saved.

**Conclusion:** the fitter trains, strict-loads against the authors' module, and produces an
architecture-compatible normalized checkpoint. The trivial "fitter is broken" explanation is dead.

**But the gate is narrow, by construction.** ens1 produces **zero epistemic uncertainty**
(`std` across one member is zero), and the ANYmal CSV has zero terminations, so this gate is silent on
the two mechanisms that matter most: the MOPO penalty and the termination head. It is a
convergence/plumbing gate, not a policy-usefulness gate.

---

## 2. The ANYmal structural wall: the MOPO discriminator cannot be built there

The intended decisive test was: build an ANYmal WM with *our* fitter at ens5, run the offline policy at
the Go2-working penalty (−0.25), compare to the authors' WM. Two findings collapsed this route:

1. **The penalty was inert in every ens1 ANYmal run.** Epistemic uncertainty is ensemble-mean
 disagreement; at ens1 it is identically zero, so `penalty × 0 = 0` for any coefficient. A 4× penalty
 swing (−1.0 → −0.25) moved `base_contact` by 0.004, which proves the penalty multiplied zero. Every ens1
 ANYmal "penalty" run was pure unpenalized MBPO.
2. **There is no ens5 reference ANYmal WM on disk.** The only reference is ens1. So the penalty-active,
 reference-vs-fitted A/B that would cleanly separate "fitter" from "data" **cannot be constructed on
 ANYmal**: the reference simply does not carry an ensemble.

**Bounded finding that survives:** under unpenalized rollout, our fitted ANYmal WM is *more exploitable*
than the reference (≈97% vs ≈76% `base_contact`; imagined episode length ≈27 vs ≈153). So the fitter is
sound but its learned dynamics are somewhat more exploitable than the authors', real and citable, but
not a verdict on the uncertainty mechanism, which ens1 cannot test. The mechanism can only be measured
where ens5 and a live penalty coexist: **Go2.**

---

## 3. The exploit-rollout diagnostic: testing where the policy actually fails

The n10 noise-level holdout (Phase 4F) is a *near-OOD* check on the expert+noise manifold; it does not
test the region an optimized offline policy actually drifts into. The decisive instrument is the
**exploit-rollout diagnostic**: roll a trained offline policy in the real env, dump the 66-column
world-model input trace, and score the **fitted WM's** epistemic uncertainty on the states the policy
actually visits, at the fall transition, and one and five steps *before* failure (the lead-time where a
penalty would have to fire to steer the policy away).

Two old offline policies were traced: `_9` (most fall-heavy, `base_contact ≈ 0.14`) and `_6`
(better-tracking, fewer falls). Both scored against three WMs on the **same** trace, read as a *delta*,
with the walking baseline printed (not absolute AUC).

---

## 4. The result: prediction improves, disagreement does not track error

The pattern is stable across both traces. Representative numbers (epistemic = ensemble-mean
disagreement; MSE = next-state prediction error; both fall-vs-walking ratios):

| trace `_9` | old curated WM | new +fail WM |
|---|---|---|
| fall epistemic AUC | 0.982 | 0.723 |
| pre-5 epistemic AUC | 0.969 | 0.649 |
| fall/walking epistemic ratio | 4.97× | 1.48× |
| pre-5 epistemic ratio | 1.90× | **1.00×** |
| pre-5 **MSE** ratio | 3.8× | **25.2×** |

| trace `_6` | old curated WM | new +fail WM |
|---|---|---|
| fall epistemic AUC | 0.979 | 0.644 |
| pre-5 epistemic AUC | 0.969 | 0.636 |
| pre-5 epistemic ratio | 4.61× | **1.14×** |
| pre-5 **MSE** ratio | 5.3× | **18.8×** |

The decisive line is the new WM's pre-failure rows: **prediction error is 19-25× higher than walking,
while disagreement is ≈1×, flat.** The model is much *wronger* near failure but does not *report* being
wronger. That is the precise violation of the assumption MOPO depends on (epistemic uncertainty as a
proxy for model error). The termination head, by contrast, stays near AUC 1.0 on both, confirming the
collapse is specific to the epistemic *penalty* signal, not fall *classification*.

---

## 5. The baseline-and-MSE cross-check: miscalibration, not ignorance

Two confounds were ruled out before believing the negative:

- **Walking-uncertainty baseline rose.** Old curated walking epistemic mean ≈ 0.247; new +fail ≈ 0.746
 (≈3×). So the ratio compression is partly the *walking floor rising*, not fall-uncertainty
 disappearing (fall means are comparable: 1.23 vs 1.10). The new WM is **less selective**, not "lower
 uncertainty everywhere."
- **Prediction did not get worse; it got better.** New +fail walking MSE ≈ 0.032 vs old curated ≈ 0.089
 (≈3× *better* normal-dynamics prediction). The new WM is the best of the three at modeling the regime
 it covers.

Together: the new WM **knows more and signals less**. High error with flat disagreement is
*confidently wrong*, the dangerous regime, and it is miscalibration, not ignorance.

*(The `go2_heldout_wm` bracket is excluded from the headline comparison: it has no saved normalizer and
is scored in raw space, so it is not on the same axis as the curated and +fail WMs.)*

---

## 6. Why broadening data does this: the shared-trunk ceiling

A structural check of `SystemDynamicsEnsemble` explains the mechanism and, importantly, confirms it is
**not a project bug**:

- The architecture is a **shared recurrent trunk + independent ensemble heads**: one `state_base` GRU
 feeding 5 `state_heads`, one `auxiliary_base` GRU feeding 5 `auxiliary_heads`. This **matches the
 paper** ("bootstrap ensembles to the prediction head after a shared recurrent feature extractor"), so
 it is faithful reproduction, not deviation.
- Epistemic uncertainty is therefore *head disagreement over shared features*, decorrelated only by
 per-member bootstrap resampling and init. On well-covered data with shared features, that
 decorrelation is weak. Measured directly: **state-head pairwise cosine ≈ 0.96** (nearly parallel),
 versus auxiliary-head cosine ≈ 0.06.
- **The architecture is constant across the old and new WMs**, so it does not *cause* the delta, it
 sets a *ceiling* on achievable diversity. Broadening the data drove the heads up against that ceiling:
 once all members see the failure manifold, they agree there, and disagreement collapses exactly where
 the penalty is needed.

This is the mechanistic core of the thesis: **the paper's epistemic estimator is fragile by
construction, it yields signal when data is narrow (heads forced apart by extrapolation) and loses it
as data broadens (heads collapse onto shared features).** The Phase 4F curated WM's strong selectivity
was partly an artifact of *narrow data*, not a property that survives coverage.

---

## 7. Dead ends ruled out (so they are not re-investigated)

- **"Autoregressive training is off" (`seq_loss = 0`, `prediction_type = single`).** Investigated and
 **rejected.** `prediction_type = "single"` is hardcoded in the authors' module; `compute_state_loss`
 loops over the full forecast horizon and feeds each step's *own* prediction back as the next input -
 i.e. autoregressive rollout is active, folded into `state_loss`. The separate `sequence_loss` term is
 simply unused in single mode. This is the shipped code, identical for ANYmal and Go2, so it cannot be
 a Go2-specific difference from the paper. Not a bug.
- **"Our fitter is broken."** Rejected (§1).
- **"More penalty tuning will fix it."** Rejected. The penalty scales a signal that is not selective,
 no coefficient rescues a flat disagreement profile.
- **"Collect 6M of the same data."** Rejected on the project's own evidence: 115k → 1.288M made
 exploit-region disagreement *worse*; extrapolating predicts further degradation, not parity.

---

## 8. What the paper actually claims, and where we genuinely differ

Re-reading the paper closes the "are we doing something wrong" question:

- **Data scale.** Table S10 reports RWM-U training with ≈6M transitions (generic setting); the
 real-world ANYmal sim/real ablation (Table 1) uses ≈1M total (best: 800K sim + 200K real). Our Go2
 sets are 115k-1.288M, comparable to the 1M regime, far below 6M.
- **Data *composition*.** The paper's best dataset is **Mixed = replay-buffer data from PPO across
 training stages**; Expert-only "overfits to specific dynamics." Every Go2 set so far was
 expert-or-curated **plus injected noise**: manifold-centered, not behavioral replay. This is the one
 genuine, untested methodological difference, and the next experiment (Phase 4H, stage-Mixed) targets
 it directly.
- **The paper's own limitations section** states offline data "may lack critical transitions such as
 failures or recovery maneuvers", i.e. it names this exact difficulty as unsolved. Our negative sits
 on top of their stated limitation rather than contradicting their results.

So the honest position: there is **no code-level defect** separating us from the paper; the live
differences are data composition, scale, and a lighter robot (Go2 ≈15 kg) that sits at a skating
optimum the heavier ANYmal does not.

---

## 9. Open: resolved in Phase 4H (stage-Mixed)

The single remaining lever that tests a *real* difference from the paper: a **true stage-Mixed**
dataset built from PPO replay across verified walker checkpoints (early/mid/late stages), held at ≈1M,
fit with the identical ens5 pipeline, then re-scored on the same `_9`/`_6` exploit traces.

The **first** metric to read is **state-head pairwise cosine**, not AUC: it directly tests whether
behavioral staging decorrelated the heads below the +fail WM's ≈0.96 ceiling.

Two pre-registered branches:

- **Cosine drops and pre-failure selectivity returns** → *data composition*, not volume, is the lever;
 expert+noise was the wrong recipe and behavioral replay restores the MOPO signal. Near-positive
 result.
- **Cosine stays ≈0.96 and selectivity stays flat** → the shared-trunk ceiling (and/or Go2 morphology /
 `feet_slide`-absent-from-imagination reward support) bounds the mechanism regardless of data. The
 negative becomes bulletproof, having ruled out the data-recipe confound.

Either outcome completes the thesis. *(Result pending; this section will be filled once the stage-Mixed
fit and exploit re-scoring finish.)*

---

## 10. Status at end of Phase 4G

- Fitter: **not broken** (converges, strict-loads, produces walkers); bounded "more exploitable than
 reference" on ANYmal.
- ANYmal MOPO discriminator: **structurally impossible** (ens1 reference), closed, not deferred.
- Exploit-rollout diagnostic: **built and run**; the decisive instrument.
- Headline result: **uncertainty miscalibration under coverage**: prediction improves while
 disagreement flattens near failures, replicated on two traces, with the MSE-vs-disagreement decoupling
 as proof it is calibration not ignorance.
- Mechanism: **shared-trunk + bootstrapped-heads ceiling** (paper-faithful architecture), state-head
 cosine ≈0.96 on broad data.
- Dead ends documented (§7) so they are not re-run.
- One open lever (stage-Mixed) with both branches pre-registered (§9).
