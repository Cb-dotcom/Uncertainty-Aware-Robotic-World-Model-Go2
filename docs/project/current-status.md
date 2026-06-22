# Current Status

*A one-page snapshot of where the project actually is. For the per-claim ledger see
[Reproduction Status](../validation/reproduction-status.md); for the phase-by-phase narrative see the
Validation section (Phases 4A-4G).*

## One-line state

The project's headline result is no longer the offline walker. It is a **mechanistic negative**: on a
light, contact-dominated quadruped, broadening offline data improves world-model prediction while
**flattening the ensemble disagreement MOPO relies on**. The bottleneck is **uncertainty calibration,
not data volume**, demonstrated by a replicated decoupling between prediction error and disagreement.

## The arc, compressed

1. **Online RWM/MBPO on Go2 (Phases 4B-4C).** Fixed the scoot (`feet_slide`); plain MBPO finetune
 collapses via model exploitation.
2. **Pivot to offline (Phase 4D).** The thesis method is offline RWM-U / MOPO-PPO; the online collapse
 is motivation. Both online penalty arms collapse from a shared good state.
3. **Offline pipeline + ANYmal gate (Phase 4E).** Built the offline Go2 pipeline and a real-env eval
 harness; found Bug #1 (untrained-WM). ANYmal apparatus check passes with the authors' WM (0.417).
4. **Curated WM + offline walker (Phase 4F).** Found and fixed Bug #2 (normalizer freeze); produced a
 command-following but weak, seed-fragile offline walker at offline-ANYmal parity.
5. **Uncertainty-calibration arc (Phase 4G).** Ran the ANYmal fitter gate (fitter not broken), hit the
 ANYmal structural wall (ens1 reference → MOPO discriminator impossible there), built the
 exploit-rollout diagnostic, and found the headline result below.

## The headline finding (Phase 4G)

On two independent offline-policy failure traces, the failure-augmented WM shows next-state prediction
error **19-25× higher** near failure than walking, while ensemble disagreement is **≈1×, flat**. The
older curated WM gave ≈5× selective disagreement. Cross-checks rule out the trivial reads: the new WM
predicts normal dynamics ≈3× *better*, and its walking-uncertainty baseline *rose* ≈3×, so it is
**less selective**, not lower everywhere. High error with flat disagreement is *confidently wrong*:
miscalibration, not ignorance.

**Mechanism.** The world model is a **shared recurrent trunk + independent bootstrapped heads**: the
paper's own architecture, faithfully reproduced. Epistemic uncertainty is head disagreement over shared
features (measured state-head cosine ≈0.96, nearly parallel). The architecture sets a *ceiling* on
diversity; broadening the data drives the heads up against it, collapsing disagreement exactly where the
penalty is needed. The paper's epistemic estimator is therefore **fragile by construction**: strong
when data is narrow, weak as data broadens.

## What is solid (thesis-bearing, independent of the open lever)

- The **uncertainty-miscalibration result** with the MSE-vs-disagreement decoupling, replicated on two
 traces.
- The **shared-trunk ceiling** mechanism, with the paper-faithful architecture confirmed.
- The **exploit↔freeze conservatism frontier** (Phases 4E-4F) on a light quadruped.
- Two diagnosed root-cause **bugs** (untrained-WM, normalizer freeze) and the confound ledger
 ([Engineering Findings](../validation/engineering-findings.md)).
- The bounded fitter verdict: **not broken**, but more exploitable than the ANYmal reference.
- A deployable online `ens5` walker (hardware-demo artifact, separate from the offline result).

## What is open

- **Phase 4H, stage-Mixed.** A true PPO-replay-across-stages dataset (≈1M), fit ens5 identically, then
 re-score the `_9`/`_6` exploit traces. First metric to read: **state-head cosine** (does behavioral
 staging break the ≈0.96 ceiling?), then the exploit delta. Both branches pre-registered in
 [Phase 4G §9](../validation/phase-4g-uncertainty-calibration.md#9-open-resolved-in-phase-4h-stage-mixed).
- The **morphology / reward-support axis** (Go2 ≈15 kg at the skating optimum; `feet_slide` absent from
 the imagined reward), the remaining live hypothesis if stage-Mixed does not restore selectivity.

## Honest framing for write-up / defense

Lead with the **calibration finding + the shared-trunk ceiling + the two bugs**: solid regardless of
the stage-Mixed outcome. Present the offline walker as a supporting *existence proof* at the
exploit↔freeze frontier, not the headline. State plainly: there is **no code-level defect** separating
us from the paper; the genuine differences are data composition (expert+noise vs PPO-replay Mixed),
scale (≈1M vs the paper's 6M), and a lighter robot. The negative sits directly on top of the paper's own
stated limitation that offline data lacks the failure/recovery transitions that matter.
