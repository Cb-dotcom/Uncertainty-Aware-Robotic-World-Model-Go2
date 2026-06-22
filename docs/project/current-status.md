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

The campaign is **closed on the from-scratch axis.** Stage-Mixed (Phase 4G) decorrelated the ensemble
heads but did not beat the curated baseline; the reward-support proxy (Phase 4H) froze the policy as
predicted. Across penalty, data volume, data composition, and reward support, no from-scratch
configuration produces a robust offline Go2 walker, and the reason is mechanistic, not a bug.

The **warm-start** control has now been run on Go2 (Phase 4H, 3 seeds): initializing the offline policy
from the clean online ens5 walker and refining against the Go2 world model. It is **locally
non-destructive**, the output equals the input within seed noise (delta over the init checkpoint
approximately zero, with feet_slide slightly worse), so it does not demonstrate refinement; it pins the
discovery-versus-coverage explanation. With this, all levers are closed. Structural directions an
offline-MBRL group would propose next (independent-trunk ensembles, better-calibrated uncertainty
estimators, FK reward targets in the world model, warm-start refinement on coverage-rich data,
morphology-aware reward) are recorded in
[Phase 4H §5](../validation/phase-4h-policy-headtohead-and-reward-support.md#5-structural-directions-what-an-offline-mbrl-group-would-propose-next).

## Honest framing for write-up / defense

Lead with the **calibration finding + the shared-trunk ceiling + the two bugs**: solid regardless of
the stage-Mixed outcome. Present the offline walker as a supporting *existence proof* at the
exploit↔freeze frontier, not the headline. State plainly: there is **no code-level defect** separating
us from the paper; the genuine differences are data composition (expert+noise vs PPO-replay Mixed),
scale (≈1M vs the paper's 6M), and a lighter robot. The negative sits directly on top of the paper's own
stated limitation that offline data lacks the failure/recovery transitions that matter.


## Research questions (thesis spine)

The campaign resolves into four questions, each answered by experiment:

- **RQ1. Can offline RWM-U discover Go2 locomotion from scratch?** Not reliably. From-scratch runs
  collapse to standing, sliding, weak tracking, or fall-prone motion, surviving penalty tuning, failure
  augmentation, true-Mixed composition, and a reward-support proxy.
- **RQ2. Is the failure only uncertainty calibration or missing failure data?** No. +fail and true-Mixed
  change the uncertainty behavior (flat vs over-conservative) but do not recover robust from-scratch
  walking, and true-Mixed does not beat the curated baseline.
- **RQ3. Is it a missing reward-support issue?** A crude stance-motion proxy reduces sliding but
  suppresses locomotion (freeze), so proxy reward shaping inside an exploitable model is not enough.
- **RQ4. Can offline RWM-U refine an already-competent Go2 walker?** Warm-start is *locally
  non-destructive*: across three seeds it preserves strong walking (~0.65-0.69 m/s, corr ~0.98), but the
  output equals the competent input within seed noise and PPO barely updates, so this is staying-put, not
  measurable refinement.

**Single defensible thesis claim.** Offline RWM-U transferred to Go2 as a *locally non-destructive*
method around a competent policy, not as a reliable from-scratch locomotion-discovery method; the model
is accurate only where the data is dense, and a light, slide-prone morphology with a slide-blind imagined
reward makes from-scratch discovery exploitable, which together explain why a coverage-rich, heavier-robot
setup reaches 0.91 and a 115k-transition Go2 setup does not.
