# Phase 4F: Curated Dataset, World-Model Fit, the Freeze Bug, and the Offline Walker

*Engineering / validation log. Continues Phase 4E. Work spans 2026-06-19 → 2026-06-21 on the lab
workstation.*

---

## 0. Scope

This phase is the contribution proper. It builds a curated, coverage-broadened Go2 dataset; fits a Go2
world model on it with a standalone offline trainer that lifts the exact system-dynamics loss from the
online runner; validates the fitted termination head on held-out falls; finds and fixes a
normalizer-consistency bug that had frozen every offline Go2 policy; and produces the first
command-following offline Go2 walker, characterized honestly, including its seed sensitivity and gait
quality.

---

## 1. The curated dataset: goal-directed falls, not exogenous shoves

Phase 4E established that the limiting factor was world-model coverage, and that pushes produce the
*wrong kind* of falls. The curated recipe instead perturbs a *competent* policy with escalating action
noise, so the robot falls *while still trying to track*, failures on the edge of the walking manifold,
exactly where an offline tracking policy operates. This mirrors Paper 2's noise-scale schedule
(`[0.1, 0.2, 0.4, 0.5, 0.8]`).

Collection used `collect_go2_dataset.py` (single env, to preserve the temporal sequence the GRU world
model trains on) across a noise sweep on the competent checkpoint. The validation gate is the per-segment
termination count:

| segment | noise | rows | real falls | mean &#124;v_xy&#124; | tilted frac |
|---|---|---|---|---|---|
| n00 | 0.0 | 10000 | 1 | 0.706 | 0.002 |
| n02 | 0.2 | 10000 | 1 | 0.645 | 0.005 |
| n04 | 0.4 | 15000 | 1 | 0.709 | 0.002 |
| n08 | 0.8 | 30000 | 33 | 0.545 | 0.020 |
| n10 | 1.0 | 25000 | 67 | 0.452 | 0.047 |
| n12 | 1.2 | 25000 | 68 | 0.408 | 0.058 |
| **total** | | **115000** | **171** | | |

This is the covered-and-competent-source dataset every prior WM lacked: clean tracking dynamics from the
low-noise segments and **goal-directed failure coverage** from the high-noise ones, with even the
noisiest segment still moving at 0.41 m/s (not degenerate spawn-flailing). Concatenation forces a
termination on the last row of each segment so the GRU's sequence sampler never draws a window across a
segment seam.

> **Two earlier dataset attempts failed the gate first**, and the gate caught them, a competence ladder
> (`model_250…2000`) came back with ~0 falls (the Go2 pretrain stands early and walks late; neither tips
> over), confirming that *noise on a competent policy*, not checkpoint diversity, is what produces
> goal-directed falls. This is the discipline that kept the project from fitting a WM on fall-blind data
> a third time.

---

## 2. The standalone world-model fit

`fit_world_model.py` instantiates the `SystemDynamicsEnsemble` and loops the **exact**
`update_system_dynamics` body lifted from `mbpo_on_policy_runner.py` (verified: `sd.reset()` per step,
`bootstrap=True`, the seven weighted loss terms, `clip_grad_norm_`, the optimizer dance). The only new
code is the CSV→buffer load, the outer iteration loop, and the save. Two design points that mattered:

- **Architecture verified by strict load.** Before training, the freshly-built ensemble loads the
 authors' ANYmal `model_2000` with `strict=True`. A bit-identical strict load proves the construction
 matches the shipped WM exactly, so the fitted WM will load through `--wm-checkpoint`, a stronger
 guarantee than reading the constructor signature.
- **The termination-head imbalance lever (`pos_weight`).** At 171 positives in 115000 rows (0.154%), the
 naive BCE termination head collapses to all-negative (the Phase 4D blindness). The fit injects
 `pos_weight` into the termination BCE on this instance only (online code untouched). The auto value at
 this imbalance is ~648.

---

## 3. The held-out termination gate: AUC 0.978, but bimodal

Scoring the head on its *training* falls is memorization (it returns a degenerate 1.000/0.000 split). The
honest gate is a held-out split: fit on segments n00-n10 (first 90k rows, 103 falls), test on n12 (the
final 25k rows, 68 falls the WM never saw), with negatives drawn from n12's noisy-but-upright windows
(the regime the policy actually occupies).

**Result: ROC-AUC = 0.978** on the held-out falls. The head **generalizes**: it ranks unseen falls,
in the noisiest regime, cleanly above noisy-upright negatives. That directly refutes the Phase 4D
online termination-blindness *on a properly-built WM*.

**But read the distribution, not the AUC.** Recall at threshold 0.5 is only ~0.50, and the threshold
sweep is flat (recall ~0.52 → 0.50 → 0.44 across 0.3 → 0.7). The fall probabilities are **bimodal**:
~44% above 0.7, ~48% at or below 0.3, only ~7% in the middle. The head catches roughly half the unseen
fall modes at high confidence and misses the other half near zero, it is **half-covered**, not
under-confident. The honest summary for the thesis is recall-at-operating-point and the bimodal
coverage, **not** the headline 0.978 (which is flattered by easy negatives at ~0.000).

> **Methodology caveats recorded.** AUC over 68 positives vs 2000 negatives is optimistic; the held-out
> negatives are still partly on-manifold (one "walking" window scoring 1.0 is almost certainly a real
> near-fall recovery, making the false-alarm rate conservative). And this gate validates *fall
> detection*, not *horizon dynamics accuracy*, necessary, not sufficient, for the policy to walk.

---

## 4. Bug #2: the normalizer-consistency freeze

The first policy run against the fitted curated WM **froze**: imagined episode length pinned at exactly
**1.00** for all 500 iterations, every env, zero variance. A flat, zero-variance 1.00 is the wrong shape
for distribution-shift drift (which gives spread); it is the signature of a head firing on *every* state.

Root cause, found by reading the dataset class rather than guessing: the offline pipeline's `Dataset`
**z-scores** state and action (computes mean/std from the data, line 131; subtracts/divides, line 162),
and the imagination loop runs entirely in this normalized space. But `fit_world_model.py` had trained the
WM on **raw** values (its docstring's "identity normalizer" assumption was simply wrong). At policy time
the WM was fed zero-mean/unit-std inputs it had never seen, so every head, including termination -
misfired, the termination gate fired on the first step of every imagined episode, and episode length
pinned at 1.

This also explained the ANYmal/Go2 split with no appeal to data diversity: ANYmal Phase 4E used the
*authors'* WM, trained through their pipeline in normalized space, consistent. Our Go2 WM was the only
one we trained ourselves, and we trained it raw, inconsistent. Same imagination code, opposite outcome,
one normalizer.

The validation gate had also been blind to this because the held-out gate scored the WM in *raw* space -
consistent with the broken training, so structurally unable to see the deployment-space mismatch. **Both
fit and validation lived in raw space; only deployment lived in normalized space, and nothing tested
deployment space.** That is the real methodology hole this bug exposed.

**Fix:** z-score state and action in the fit using mean/std from the same CSV (contact and termination
left raw, the pipeline only normalizes state/action), and point the Go2 config's `dataset_folder` at the
curated data so the deployment normalizer is computed from the *same* rows. The `s_std[:3] ≈ [0.42,
0.43, 0.19]` values printed by the fixed fit quantify the mismatch: those state dims were being amplified
2-5× at deployment, exactly the misfire. See
[Engineering Findings §2](engineering-findings.md#bug-2-normalizer-consistency-freeze).

> This is the single most important reason the verify-before-run discipline mattered: the diverse-data
> rebuild we nearly committed days to would have fed beautifully diverse data through the *same*
> raw-vs-normalized mismatch and frozen at 1.00 again.

---

## 5. The offline walker: and an honest account of it

With all three surfaces (WM training, deployment normalizer, imagination resets) finally reading the same
curated data in the same normalized space, the freeze lifted and the penalty became meaningful. The
best result, at penalty −0.25:

| metric | `ens5` online walker (ceiling) | offline best seed (−0.25) | offline ANYmal (gate, −1.0) |
|---|---|---|---|
| actual speed (m/s) | 0.669 (cmd 0.709) | **0.22** (cmd 0.74) |, |
| `corr(cmd_vx, act_vx)` | +0.98 | **+0.78** |, |
| `corr(cmd_vy, act_vy)` | +0.98 | +0.27 |, |
| `track_lin_vel_xy_exp` | 0.587 | **0.44** | 0.417 |
| `feet_slide` per metre | baseline | ~9× ceiling |, |

**Honest framing.** This is a *genuine command-following offline walker*, the thing that was actually in
doubt, but a **weak, conservative, partially-tracking, slip-heavy** one. It reaches ~30% of commanded
speed, tracks forward direction well (vx corr 0.78) but lateral poorly (vy 0.27), and slides ~9× more per
metre than the online `ens5` reference. Its `track_lin` 0.44 is *comparable to the offline ANYmal
baseline* (0.417), i.e. on par with what the same offline optimizer achieves on the reference robot -
**not** on par with the online `ens5` walker (0.587, corr 0.98). The slip-heavy gait is consistent with
the noise-perturbed training distribution and with `feet_slide` being absent from the imagined reward.

**Seed sensitivity (bistability).** Across seeds the outcome is bimodal: roughly 2 in 10 seeds produce
the walker above; the majority collapse to **standing** (the freeze basin), with a couple marginal. So
the result is **robustness-limited**: a walker exists and is reproducible, but it is a minority basin, not
the typical seed. The seeds vary only the *policy*, the world model is `n=1`, so this measures
policy-initialization robustness, not method robustness. Both framings are defensible; the honest
headline is "a demonstrated but seed-fragile offline walker at offline-ANYmal parity," not "it walks,
done."

---

## 6. The conservatism / coverage frontier (the structural result)

Pulling Phases 4E-4F together, one morphological property, Go2 is light (~15 kg) and contact-dominated -
produces a two-sided failure with a narrow operating band between:

- **no penalty** → world-model **exploitation** (imagined reward decouples from real; exploit-and-fall or,
 on the curated WM, exploit-and-stand);
- **strong penalty** → **over-conservatism / freeze** (the policy refuses to leave the data manifold;
 imagined episodes terminate immediately; the robot stands);
- a **narrow intermediate band** (≈ −0.25 here) where a fragile walker emerges.

This frontier, and the normalizer-consistency requirement that gates *any* of it being measurable, is
the structural contribution, independent of how strong the walker is.

---

## 7. Status at end of Phase 4F

- Curated coverage-broadened dataset: **built and gate-validated** (171 goal-directed falls, both axes
 present).
- Standalone WM fit: **built** (exact loss lift, strict-load architecture check, `pos_weight` imbalance
 lever).
- Held-out termination gate: **AUC 0.978**, honestly bimodal / half-covered.
- Bug #2 (normalizer freeze): **found, root-caused, fixed.**
- Offline Go2 walker: **achieved**: command-following, but weak/conservative/slip-heavy and
 seed-fragile (minority basin), at offline-ANYmal parity on `track_lin`.
- Structural result: the **exploit ↔ freeze conservatism frontier** plus the normalizer-consistency
 finding.
- Open: seed walk-rate over a larger set; a rendered video to confirm trot-vs-shuffle gait; the
 diverse-data rebuild and the ANYmal-WM diagnostic as the named next experiments (see
 [Engineering Findings](engineering-findings.md) and [Reproduction Status](reproduction-status.md)).
