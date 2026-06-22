# Phase 4H: Reward Support, Warm-Start Control, and the Discovery-vs-Coverage Closure

*Engineering / validation log. Continues Phase 4G. This phase closes the from-scratch offline-walker
question on Go2: after exhausting the penalty, data-volume, data-composition, and reward-support levers,
no from-scratch configuration produces a robust offline Go2 walker, while warm-start from a competent
policy is locally non-destructive. It records the reward-support experiment and the warm-start control,
states precisely what was and was not ruled out, frames the gap to the paper's 0.91, and lays out the
structural changes an offline-MBRL group would propose next.*

---

## 0. What this phase settles

Phase 4G established the calibration finding (coverage flattens disagreement) and the stage-Mixed
mechanism contrast (composition moves the failure mode without beating curated). The open question for
the *walker* (as opposed to the uncertainty signal) was reward support: the offline policy slides
because nothing in the **imagined** reward charges it for sliding, even though the real environment
does. This phase pulled that lever and found the predicted null.

> **One-line result.** Adding a contact-conditioned joint-motion proxy to the imagined reward reduced
> the real feet-slide metric, but only by suppressing locomotion: actual speed fell to 0.03-0.05 m/s
> and command correlation collapsed toward zero across three seeds. The proxy converted *sliding* into
> *standing*, not into clean walking.

---

## 1. The reward-support experiment

The imagined reward emits `track_lin`, `track_ang`, `feet_air_time`, `dof_acc_l2`, and contact terms,
but **not** `feet_slide`. Exact `feet_slide` (foot Cartesian speed during contact) is not computable
from the world model's outputs, which predict the 45-dim state and 8 contacts but not foot Cartesian
velocity. A from-scratch FK/Jacobian reconstruction was judged out of scope for the deadline and,
critically, would inherit the same world-model-fidelity ceiling (see §3). A contact-conditioned
joint-velocity proxy (`stance_joint_vel_l2`, penalize joint motion while the corresponding foot is
predicted in contact) was added instead, at a single pre-registered weight, with the verdict
pre-registered as *freeze*.

A correctness fix was also applied: the imagined `dof_acc_l2` term was differencing joint **position**
instead of joint **velocity** (wrong observation slice). The fix is correct but immaterial to the gait,
because the term's weight (~ -2.5e-7) makes its contribution a rounding error next to `track_lin` at
weight 1.0.

Joint ordering was verified before trusting the proxy (Go2 actuators are joint-major: all four hips,
then thighs, then calves), and the per-leg grouping was built to match
(`FL=[0,4,8], FR=[1,5,9], RL=[2,6,10], RR=[3,7,11]`).

---

## 2. Result: the predicted freeze, across three seeds

| run | episode length | feet_slide | base_contact | recomputed track_lin | actual speed | corr vx / vy |
|---|---|---|---|---|---|---|
| 31 (joint_acc fix only, 1 seed) | 933.9 | -0.049 | 0.075 | 0.389 | 0.148 m/s | +0.560 / +0.245 |
| 41 (proxy) | 977.1 | -0.017 | 0.042 | 0.278 | 0.034 m/s | +0.151 / +0.019 |
| 42 (proxy) | 931.1 | -0.019 | 0.105 | 0.258 | 0.048 m/s | +0.162 / +0.001 |
| 43 (proxy) | 941.3 | -0.020 | 0.030 | 0.304 | 0.039 m/s | +0.156 / +0.045 |

The proxy reduced real feet_slide (-0.049 to ~ -0.018) but collapsed locomotion: speed dropped roughly
3-4x and command-following fell from +0.56 to ~+0.15. Three of three proxy seeds froze in a tight
cluster, so the conclusion is solid: this is the pre-registered null.

**Why it froze, stated precisely.** The proxy is confounded with normal walking. Stance-phase joint
velocity is *supposed* to be nonzero while the body pivots over a planted foot; penalizing it penalizes
the legged dynamics of walking about as much as it penalizes skating, so the optimizer takes the
standing solution. This is a proxy-design limitation. It is **not**, on its own, evidence about
world-model fidelity; that argument rests on the separate exploit-trace decoupling (Phase 4G), and the
two claims are kept apart deliberately. Run 31 is a single seed and is reported as one draw from the
documented ~2/10 walking basin, not as a clean baseline.

---

## 3. Why reward shaping in imagination is bounded (the unifying point)

Any slide term lives in the **imagined** reward, computed on the world model's *predicted* contacts and
joint velocities, the same predictions the policy has spent the campaign learning to exploit. The
policy does not optimize real slide; it optimizes the model's belief about slide. If it can drive the
model into contact-misprediction regions (the exploitation measured in Phase 4G), it can satisfy an
imagined slide penalty while still skating in reality. Offline reward engineering therefore cannot
escape the model it is computed inside: reward-support fixes are bounded by world-model fidelity, which
is itself the bound this project ran into. This ties the reward axis back to the uncertainty/exploitation
result rather than standing as a separate failure.

---

## 4. The from-scratch closure (and the one lever not pulled)

Across the campaign, every **from-scratch** offline lever has now been tested on Go2 and none produces a
robust walker:

- **Penalty** (Phase 4F): a minority basin (~2/10 seeds) at -0.25; weak, slip-heavy, seed-fragile.
- **Data volume** (Phase 4G, +fail): more failure data flattened the disagreement signal.
- **Data composition** (Phase 4G, stage-Mixed): decorrelated the heads but did not beat curated, at a
  large off-support MSE cost.
- **Reward support** (this phase): the proxy froze the policy.

**Honest scope, now closed.** This is a *from-scratch* negative. The lever never previously applied to
Go2 was **warm-start**: initializing the offline policy from the clean online ens5 Go2 walker and
refining it against the curated Go2 world model under the MOPO penalty (the path that produced the
working ANYmal walker `model_6999`). This has now been run on Go2, three seeds.

**Result, stated against the right control.** The warm-start outputs are strong walkers:

| run | episode length | base_contact | track_lin | speed cmd / actual | corr vx / vy |
|---|---|---|---|---|---|
| 51 | 993.0 | 0.018 | 0.947 | 0.693 / 0.646 | +0.976 / +0.972 |
| 52 | 991.7 | 0.017 | 0.961 | 0.709 / 0.667 | +0.986 / +0.986 |
| 53 | 1000.0 | 0.000 | 0.961 | 0.725 / 0.690 | +0.986 / +0.983 |

But the **input** to warm-start (the online ens5 init policy) already scores 0.669 m/s actual, corr
+0.982 / +0.984, base_contact 0.0156, track_lin 0.955, feet_slide -0.025 on the *same* evaluator. The
outputs equal the input within seed noise, and the training log shows surrogate loss ~ -0.002 and value
loss ~0.002 from the start, i.e. PPO barely updated because the initialized policy is already near a
fixed point of the imagined objective. So the honest claim is **local non-destructiveness, not
refinement**: 500 iterations of offline MOPO-PPO around an already-competent policy did not break it,
but it also did not measurably improve it. The delta over the init checkpoint is approximately zero, and
on one axis it is slightly *negative*: warm-start feet_slide drifted to -0.075 to -0.115, i.e. *worse*
(more sliding) than the init policy's -0.025. To whatever tiny extent the policy moved, it moved toward
sliding, not away. ("Refinement succeeded" would require a positive delta over the init eval; the data
shows none, and the slide metric regressed. The init-eval control is the single number that pins this
and is reported alongside the table.)

This is consistent with the rest of the campaign and is the cleanest possible closing control: the world
model is accurate only where the data is dense, which is a small neighborhood of the competent policy,
so it is *locally* non-destructive there while being *globally* too inaccurate to support from-scratch
discovery. Discovery needs a globally accurate model; staying-put needs only a locally accurate one. The
from-scratch negative stands, and warm-start does not overturn it; it pins the discovery-versus-coverage
explanation.

---

## 4b. The gap to the paper's 0.91, and why from-scratch is the hard case

Three reinforcing factors explain the from-scratch failure and the distance to the paper's reported
0.91 normalized score, without invoking any project-level defect.

**Global versus local model fidelity.** From-scratch discovery forces the policy through every poor
intermediate gait between random and walking, so the world model must be accurate along that whole
path. The curated WM is fit to ~115k transitions from a narrow band of behavior, so it is accurate only
where that data sits and exploitable elsewhere; the policy finds those errors and settles in them
(skate, freeze). Warm-start needs the model accurate only in a small neighborhood of one good policy,
which is exactly where the data is densest, so it holds. Discovery needs a globally good model;
staying-put needs only a locally good one.

**Morphology.** A 15 kg Go2 can satisfy a velocity-tracking reward by sliding, and the imagined reward
is structurally blind to this because the WM never outputs foot Cartesian velocity, so there is no exact
slide term (the stance proxy froze instead of fixing it, §1-2). On a ~50 kg ANYmal, skating is far less
available, so the easy optimum sits much closer to real walking. The same method lands differently
because the cheap exploit differs by mass.

**The paper's "offline" is closer to warm-start than to from-scratch.** Their dataset is ~800k sim plus
~200k real Mixed transitions spanning many policies, so the good-walking region is already densely
covered in the data the WM is fit to. Optimizing against a model whose data already contains competent
walking is refinement toward a well-represented region, not cold discovery through uncovered space. So
the paper's 0.91 and this project's warm-start non-destructiveness may be the *same* mechanism: walking
works when the model covers walking. From-scratch fails here because a 115k Go2 set on a skating-prone
robot does not densely cover good walking at the commanded speeds, so discovery must cross terrain the
WM gets wrong. The gap is not a methodological error; it is roughly an order of magnitude less and far
less diverse data, a lighter robot whose easy optimum is sliding, and a reward that cannot see sliding.

## 5. Structural directions (what an offline-MBRL group would propose next)

The campaign shows the limits are not bugs and not data volume; they are structural properties of the
RWM-U / MOPO-PPO recipe interacting with a light quadruped. The directions below change the *structure*,
not the dataset, and are the honest "what next" rather than claims this project validated.

1. **Decorrelate the uncertainty estimator.** Epistemic uncertainty here is head disagreement over a
   *shared* recurrent trunk; the trunk caps diversity (Phase 4G). Independent per-member trunks (a true
   deep ensemble), or an explicitly diversity-regularized ensemble, would give disagreement a higher
   ceiling and a better chance of staying selective under broad data. This directly targets the measured
   mechanism.

2. **Replace ensemble-disagreement uncertainty with a better-calibrated estimator.** The decoupling of
   prediction error from disagreement is a known deep-ensemble pathology. Alternatives that calibrate
   off-support (e.g. distance-aware or evidential heads, or normalizing-flow density on the
   state-action support) would make the MOPO penalty fire where error actually rises rather than where
   the heads happen to disagree.

3. **Restore propagated/observable reward support inside the world model.** The walker fix needs an
   *exact* slide signal in imagination, which requires the world model to predict foot Cartesian
   velocity (extend the prediction head with FK targets), so the reward is computed on a quantity the
   model actually represents rather than a confounded joint-motion proxy.

4. **Warm-start offline refinement as the deployment recipe.** Treat offline MBRL as *refinement* of a
   simulation-trained policy rather than from-scratch discovery, matching the finetune-resume path that
   works on ANYmal. This concedes the discovery claim but is the realistic route to a deployable light-
   quadruped policy.

5. **Address morphology in the reward, not the model.** A 15 kg Go2 sits at a skating optimum that a
   50 kg ANYmal does not; a gait/air-time/contact-schedule reward designed for the light morphology
   (validated *online* first, where the penalty is exact) is a prerequisite before expecting any offline
   method to produce clean walking.

---

## 6. Status at end of Phase 4H

- Reward-support lever: **pulled, predicted freeze confirmed** across three seeds.
- From-scratch offline Go2 walker: **negative across all four levers** (penalty, volume, composition,
  reward support), mechanistically explained.
- Reward shaping shown to be **bounded by world-model fidelity** (§3), unifying the reward and
  uncertainty axes.
- Warm-start control: **run on Go2 (3 seeds), locally non-destructive** (output ~ input within seed
  noise, near-zero PPO updates); pins the discovery-versus-coverage explanation, does not overturn the
  from-scratch negative.
- Structural directions (§5) recorded as the honest "what an offline-MBRL group changes next," not as
  validated claims.
- **The from-scratch study is complete and writeable now.**
