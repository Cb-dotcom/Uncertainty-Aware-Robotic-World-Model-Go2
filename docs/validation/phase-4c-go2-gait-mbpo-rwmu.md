# Phase 4C — Go2 Gait Correction, the Plain-MBPO Negative Control, and the RWM-U Setup

*Engineering / validation log. Continues the Phase 4B inventory + scaffold notes. Work spans
2026-06-11 → 2026-06-12 on the lab workstation (`rwmu-cogar-cb`), RTX 6000 Ada, shared with a
root-owned vLLM (~41.5 GB), so all training ran at reduced parallelism.*

---

## 0. Scope

This document covers everything from the end of Phase 4B (Go2 MBPO infrastructure built and running,
but the Go2 baseline/pretrain learning a **scooting** gait) through to the launch of the
uncertainty-aware (RWM-U) comparison. It records the experiments, the reasoning behind each decision,
the dead ends, the root cause, and the current open experiment. The headline outcome: the Go2 scoot
was a reward-shaping loophole (fixed with a `feet_slide` penalty), and plain MBPO finetuning from the
fixed baseline **degrades** the policy — a clean negative control that motivates the uncertainty-aware
variant.

---

## 1. Starting point (end of Phase 4B)

By the end of Phase 4B the Go2 MBPO pipeline was complete and verified to run end-to-end:
Init / Pretrain / Baseline / Finetune stages registered, the Go2 manager-based MBRL env ported from
ANYmal-D, imagination tags confirmed in TensorBoard. The remaining problem was **behavioral, not
structural**: the Go2 pretrain/baseline learned excellent velocity tracking but executed it by
keeping the body low and sliding the feet ("scooting"/"skating") rather than taking real steps.

The trap that made this slow to diagnose: the scoot produces **good-looking metrics**. The scooting
pretrain (`2026-06-11_12-35-49_pretrain`) ended with `track_lin_vel_xy_exp ≈ 0.965`,
`error_vel_xy ≈ 0.147`, episode length 1000, `base_contact ≈ 0`. Every scalar said "healthy." Only the
render showed the belly-low slide. **Lesson carried through the whole phase: scalar tracking metrics
cannot distinguish walking from skating — the render and the aggregate termination/air-time signals
are the verdict.**

---

## 2. Confirming the scoot is real (not an instrumentation artifact)

First we ruled out the cheap explanation that `feet_air_time ≈ 0` was a dead/miswired sensor:

- The `feet_air_time` contact sensor resolves to the four Go2 foot bodies (`.*_foot`), confirmed.
- A clean, versioned render of the pretrain checkpoint (built a Go2 Visualize task mirroring
  ANYmal-D's) showed the robot splayed, belly near the ground, feet sliding.

So `feet_air_time ≈ 0` meant a genuine near-continuous-contact gait, not a broken sensor. The scoot
was real.

---

## 3. First hypothesis — and why it was wrong

The initial suspicion (carried in from the Phase 4B notes) was that two reward weights were
mis-set: `feet_air_time` too strong and `flat_orientation_l2` too strong. A "rewardfix" lowering
`feet_air_time` 0.5→0.25 and `flat_orientation_l2` −5.0→−2.5 was applied and re-pretrained
(`12-35-49`). **It still scooted.** That ruled out the first hypothesis and forced a more disciplined
approach: stop guessing single weights, and localize the cause by comparison.

---

## 4. Triangulation — localizing the cause

Three facts were used to bound the problem:

1. **ANYmal-D walks** on this exact RWM pipeline (same Isaac Sim 5.1, same rsl_rl_rwm, same
   workstation) — verified by render. ⇒ the method, simulator, and pipeline are not the cause.
2. **Stock IsaacLab Go2** could be trained here and produced a walk in at least one rollout. ⇒ the Go2
   actuator (`UNITREE_GO2_CFG`: effort 23.5, stiffness 25, damping 0.5, action scale 0.25), USD, and
   physics can produce walking in this sim.
3. **Only our Go2 RWM config scoots.**

Together these pointed at the **Go2 reward/task configuration**, not the world model, the robot, or the
simulator. A direct grep of stock IsaacLab Go2 flat vs our config surfaced the real differences (the
earlier `undesired_contacts` hypothesis was refuted — it is disabled in *both*):

| term | ours (RWM Go2) | stock IsaacLab Go2 flat |
|---|---|---|
| `dof_torques_l2` | −2.5e-5 | −2e-4 (≈8× stronger) |
| `track_lin_vel_xy_exp` | 1.0 | 1.5 |
| `track_ang_vel_z_exp` | 0.5 | 0.75 |
| `stand_still` | −1.0 (RWM-custom) | none |
| `feet_air_time` threshold | 0.25 | 0.5 |
| `undesired_contacts` | None | None |

---

## 5. The stock control — and the reframe that changed the whole picture

To test "is it our reward weights or the environment," we trained **stock Go2 PPO** as a control.

- A 300-iteration run was inconsistent (one of four envs walked, the pinned env scooted). Diagnosed as
  **undertrained**: `error_vel_xy` was still falling at the cap (0.32→0.29→0.27→0.24, no plateau),
  action-noise still dropping.
- A converged 800-iteration run (`2026-06-11_22-03-44`) plateaued at `error_vel_xy ≈ 0.183`,
  `track_lin ≈ 1.43`, `feet_air_time ≈ −0.026` — and **still scooted** on render.

**This was the pivotal result.** A *converged, off-the-shelf, validated* Go2 config scoots in this
setup. That killed the hypothesis that the scoot was specific to our reward weights — both stock and
ours skate. The reframe:

> A low skate is the genuine optimum of a velocity-tracking reward for a ~15 kg quadruped on flat,
> low-perturbation ground when nothing in the reward penalizes sliding.

The reward breakdown of the scooting pretrain made the mechanism concrete: per step,
`track_lin (0.965) + track_ang (0.453) ≈ 1.42` of positive reward, while *every* shaping/penalty term
combined was ≈ −0.12, and `feet_air_time` — the only term meant to induce stepping — was ≈ −0.0036,
i.e. essentially zero. ~92% of the reward magnitude is tracking, which a scoot satisfies perfectly,
and there is no meaningful "lift your feet" pressure. The light body slides almost for free
(contrast ANYmal-D at ~50 kg, which physically cannot slide cheaply — same reward, different optimum).

This is a **known IsaacLab Go2 failure mode**, not a project bug: public reports describe the policy
"learning to walk while dragging its feet," with floor friction implicated, and users noting they were
"never completely happy with the results" of the stock Go2 tutorials. Clean Go2 gaits in the
literature come from rough-terrain training plus richer gait-shaping reward systems (e.g. Unitree's
`unitree_rl_lab`), not the minimal flat config.

---

## 6. The fix — a `feet_slide` penalty

`feet_slide` (already present in IsaacLab's locomotion MDP but unused by stock Go2) directly penalizes
foot world-frame velocity while a foot is in contact — i.e. it charges for exactly the sliding seen in
the render. We swept it on stock Go2 (fast plain-PPO loop) before committing it to the RWM pipeline:

| setting | result |
|---|---|
| `feet_slide=-0.25`, `feet_air_time=0.25`, `undesired_contacts=None` | **best** — body higher, legs extended, less skating; `error_vel_xy≈0.23`, `base_contact≈0.025`, `feet_slide≈−0.14` |
| `feet_slide=-0.10` + `undesired_contacts=-1.0` | worse; contact term sat at ≈0 (never fired), no help |
| `feet_slide=-0.50` | over-penalized — stompy, `error_vel_xy≈0.48–0.59`, `base_contact≈0.10` |
| `feet_slide=-0.25`, `feet_air_time=0.50` | worse — rocking, `feet_air_time≈−0.043`, tracking degraded |

**Decision: `feet_slide=-0.25`, `undesired_contacts=None`, `feet_air_time=0.25`.** Do not raise
`feet_slide`; do not add the contact term.

**Decision: do NOT adopt the full Unitree RL Lab reward stack.** Reasons: (a) it is a different
*environment* (terrain curriculum, command curriculum, standing envs, air-time-variance/energy/
joint-pos terms working together), and our own ablations proved piecemeal porting fails; (b) most of
its terms cannot be computed in the RWM imagination (see §8); (c) the world-model experiment needs a
*non-degenerate* baseline, not a show-quality gait — and an imperfect-but-walking baseline is actually
preferable because it leaves headroom for MBPO/uncertainty to demonstrate improvement.

---

## 7. The accepted RWM pretrain baseline

`feet_slide=-0.25` was ported into the RWM Go2 config (added to the shared `RewardsCfg_TRAIN` so all
stages inherit it) and the Go2 RWM pretrain was rerun.

- **Run:** `2026-06-12_09-07-38_pretrain`, checkpoint `model_2000.pt`
- **Final:** `track_lin ≈ 0.927`, `error_vel_xy ≈ 0.204`, episode length ≈ 982,
  `base_contact ≈ 0.021`, `feet_slide ≈ −0.06`
- **Render:** body upright and supported, stepping, no belly-skate. Residual "fast low shuffle"
  (low foot clearance) is acceptable for a baseline.

This is the **accepted clean RWM pretrain baseline** and the starting point for all MBPO/RWM-U
finetuning.

---

## 8. Plain MBPO finetune — and the imagination-reward limitation

Two issues surfaced here, both instructive.

**(a) Wrong pretrain loaded.** The first finetune launch silently loaded an old (scoot) pretrain
because `load_run` / `system_dynamics_load_path` had not been repointed. Caught by inspecting the run's
`agent.yaml` (`Loading model checkpoint from …`). Fixed to load
`2026-06-12_09-07-38_pretrain/model_2000.pt`. *Process lesson: always verify the loaded checkpoint
from the run's resolved config, not the launch command.*

**(b) `KeyError: 'feet_slide'` at the first imagination step.** The finetune crashed in
`manager_based_mbrl_env.py::_post_imagination_step` because `feet_slide` is an active reward term in
the real env but is **absent from the imagination reward dictionary**. Root cause:
`feet_slide` needs per-foot world-frame velocity, which the RWM world model does **not** predict — it
predicts the compact system state (base lin/ang velocity, projected gravity, joint pos/vel/torque) and
per-body contact, not full foot kinematics. Computing `feet_slide` in imagination would require either
predicting foot velocities or running forward kinematics over the predicted joint state inside
imagination — a non-trivial extension, not a patch.

**Runtime fix (skip patch):** `_post_imagination_step` now skips reward terms not present in the
imagination reward dict:

```python
if term not in self.imagination_reward_per_step:
    continue
term_value = self.imagination_reward_per_step[term]
```

This keeps `feet_slide` active in **real** rollouts while excluding it from **imagined** ones. It is
backward-compatible (ANYmal-D has no missing terms, so it is unaffected). This is the rough runtime
version of the clean design — formally separating *imagination-computable* from *real-only* reward
terms. The clean-paper wording: *the anti-skate `feet_slide` reward is required for a non-degenerate
real-world gait but lies outside the world model's imagination reward support, so unsupported terms are
excluded from imagined rollouts.*

---

## 9. Plain MBPO finetune — result and interpretation (negative control)

- **Run:** `2026-06-12_11-07-01_finetune`; `ensemble_size=1`, `uncertainty_penalty_weight=−0.0`;
  loaded `09-07-38_pretrain/model_2000.pt`.
- **Final / late metrics:** mean reward 23→~8, episode length 982→~610, `track_lin` 0.93→0.50,
  `base_contact` **0.02 → 0.38–0.50**, `feet_slide` swinging to ≈ −0.32.

The finetune did not merely nudge the gait toward skating — it made the real-rollout distribution
**unstable/collapsing**: the robot falls (base contact) ~40–50% of episodes. Note the render caveat:
a single rendered rollout can still look stable, because the camera follows env 0; the **aggregate**
termination/length metrics are what reveal the collapse. *Judge by aggregate metrics, not one clip.*

**Confounds checked:**
- `Imagination/feet_slide = 0` — confirmed `feet_slide` is real-only, imagined-zero (the reward
  mismatch is real but **constant** across the planned plain-vs-RWM-U comparison, so it does not
  confound it).
- `Model Based/epistemic_uncertainty = 0` — with one model there is no epistemic signal, so nothing
  penalizes model error.
- Normalizers identity for state/action; `actor/critic_obs_normalization=false` — no plumbing bug.

**Interpretation:** this is **model exploitation**, the canonical model-based RL failure. With a single
imperfect world model and zero uncertainty penalty, PPO drives the policy into regions where the model
is optimistic; the policy scores well in imagination but falls in reality. The reward mismatch
(`feet_slide` absent from imagination → imagined returns over-estimate real ones) is a second,
consistent source of imagined-vs-real divergence. Both are "imagination ≠ reality," which is precisely
what the uncertainty-aware variant (RWM-U) exists to mitigate.

**Framing:** this run is a **strong negative control**, not "RWM/MBPO fails on Go2." The thesis arc is:
pretrain (good gait) → plain MBPO (degrades via model exploitation) → RWM-U (expected to prevent it).

---

## 10. RWM-U experiment (current / in progress)

The plain run varied two things from a hypothetical RWM-U run (ensemble size **and** penalty), so it
cannot isolate the uncertainty penalty. The clean design holds the ensemble constant and varies only
the penalty:

1. **Re-pretrain with an ensemble** so epistemic uncertainty is meaningful (a penalty on a 1-model
   ensemble multiplies zero). Run **`2026-06-12_13-39-03_pretrain_ens5`** (`ensemble_size=5`,
   `feet_slide=-0.25`, identity normalizers). ~3.5 h at ~6.3 s/iter; no OOM.
2. From the **same** `pretrain_ens5/model_2000.pt`, run two finetunes:
   - `finetune_ens5_pen0` — `uncertainty_penalty_weight = −0.0` (ablation: ensemble, no penalty)
   - `finetune_ens5_pen1` — `uncertainty_penalty_weight = −1.0` (method: RWM-U)
   - identical otherwise (same `feet_slide`-skip handling, normalizer, max_iterations).

Now the only variable between the two arms is the penalty, so any difference is attributable to it.
The `ensemble_size=1` plain run stays as a supplementary baseline, not the ablation partner.

**Pre-registered evaluation protocol** (decided before seeing results):
- Compare **aggregate real-rollout metrics averaged over the last ~50 iterations** (not a snapshot):
  primary = `base_contact` termination rate; secondary = episode length, `error_vel_xy`, `feet_slide`,
  mean reward.
- Track the **trajectories** of `base_contact`, mean reward, and `Model Based/epistemic_uncertainty`
  over finetuning for both arms — the expected figure is `pen0` degrading while `pen1` holds, with the
  uncertainty curve explaining why (pen1 refuses the high-uncertainty regions pen0 drifts into).
- Render both final checkpoints with the same pinned single-robot command **and** a wide multi-env
  view (judge the distribution, not one env).
- **Success criterion:** `pen1` keeps `base_contact` materially below `pen0` and preserves the upright
  gait. If `pen1 ≈ pen0`, `−1.0` is too weak online → sweep up (`−2.0`), or the degradation is
  reward-mismatch-driven and the fix is an imagination-computable anti-skate proxy (see §14).
- Caveat: n=1 vs n=1 is suggestive, not conclusive; 2–3 seeds per arm if time allows.

---

## 11. Runs registry

| run | role | key config | status / result |
|---|---|---|---|
| `2026-05-29_20-48-46_pretrain` | old Go2 pretrain | pre-rewardfix | superseded |
| `2026-06-11_12-35-49_pretrain` | scoot pretrain (rewardfix) | no feet_slide | **scoots** (negative) |
| `IsaacLab .../2026-06-11_20-29-54` | stock control, 300 it | stock flat | undertrained |
| `IsaacLab .../2026-06-11_22-03-44` | stock control, 800 it | stock flat | converged, **scoots** |
| `2026-06-12_09-07-38_pretrain` | **accepted baseline** | `feet_slide=-0.25`, ens1 | **walks**, base_contact 0.02 |
| `2026-06-12_11-07-01_finetune` | **plain-MBPO negative control** | ens1, pen=0 | **collapse**, base_contact 0.40+ |
| `2026-06-12_13-39-03_pretrain_ens5` | RWM-U pretrain | ens5 | running |
| `finetune_ens5_pen0` (planned) | ablation | ens5, pen=0 | pending |
| `finetune_ens5_pen1` (planned) | RWM-U method | ens5, pen=−1.0 | pending |

---

## 12. Reward configuration (final RWM Go2, post-fix)

Shared `RewardsCfg_TRAIN` (inherited by Pretrain/Baseline/Finetune/Visualize):
`track_lin_vel_xy_exp 1.0`, `track_ang_vel_z_exp 0.5`, `lin_vel_z_l2 −2.0`, `ang_vel_xy_l2 −0.05`,
`dof_torques_l2 −2.5e-5`, `dof_acc_l2 −2.5e-7`, `action_rate_l2 −0.01`,
`feet_air_time 0.25` (threshold 0.25), `flat_orientation_l2 −2.5`, `dof_pos_limits 0.0`,
`stand_still −1.0`, **`feet_slide −0.25`** (sensor + asset `body_names=".*_foot"`),
`undesired_contacts None`. Actuator/asset = stock `UNITREE_GO2_CFG`.

Note: we deliberately did **not** match stock's other weights (torque −2e-4, track 1.5/0.75); the
reward breakdown showed those deltas are marginal relative to tracking, and `feet_slide` is the term
that actually changed the gait. The threshold deviation (`0.25` vs stock `0.5`) is retained.

---

## 13. Reproducibility notes / runtime patches

The current method depends on **three runtime edits that the config installer does not carry** (they
live in the `robotic_world_model` / `IsaacLab` submodules, not in `scripts/phase4b/go2_rwm_config/`):

1. `manager_based_mbrl_env.py` — the `feet_slide` skip patch (§8). **Required for every finetune.**
2. `visualize.py` — `hasattr(init_imagination_history)` guard + headless lighting guard.
3. `play.py` — single-robot forward-command + follow-cam pins (stock renders only).

These must be re-applied after any submodule reset and should be formalized as tracked `.patch` files
under `scripts/phase4b/` for reproducibility. Also: the container has **no `python3`/`python`** on PATH
— always use `/isaac-sim/python.sh`; calling the installer as `python3` was the only reason it
"failed," and the manual-copy workaround risks laptop↔container config drift. The laptop git remains
the source of truth; ensure `feet_slide`, the corrected `load_run`, and `ensemble_size=5` are committed
there.

---

## 14. Open questions / next steps

- **Decided by `pen0` vs `pen1`:** does the uncertainty penalty prevent the plain-MBPO collapse? If
  yes → RWM-U validated, model exploitation confirmed. If `pen1` also collapses → either `−1.0` is too
  weak (sweep `−2.0`) or the degradation is dominated by the reward mismatch rather than dynamics
  exploitation.
- **Imagination-computable anti-skate proxy:** if the reward mismatch is the issue, replace real-only
  `feet_slide` with a term computable from predicted state — a contact-pattern penalty (per-foot
  contact *is* predicted), so imagined and real rewards agree. This is the principled upgrade, only if
  needed.
- **RWM-U beyond this:** uncertainty-error correlation and lambda sweep for online Go2 (mirroring the
  Phase 3 offline validation), once a stable RWM-U gait exists.
- **Hardware (far off, Phase 4G):** the plain-MBPO checkpoint is categorically not deployable. Deploy
  only a *final-method* policy that walks robustly across the command distribution (0.2/0.4/0.6 m/s,
  turns, near-zero), survives pushes and friction/mass/latency randomization, with verified joint
  order/signs and obs/action scaling, then a tethered low-speed first run with an e-stop. Keeping
  `feet_slide` in the real reward helps sim2real — dragging policies are exactly the ones that fail to
  move on hardware.
