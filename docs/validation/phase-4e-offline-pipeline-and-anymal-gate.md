# Phase 4E — The Offline Pipeline, the Eval Harness, and the ANYmal Gate

*Engineering / validation log. Continues Phase 4D. Work spans 2026-06-16 → 2026-06-19 on the lab
workstation.*

---

## 0. Scope

This phase builds the offline RWM-U / MOPO-PPO pipeline for Go2 from the existing ANYmal-D offline
infrastructure, builds a real-environment evaluation harness (the only honest verdict on whether a
policy walks), validates the entire apparatus on ANYmal-D against the authors' own assets, and runs the
first Go2 offline policies. It also records two distinct root-cause bugs that each silently invalidated
a result before being found — both worth carrying forward as engineering findings (see
[Engineering Findings](engineering-findings.md)).

---

## 1. Building the offline Go2 pipeline

The offline trainer (`scripts/reinforcement_learning/model_based/train.py`) was hardwired to ANYmal-D.
Porting to Go2 was four pieces, all confirmed against source before writing (not assumed):

1. **`configs/go2_flat_cfg.py`** — mirrors `anymal_d_flat_cfg.py`. Go2-specific fields: the 11-term
   reward dict matching exactly the keys the env emits (`feet_slide` deliberately omitted — it needs
   foot world-velocity, absent from the 45-dim imagined state; `undesired_contacts` omitted — Go2 uses
   `stand_still` as its collision proxy); normalizer fields left `None` so the trainer auto-computes
   mean/std from the dataset; `environment` tag placed in `ExperimentConfig` (verified against the base
   contract — top-level placement is dead code).
2. **`envs/go2_flat.py`** — faithful port of `anymal_d_flat.py`, lifting the reward/contact/termination
   logic from the online Go2 MBRL env. Uses `self._step_dt` / `self._num_envs` (the offline base
   convention), not the online env's attribute names — porting the offline file is what avoided that
   trap.
3. **`train.py` dispatch** — `go2_flat` branch in both `resolve_task_config` (→ `Go2FlatConfig`) and
   `resolve_environment_cls` (→ `Go2FlatEnv`), plus the `Go2FlatConfig`/`Go2FlatEnv` imports.
4. **dataset CSV** — there is **no dataset writer** in the repo (the shipped ANYmal CSV is an asset). A
   collector (`collect_go2_dataset.py`) was written to roll out a Go2 policy and dump the 66-column
   schema the loader expects: `state(45) | action(12) | extension(0) | contact(8) | termination(1)`.

**Dataset schema, confirmed empirically** against the shipped ANYmal `state_action_data_0.csv`: 66
columns, headerless, one row per timestep, concatenated along time. The reference is a single
continuous trajectory with **zero terminations** (a clean expert that never falls) — which set the
template for the first Go2 dataset and, later, became the thing that had to change.

---

## 2. The eval harness — and the metric that lied

Imagined metrics (imagined reward, imagined episode length) can be gamed by world-model exploitation, so
they cannot be the verdict. The harness (`eval_go2_offline_policy.py`) loads a trained policy into the
**real** Go2 env, rolls out with normal resets, and reports aggregate real metrics.

The harness immediately earned its place. It surfaced a contradiction the summary columns hid: a
no-penalty Go2 policy logged `error_vel_xy ≈ 0.216` (implying decent tracking) but
`track_lin_vel_xy_exp ≈ 0.015` (implying none). These are mathematically incompatible under the reward's
own smooth exponential. A **velocity-trace** instrument was added — logging commanded vs actual
base-frame velocity per step plus their correlation — and it adjudicated:

- actual base speed ≈ **0.03 m/s** against a commanded ≈ **0.75 m/s**;
- `corr(cmd_vx, act_vx) ≈ +0.14`, `corr(cmd_vy, act_vy) ≈ −0.01`.

**The policy was standing nearly still, ignoring the command.** `error_vel_xy` had been *flattering a
non-mover* (its mean is depressed by the standing-command fraction and resample timing). The raw trace
recomputed the true error at ~0.74 m/s — the honest number. **Lesson, carried forward: tracking is
judged by cmd-vs-actual speed and correlation from the trace, not by `error_vel_xy` or the logged reward
column.** This also re-validated the methodology: the known-good `ens5` online walker traces at actual
0.669 / cmd 0.709, `corr +0.98` — a real walker reads as a walker, so the harness is sound.

---

## 3. The ANYmal gate — validating the apparatus

Before trusting any Go2 number, the whole apparatus (method + pipeline + harness) was validated on
ANYmal-D using the **authors' own** shipped world model (`assets/models/pretrain_rnn_ens.pt`) and
dataset — `--task anymal_d_flat`, no new code.

| arm | penalty | result |
|---|---|---|
| **A** — authors' default | −1.0 | trains; offline policy evaluates to `track_lin_vel_xy_exp ≈ 0.417` |
| **B** — uncertainty-unaware | 0.0 | collapses |

Run A walking through our harness validates the method, our pipeline, and our eval end-to-end, and gives
a reference number. A-vs-B reproduces Paper 2's central MOPO-vs-unaware comparison in one shot. **This
gate is what licenses every subsequent Go2 claim.**

> **Confound recorded.** Run A uses the *authors'* world model; our Go2 runs use a world model *we* fit.
> So "Go2 ≈ ANYmal" later means "our optimizer on their WM" vs "our optimizer on our WM" — the ANYmal
> gate validates the *policy* pipeline against a known-good WM; it does not validate our *WM-fitting*
> pipeline. That separation is exactly what the [ANYmal-WM diagnostic](engineering-findings.md#the-anymal-wm-diagnostic-named-future-work)
> is designed to test, and it is named future work, not done.

---

## 4. Bug #1 — the pipeline was running against an *untrained* world model

The first Go2 offline runs (and the first interpretation of the penalty sweep) were invalidated by a
silent bug: the offline trainer builds the world-model ensemble and **only loads** it from
`resume_path`; it never fits a WM. The Go2 config had no Go2 WM to point at, so the policy was being
optimized against a freshly-initialized (random) ensemble. Penalizing ensemble disagreement on a random
ensemble penalizes noise everywhere — which is why the early penalty sweep "collapsed" uniformly and
meant nothing.

The fix added a `--wm-checkpoint` override and, more importantly, reframed the work: a real Go2 world
model has to be *produced*. Two routes — extract it from online pretrain, or fit it offline from a
dataset — set up the rest of the campaign. The early penalty sweep was discarded as run against a random
WM. See [Engineering Findings §1](engineering-findings.md#bug-1-policy-optimized-against-an-untrained-world-model).

---

## 5. The narrow-WM coverage frontier (real WM, online-pretrain source)

With a *trained* online-pretrain WM loaded via `--wm-checkpoint`, the penalty sweep was rerun. Read
through the velocity trace, the picture was a **coverage/conservatism frontier**, not a walker:

- **no penalty** → the policy exploits the WM: imagined reward high (~5.7) while real reward ≈ 0; survives
  longer than at −1.0 but does not track (actual ≈ 0.03 m/s).
- **strong penalty** → the policy freezes: the only way to stop accruing the per-step uncertainty penalty
  off the narrow data manifold is to drive the imagined rollout into an immediate predicted-terminal
  state. Imagined episode length collapses to ~1.

No penalty value produced a command-following walker. The root cause was diagnosed as **world-model
coverage**: the WM was trained on a single clean policy's near-fall-free distribution, so it is only
reliable in a thin band around that trajectory, and following arbitrary velocity commands requires
leaving that band. This is the textbook offline-RL distribution-shift failure, and it set up the curated
dataset of Phase 4F.

A separate attempt to broaden coverage by re-pretraining *with strong (±1.0) pushes* was run and traced:
the push-broadened pretrain **itself froze** (actual 0.04 m/s, `corr ~0`) — the pushes turned the
pretrain into a brace-and-stand policy, and a WM fit on a stander's buffer learns standing-as-competence.
The offline policy faithfully reproduced the freeze. **Conclusion: push amplitude couples competence and
coverage inversely and cannot deliver both; the falls it produces are exogenous shoves, the wrong kind
for an offline tracking policy.** This retired the push approach and motivated the noise-perturbed
curated dataset.

---

## 6. Status at end of Phase 4E

- Offline Go2 pipeline: **built, committed, runs end-to-end.**
- Eval harness + velocity trace: **built, validated** (catches non-movers; a known walker reads as a
  walker).
- ANYmal gate: **passed** (A 0.417, B collapses) — apparatus validated, reference number in hand.
- Bug #1 (untrained WM): **found and fixed**; early sweep discarded.
- Go2 offline against an online-pretrain WM: **does not walk** — a coverage frontier (exploit ↔ freeze),
  not a walker. Root cause = world-model coverage from narrow, near-fall-free pretrain data.
- Next (Phase 4F): build a *curated, coverage-broadened* dataset and fit a Go2 world model on it with a
  standalone trainer, then re-run the offline policy.
