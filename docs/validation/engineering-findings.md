# Engineering Findings, Confounds, and Threats to Validity

*Cross-cutting page. These findings span Phases 4D-4F and are collected here because each is reusable
beyond the run that produced it, three are bugs whose diagnosis is itself a result, and the rest is the
confound ledger a faithful reproduction must surface.*

---

## Bug #1: policy optimized against an untrained world model

**Symptom.** The first Go2 offline policy runs, and the first penalty sweep, produced uniform meaningless
collapse regardless of penalty value.

**Root cause.** The offline trainer (`model_based/train.py`) **builds** the `SystemDynamicsEnsemble` and
**loads** it from `resume_path`; it never *fits* a world model. The Go2 config had no Go2 world model to
load, so the policy was being optimized against a freshly-initialized (random) ensemble. A MOPO penalty
on a random ensemble penalizes disagreement *everywhere*, hence the uniform collapse.

**Fix.** Added a `--wm-checkpoint` override; reframed "produce a Go2 world model" as a required, separate
step (extract from online pretrain, or fit offline, Phase 4F took the fit route). The early penalty
sweep was discarded.

**Lesson.** Confirm what a pipeline *loads* from its resolved config, not what the launch command implies
- the same class of error as the Phase 4C "wrong pretrain silently loaded" bug. A green run against a
random model looks exactly like a real experiment until you check the checkpoint provenance.

---

## Bug #2: normalizer-consistency freeze

**Symptom.** The first policy run against the *fitted* curated world model froze: imagined episode length
pinned at exactly **1.00** for all 500 iterations, every env, zero variance.

**Root cause.** The offline pipeline's `Dataset` z-scores state and action (mean/std computed from the
data), and the imagination loop runs in that normalized space. But `fit_world_model.py` trained the WM on
**raw** values. At deployment the WM was fed normalized inputs it had never seen; every head misfired; the
termination gate fired on step 1 of every imagined episode; episode length pinned at 1. The measured
`s_std[:3] ≈ [0.42, 0.43, 0.19]` means those state dims were amplified 2-5× at deployment, the misfire,
quantified.

**Why it hid.** Both the fit and the held-out validation ran in *raw* space, internally consistent, so
neither could see the mismatch. Only *deployment* ran in normalized space, and nothing tested deployment
space. The methodology hole was the absence of a deployment-space check, not the bug itself.

**Fix.** Z-score state/action in the fit using mean/std from the same CSV (contact/termination left raw);
point the Go2 config's `dataset_folder` at the curated data so the deployment normalizer is computed from
the same rows. All three surfaces, fit, deployment normalizer, imagination resets, then read identical
data in identical space.

**Lesson.** The shape of a failure is diagnostic: a flat, zero-variance value is mechanical (a mismatch),
not statistical (drift). Reading the dataset class settled it in minutes; a diverse-data rebuild would
have re-frozen at 1.00 for days.

---

## The metric that lied: `error_vel_xy` flatters non-movers

`error_vel_xy` (an IsaacLab command-manager metric) reported ~0.216 for a policy whose true tracking
reward was ~0.015, mathematically incompatible under the reward's smooth exponential. The velocity trace
(cmd vs actual base-frame speed + correlation) showed the policy standing at ~0.03 m/s against a ~0.75
command, `corr ~0`. `error_vel_xy`'s mean is depressed by the standing-command fraction and resample
timing, so it under-reports the error of a non-mover.

**Standing rule.** Tracking is judged by cmd-vs-actual speed and correlation from the velocity trace -
**not** by `error_vel_xy` or the logged `track_lin_vel_xy_exp` column. A known-good walker (`ens5`) reads
correctly through the trace (actual 0.669 / cmd 0.709, `corr +0.98`), which is how the harness itself was
validated.

---

## Confound ledger (threats to validity)

Surfaced in-flight and recorded so they are answered before they are asked.

| confound | what it limits | status |
|---|---|---|
| **World model is `n=1`.** | Seed studies vary only the policy, so they measure *policy-init* robustness, not *method* robustness. | Stated as a limitation; not resolved. |
| **ANYmal-vs-Go2 ensemble mismatch.** | An early "robot is the only difference" comparison carried an ensemble-size difference (ANYmal finetune was ens1; Go2 ens5), so it varied two things. | Caught; the clean within-Go2 ens5 comparisons are the ones used. |
| **Push-falls are the wrong kind of falls.** | Falls from exogenous shoves live in states an offline tracking policy can never enter, and teach brace-and-stand. | Caught; retired the push approach in favor of noise-on-competent (goal-directed falls). |
| **Held-out negatives are on-manifold.** | The termination gate's false-alarm rate is conservative (some "walking" negatives are real near-falls); the gate validates fall *detection*, not off-manifold behavior. | Stated; the deployment-space behavior is what Bug #2 exposed. |
| **"Offline" means frozen-WM, not offline-data RL (strict).** | The world model is still produced by an upstream stage; the *policy* is what is trained without env interaction. | Stated explicitly; defensible standard setup, but not overclaimed. |
| **`feet_slide` absent from imagined reward.** | Imagined returns over-estimate real ones; the offline gait is slip-heavy partly because skating is never penalized in imagination. | Stated; an imagination-computable anti-skate proxy is the principled upgrade. |
| **Held-out AUC is optimistic.** | 68 positives vs 2000 negatives; AUC flattered by easy negatives. Report recall-at-operating-point and the bimodal coverage instead. | Stated. |

---

## The ANYmal-WM diagnostic (named future work)

The single cleanest experiment that would separate "our WM-fitting pipeline is weak" from "our Go2 data
is too narrow":

> Build an ANYmal world model with **our** `fit_world_model.py` on the authors' ANYmal dataset, run the
> offline policy against *our-built* ANYmal WM, and compare to the **0.417** obtained from the authors'
> WM (Phase 4E ANYmal gate).

- If our-WM ANYmal lands near 0.417 and is seed-robust → our WM-fitting pipeline is sound, and the Go2
 weakness is purely a **data-coverage** problem → the fix is the diverse-data rebuild
 (random/medium/expert behavior policies, Paper 2's recipe).
- If our-WM ANYmal is also weak/fragile → our WM **fit itself** differs from theirs, and no Go2 data will
 fix it → a different fix entirely.

One ~90-minute fit plus a couple of trace evals decides which world we are in, *before* committing days to
a Go2 data rebuild. Minor prerequisite: `fit_world_model.py` currently hardcodes the Go2 config and must
be pointed at ANYmal's dims/config (a small edit, not a rewrite).

---

## Process lessons carried through the campaign

- **Verify the loaded/resolved config before every run**: both bugs above, and the Phase 4C
 wrong-pretrain bug, were checkpoint/config provenance failures.
- **The container has no `python3`/`python`** on PATH, always `/isaac-sim/python.sh`; the offline fit and
 trace are pure-PyTorch but still launch through it.
- **`PYTHONUNBUFFERED=1`** for detached runs, Isaac's `simulation_app.close()` hard-exits without
 flushing Python's stdout, so buffered summaries are otherwise lost.
- **Shared multi-tenant GPU**: `nvidia-smi` before every launch; the container periodically loses GPU
 visibility (NVML cgroup revocation), fixed by `docker restart` (bind mounts preserve all data).
- **Judge by aggregate metrics and the trace, never one rendered clip**: the render camera follows env 0
 and can show a stable rollout while the distribution collapses.
