# Roadmap

This roadmap tracks the project at the level of research phases, not individual run logs. It should answer:

- what has been validated,
- what is currently active,
- what remains blocked or future,
- what evidence is needed before moving to the next stage.

Detailed run metrics, accepted checkpoints, failed runs, and command transcripts belong in the validation and experiment-log pages.

## Success criteria

Success for the project, in order:

1. Reproduce the base RWM pipeline on lab hardware and match the published behavior within reasonable margin.
2. Understand and validate the RWM pretrain and MBPO finetune stages, including their failure modes.
3. Activate and validate the RWM-U path, showing that the uncertainty signal changes policy optimization in a measurable way.
4. Port the pipeline to Unitree Go2 in simulation and obtain a stable, non-skating locomotion policy.
5. Compare plain MBPO against uncertainty-penalized RWM-U on Go2 from a controlled shared pretrain.
6. Prepare the Go2 policy for real-hardware deployment through robustness checks, safety procedures, and deployment-gate validation.
7. Identify and implement at least one research contribution or extension on top of this foundation.

The key ordering principle remains:

```text
reproduction -> understanding -> controlled modification -> Go2 transfer -> hardware -> contribution
```

## P0: Local setup and baseline validation

**Status: complete.**

The local environment was brought up, the upstream baseline was executed headlessly, and the project state was frozen in `manifests/`.

The baseline `Init-v0` task reached PPO learning iterations on the local machine. This established the first working reference state from which later phases proceeded.

Outputs:

- working local Isaac Lab / RWM environment,
- baseline headless launch validation,
- frozen manifest records,
- initial documentation structure.

## P0.5: Repository organization and reproducibility

**Status: mostly complete; hygiene remains ongoing.**

The top-level repository was created, the documentation site was deployed via GitHub Pages and MkDocs Material, upstream dependencies were converted to submodules, and forks were introduced for repositories expected to receive project-specific changes.

The documentation was reorganized around project use rather than raw notes.

Completed:

- top-level research workspace created,
- `docs/`, `scripts/`, `manifests/`, and `upstream/` roles established,
- submodule layout documented,
- GitHub Pages documentation site deployed,
- fork/pinned-submodule policy documented,
- Go2 config install mechanism added under `scripts/phase4b/`.

Remaining hygiene:

- keep submodule pointers pushed and reproducible,
- verify clone-with-submodules workflow from a fresh checkout when moving machines,
- keep runtime patches formalized as tracked patch files where possible,
- avoid drift between `scripts/phase4b/go2_rwm_config/` and the installed runtime copy under `upstream/robotic_world_model/`.

## P1: RWM understanding and validation

**Status: largely complete.**

The goal was to understand the base RWM pipeline before extending toward RWM-U or transferring to Go2.

Completed:

- reduced-scale `Pretrain-v0` validated locally,
- four task modes enumerated and documented,
- runtime pipeline traced from launcher to checkpoint,
- RWM paper analyzed with equations, figures, and quantitative results,
- upstream codebase analyzed architecturally,
- paper and code mapped against each other,
- major paper/code discrepancies documented,
- RWM pretraining cleanup bug identified and fixed in `rsl_rl_rwm`.

Relevant pages:

- [World-Model Pretraining Check](../validation/world-model-pretraining-check.md)
- [Task Modes](../world-model/task-modes.md)
- [Runtime Pipeline](../world-model/runtime-pipeline.md)
- [Paper Analysis](../world-model/paper-analysis.md)
- [Implementation Analysis](../world-model/implementation-analysis.md)
- [Paper-to-Code Synthesis](../world-model/paper-to-code-synthesis.md)

Remaining:

- final documentation polish,
- second-pass verification that all parameter values cited in analysis pages match the current pinned/forked code.

## P1.5: Full RWM training on lab hardware

**Status: partially superseded by active lab-side Go2 experiments.**

The original goal was to run default-scale ANYmal-D `Pretrain-v0` and `Finetune-v0` on the lab workstation before moving to Go2.

That goal remains scientifically clean, but the project has already progressed into Go2 simulation on lab hardware. Therefore this phase is now split into two interpretations:

1. **Original reproduction objective:** full ANYmal-D reproduction against paper-reported metrics.
2. **Practical project objective:** lab hardware can run the RWM/RWM-U training stack at meaningful scale.

Completed toward the practical objective:

- lab/container execution path works,
- RWM/RWM-U training commands run on the workstation,
- Go2 RWM pretrain and finetune jobs run through Isaac Sim,
- model-based finetune crash modes were identified and patched,
- ensemble-5 pretraining is running for the RWM-U comparison.

Still valuable but not blocking Go2 work:

- full ANYmal-D default-scale reproduction,
- multi-seed ANYmal-D comparison to paper-reported velocity-tracking reward,
- strict reproduction of paper-reported `0.90 ± 0.04` RWM finetune behavior.

This phase should not block the Go2 RWM-U ablation, but the documentation should be honest that a full ANYmal-D paper reproduction is not yet the main completed result.

## P2: RWM-U analysis and validation

**Status: active / in progress.**

The goal is to understand and validate the uncertainty-aware extension: ensemble world models, epistemic uncertainty estimation, and uncertainty-penalized imagined rollouts.

Originally planned validation items:

- set `ensemble_size > 1`,
- verify ensemble construction,
- activate an uncertainty penalty,
- verify the imagined reward includes the uncertainty penalty,
- reproduce the paper's offline/mixed-dataset RWM-U result.

Current project state:

- `ensemble_size = 5` Go2 pretraining has been launched,
- the clean ablation design has been chosen,
- plain MBPO negative control has already shown aggregate instability,
- the main RWM-U comparison will be `pen0` vs `pen1` from the same ensemble-5 pretrain.

Important update:

The primary RWM-U validation path for this project is now Go2-focused rather than a direct ANYmal-D offline reproduction first. The strict paper reproduction remains useful, but the active research question is:

```text
Does uncertainty-penalized RWM-U prevent the Go2 plain-MBPO finetune from exploiting the world model and degrading the pretrain gait?
```

Required next evidence:

- ensemble-5 pretrain finishes and passes the pretrain gate,
- `finetune_ens5_pen0` runs from the same checkpoint with `uncertainty_penalty_weight = -0.0`,
- `finetune_ens5_pen1` runs from the same checkpoint with `uncertainty_penalty_weight = -1.0`,
- both arms are compared by last-window aggregate metrics, not one video.

Primary comparison metric:

```text
Episode_Termination/base_contact averaged over the last ~50 iterations
```

Secondary metrics:

```text
Train/mean_episode_length
Train/mean_reward
Metrics/base_velocity/error_vel_xy
Episode_Reward/feet_slide
Model Based/epistemic_uncertainty
Episode_Reward/track_lin_vel_xy_exp
```

## P3: Go2 simulation transition

**Status: active; core port exists.**

The original goal was to port the pipeline from ANYmal-D to Unitree Go2 in simulation.

Completed:

- Go2 Isaac Lab asset identified,
- Go2 joint/body/actuator inventory documented,
- Go2 joint order verified,
- Go2 `system_state` layout established as 45-dimensional,
- Go2 RWM task family created and registered,
- Go2 config source of truth placed under `scripts/phase4b/go2_rwm_config/`,
- Go2 config installer added,
- identity normalizers selected and validated as the current consistent choice,
- stock Go2 reward failure mode diagnosed as skating/scooting,
- `feet_slide = -0.25` selected as the working anti-skate reward patch,
- Go2 RWM pretrain produced a usable upright gait,
- plain MBPO finetune from that pretrain completed after the reward-skip patch and produced a useful negative control.

Accepted Go2 pretrain baseline:

```text
run:        2026-06-12_09-07-38_pretrain
checkpoint: model_2000.pt
reward:     feet_slide = -0.25
behavior:   usable upright non-belly-skating gait
```

Plain MBPO negative control:

```text
run:        2026-06-12_11-07-01_finetune
ensemble:   1
penalty:    -0.0
result:     aggregate degradation / distributional instability
```

Key lesson:

```text
A single rendered rollout can look acceptable while aggregate real-sim metrics show frequent base-contact termination.
```

Remaining in this phase:

- finish ensemble-5 Go2 pretrain,
- run controlled `pen0` vs `pen1` RWM-U finetunes,
- render final policies under matched conditions,
- plot aggregate trajectories across finetuning,
- decide whether the uncertainty penalty meaningfully improves distributional stability.

## P4: Go2 hardware deployment

**Status: planned; explicitly gated.**

The goal is to deploy a trained Go2 policy on real hardware.

This phase must not begin from the current plain-MBPO finetune. A policy with high base-contact termination in simulation is not hardware-ready.

Hardware deployment is gated on a final Go2 policy that passes simulation robustness tests.

Required gates before hardware:

- stable aggregate real-sim metrics,
- low base-contact termination,
- long episode length near timeout,
- controlled foot sliding,
- acceptable render under multiple commands,
- robustness to command variation,
- robustness to friction/mass/latency randomization,
- push/disturbance tests,
- verified joint order and action signs,
- verified observation scaling and action scaling,
- low-speed tethered first run plan,
- emergency-stop procedure,
- human safety plan.

Current deployable candidate:

```text
none yet
```

The accepted pretrain is a useful simulation baseline, but the hardware candidate should come from the final validated Go2 method policy, ideally the RWM-U arm if it proves more stable.

## P5: Research contribution

**Status: emerging, not finalized.**

The contribution phase no longer needs to wait until every earlier phase is complete. A concrete contribution direction is emerging from the Go2 transfer experiments:

```text
Plain MBPO can degrade a good Go2 pretrain through model exploitation / imagined-reward mismatch, while RWM-U may prevent that degradation by penalizing epistemically uncertain imagined rollouts.
```

Current candidate contribution:

```text
A controlled Go2 study of model exploitation in RWM-style MBPO, with uncertainty-aware RWM-U as the mitigation.
```

Evidence already available:

- Go2 pretrain can learn a usable anti-skate gait with real `feet_slide = -0.25`,
- plain MBPO finetune from that pretrain degrades aggregate real-sim metrics,
- imagined `feet_slide` is zero/skipped,
- epistemic uncertainty is zero in ensemble-1 plain MBPO,
- a controlled ensemble-5 `pen0` vs `pen1` comparison is now set up.

Evidence still needed:

- ensemble-5 pretrain passes the baseline gate,
- `pen0` and `pen1` are run from the same checkpoint,
- `pen1` materially improves aggregate stability versus `pen0`,
- uncertainty trajectories support the mechanism claim,
- results are robust enough to avoid overclaiming from a single seed.

If RWM-U succeeds:

```text
Contribution = uncertainty penalty prevents model-exploitation collapse during Go2 MBPO finetuning.
```

If RWM-U does not succeed at `-1.0`:

```text
Contribution direction shifts to diagnosing whether the collapse is dominated by reward mismatch, weak penalty scaling, or world-model error not captured by the current uncertainty signal.
```

Either outcome is useful as long as the comparison is controlled.

## Current near-term checklist

The active roadmap item is no longer "port Go2" in the abstract. It is the concrete RWM-U ablation.

Immediate next steps:

1. Let `2026-06-12_13-39-03_pretrain_ens5` finish.
2. Confirm `model_2000.pt` exists.
3. Check final pretrain aggregate metrics.
4. Render `pretrain_ens5/model_2000.pt`.
5. If the pretrain is clean, launch `finetune_ens5_pen1`.
6. Launch `finetune_ens5_pen0` from the same checkpoint.
7. Compare last-window aggregate metrics.
8. Plot trajectories for base contact, reward, episode length, and epistemic uncertainty.
9. Render both final checkpoints under matched conditions.
10. Decide whether RWM-U mitigates the plain-MBPO failure.