# Reproduction Status

This page is the per-claim ledger for the project. It separates *execution claims* (whether the code has
been run end-to-end on local or lab hardware) from *mapping claims* (whether the paper's content has been
identified in the code). The two axes are independent and use the [convention vocabulary](../index.md#conventions)
from the landing page.

For the procedural and architectural detail behind each row, follow the cross-references. The narrative
behind the lab campaign is in the phase logs:
[4A](phase-4a-lab-validation.md) · [4B](phase-4b-go2-scaffold.md) · [4C](phase-4c-go2-gait-mbpo-rwmu.md) ·
[4D](phase-4d-online-collapse-and-pivot.md) · [4E](phase-4e-offline-pipeline-and-anymal-gate.md) ·
[4F](phase-4f-curated-wm-and-offline-walker.md), with cross-cutting bugs and confounds in
[Engineering Findings](engineering-findings.md).

## Execution claims

What has been executed, on which hardware, with what outcome.

| Claim | Status | Evidence | Required to advance |
|---|---|---|---|
| Local Isaac Lab and RWM stack runs headlessly. | Validated | Baseline task reaches PPO learning iterations. | None for local validation. |
| Baseline ANYmal-D task `Init-v0` executes locally. | Validated | See [Baseline Execution](baseline-execution.md). | None. |
| RWM pretraining executes end-to-end at reduced scale locally and writes checkpoints. | Validated at reduced scale | See [World-Model Pretraining Check](world-model-pretraining-check.md). | Superseded by lab-scale runs. |
| RWM/RWM-U training stack runs at meaningful scale on lab hardware. | Validated | Go2 pretrain/finetune and offline fits run through Isaac Sim on `rwmu-cogar-cb`. See [Phase 4C](phase-4c-go2-gait-mbpo-rwmu.md)–[4F](phase-4f-curated-wm-and-offline-walker.md). | None. |
| Go2 RWM pretrain learns velocity tracking but **scoots** (reward loophole). | Validated | `feet_slide`-free pretrain skates; confirmed by render and by stock-Go2 control also skating. See [Phase 4C](phase-4c-go2-gait-mbpo-rwmu.md). | Fixed via `feet_slide=-0.25`. |
| `feet_slide` penalty yields a clean walking Go2 pretrain baseline. | Validated | `2026-06-12_09-07-38_pretrain` (ens1) and `2026-06-12_13-39-03_pretrain_ens5` walk, `base_contact ≈ 0.02`. | None. |
| **Online** MBPO/RWM-U finetune on Go2 collapses (model exploitation). | Validated | `pen0`/`pen1`/`h32` all collapse to ~0.5–0.7 `base_contact` from a shared good state; failure is largely penalty-independent; horizon is not the lever. See [Phase 4D](phase-4d-online-collapse-and-pivot.md). | Motivation, not contribution. |
| **Offline** RWM-U / MOPO-PPO pipeline ported to Go2 and runs end-to-end. | Validated | config/env/dispatch/collector built; `--task go2_flat` trains and saves. See [Phase 4E](phase-4e-offline-pipeline-and-anymal-gate.md). | None. |
| Real-environment eval harness + velocity trace validated. | Validated | Catches non-movers (`error_vel_xy` flattered a standing policy); known-good `ens5` walker reads correctly (actual 0.669/cmd 0.709, corr +0.98). See [Engineering Findings](engineering-findings.md#the-metric-that-lied-error_vel_xy-flatters-non-movers). | None. |
| **ANYmal gate**: offline RWM-U on the authors' WM trains and evaluates to a walking policy. | Validated | Penalty −1.0 → `track_lin_vel_xy_exp ≈ 0.417`; penalty 0 collapses. Apparatus validated end-to-end. See [Phase 4E §3](phase-4e-offline-pipeline-and-anymal-gate.md#3-the-anymal-gate--validating-the-apparatus). | Confound: uses authors' WM (see below). |
| Standalone offline Go2 world-model fit produces a loadable, accurate-enough WM. | Validated | `fit_world_model.py` lifts the exact loss; strict-load architecture check passes; held-out termination AUC 0.978 (bimodal). See [Phase 4F](phase-4f-curated-wm-and-offline-walker.md). | n=1 WM; coverage half-covered. |
| Offline RWM-U produces a **command-following** Go2 walker. | Validated (seed-fragile) | Best seed (−0.25): actual 0.22 m/s, corr +0.78, `track_lin` 0.44 (≈ offline-ANYmal 0.417). Minority basin (~2/10 seeds); majority stand. Weak/slip-heavy gait. See [Phase 4F §5](phase-4f-curated-wm-and-offline-walker.md#5-the-offline-walker--and-an-honest-account-of-it). | Robustness/coverage; not yet paper-parity. |
| Our **WM-fitting pipeline** matches the authors' on ANYmal. | Not verified | The ANYmal gate used the authors' WM. Our-fit ANYmal WM never built. | The [ANYmal-WM diagnostic](engineering-findings.md#the-anymal-wm-diagnostic-named-future-work). |
| MBPO-PPO matches the paper's `0.90 ± 0.04` tracking. | Qualitatively validated at reduced scale | Online best-checkpoint tracking ~0.92 at the paper's short budget; degrades past it (exploitation). | Not the thesis target (offline is). |
| Zero-shot hardware deployment. | Not verified | No policy deployed; the deployable artifact today is the `ens5` online walker, not the offline policy. | Robustness + safety gating (Phase 4G). |

## Mapping claims

What has been identified in the code as a counterpart to a paper claim. *(Unchanged from prior; the RWM
architecture/loss/algorithm mappings below were established during P1 and remain accurate.)*

| Paper claim | Status | Cross-reference |
|---|---|---|
| RWM world model architecture (GRU base, MLP heads). | Mapped | [Implementation Analysis §4](../world-model/implementation-analysis.md#4-system-dynamics-module). |
| Dual-autoregressive training. | Mapped | [Implementation Analysis §4–§5](../world-model/implementation-analysis.md). |
| Multi-step prediction loss (Eq. 2). | Partially mapped | [Synthesis §1](../world-model/paper-to-code-synthesis.md#1-method-components). |
| MBPO-PPO algorithm (Algorithm 1). | Mapped | [Implementation Analysis §7](../world-model/implementation-analysis.md#7-model-based-runner). |
| MOPO-PPO offline optimizer with uncertainty penalty (RWM-U). | Mapped + Validated | Offline `model_based/` pipeline exercised end-to-end on ANYmal and Go2; penalty sign/scale studied. See [Phase 4E](phase-4e-offline-pipeline-and-anymal-gate.md)–[4F](phase-4f-curated-wm-and-offline-walker.md). |
| Autoregressive evaluation under noise injection (Fig. 3b). | Mapped | Noise scales `[0.1…0.8]` match the figure. |
| Reward function (Section A.1.2). | Partially mapped | 11 active terms; Go2 set documented in [Phase 4C §12](phase-4c-go2-gait-mbpo-rwmu.md). |
| Policy observation (48-dim) vs system state (45-dim). | Mapped | Reconstructed in imagination from predicted state + command + previous action. |
| State predicted as residual; state loss as sampled MSE. | Discrepancy noted | [Synthesis §4.1–§4.2](../world-model/paper-to-code-synthesis.md). |
| Termination head trained with naive BCE (class-imbalance blind). | Discrepancy noted + addressed | Online head collapses to all-negative ([Phase 4D §3](phase-4d-online-collapse-and-pivot.md)); offline fit injects `pos_weight` ([Phase 4F §2](phase-4f-curated-wm-and-offline-walker.md)). |

## Notes

The picture changed substantially from the early-project ledger: the RWM-U path is no longer a
"structural placeholder" — it has been exercised end-to-end, offline, on both ANYmal (validation gate)
and Go2 (the contribution). The honest current frontier is **robustness and coverage**: a command-following
offline Go2 walker exists but is a seed-fragile minority basin at offline-ANYmal parity, the world model
is `n=1`, and the cleanest next experiment is the [ANYmal-WM diagnostic](engineering-findings.md#the-anymal-wm-diagnostic-named-future-work)
to localize whether the remaining gap is the WM-fitting pipeline or the dataset coverage.
