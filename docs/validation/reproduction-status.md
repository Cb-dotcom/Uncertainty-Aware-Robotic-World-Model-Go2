# Reproduction Status

This page is the per-claim ledger for the project. It separates *execution claims* (whether the code has been run end-to-end on local or lab hardware) from *mapping claims* (whether the paper's content has been identified in the code). The two axes are independent and use the [convention vocabulary](../index.md#conventions) from the landing page.

For the procedural and architectural detail behind each row, follow the cross-references.

## Execution claims

What has been executed, on which hardware, with what outcome.

| Claim | Status | Evidence | Required to advance |
|---|---|---|---|
| Local Isaac Lab and RWM stack runs headlessly. | Validated | Baseline task reaches PPO learning iterations (`manifests/baseline_state_*.txt`). | None for local validation. |
| Baseline ANYmal-D task `Init-v0` executes locally. | Validated | See [Baseline Execution](baseline-execution.md). | None. |
| RWM pretraining pipeline executes end-to-end at reduced scale on local hardware and writes checkpoints. | Validated at reduced scale | Reduced-scale Pretrain-v0 run produced `model_0.pt`, `model_1.pt`. See [World-Model Pretraining Check](world-model-pretraining-check.md). | Lab-scale pretraining for full validation. |
| Default RWM pretraining configuration runs on local hardware. | Not verified | Default config exceeds 8 GB VRAM during system-dynamics loss. | Lab workstation with sufficient VRAM. |
| RWM pretraining at the paper's full scale ($M=32$, $N=8$, batch 1024, 2500 iterations) on lab hardware. | Not verified | Lab access not yet exercised. | Lab workstation execution. |
| Trained dynamics model has useful predictive accuracy on held-out trajectories. | Not verified | Reduced-scale run was not configured for predictive performance. | Full pretrain plus held-out forecast evaluation. |
| Imagination-based finetuning (`Finetune-v0`) runs end-to-end. | Not verified | Upstream config requires a pretrained dynamics checkpoint that does not exist locally. See [Checkpoint and Finetune Status](checkpoint-and-finetune-status.md). | Generate or obtain a valid dynamics checkpoint and a project-specific finetune config. |
| MBPO-PPO produces a policy whose performance matches the paper's reported `0.90 ± 0.04` velocity-tracking reward. | Not verified | No finetuning has been run. | Finetune execution and benchmark evaluation. |
| RWM-U + MOPO-PPO configuration (`ensemble_size > 1`, `uncertainty_penalty_weight > 0`) runs end-to-end. | Not verified | Path not yet exercised. The hooks exist in code but have not been activated. | Activate config, validate at reduced scale, then full scale. |
| Zero-shot deployment of trained policy on hardware. | Not verified | No trained policy has been deployed. | Trained MBPO-PPO policy plus hardware availability. |

## Mapping claims

What has been identified in the code as a counterpart to a paper claim.

| Paper claim | Status | Cross-reference |
|---|---|---|
| RWM world model architecture (GRU base, MLP heads). | Mapped | [Implementation Analysis §4](../world-model/implementation-analysis.md#4-system-dynamics-module). |
| Dual-autoregressive training (inner + outer autoregression). | Mapped | [Implementation Analysis §4 to §5](../world-model/implementation-analysis.md). |
| Multi-step prediction loss (Eq. 2). | Partially mapped | [Synthesis §1](../world-model/paper-to-code-synthesis.md#1-method-components); discrepancy on $L_o$ form. |
| MBPO-PPO algorithm (Algorithm 1). | Mapped | [Implementation Analysis §7](../world-model/implementation-analysis.md#7-model-based-runner). |
| Imagined action selection (Eq. 3). | Mapped | [Implementation Analysis §9](../world-model/implementation-analysis.md#9-imagination-environment). |
| Reward function (paper Section A.1.2, twelve terms). | Mapped | [Synthesis §2](../world-model/paper-to-code-synthesis.md#2-reward-terms-paper-section-a12). |
| State mean predicted directly (paper does not specify) versus residual prediction (code). | Discrepancy noted | [Synthesis §4.1](../world-model/paper-to-code-synthesis.md#41-state-mean-predicted-as-a-residual). |
| State loss as Gaussian NLL (architecturally implied) versus sampled MSE (active code). | Discrepancy noted | [Synthesis §4.2](../world-model/paper-to-code-synthesis.md#42-state-loss-is-sampled-mse-not-gaussian-nll). |
| Imagination scale: 4096 envs × 100 steps per iteration (paper) versus 8192 envs × 24 steps (code). | Partially mapped | [Synthesis §4.3](../world-model/paper-to-code-synthesis.md#43-imagination-scale-differs-from-papers-stated-values). |
| Uncertainty handling (RWM-U paper, not RWM paper) hooks present in code but switched off. | Discrepancy noted | [Synthesis §4.4](../world-model/paper-to-code-synthesis.md#44-uncertainty-hooks-present-but-inactive). |

## Notes

The execution claims marked "Not verified" are not failures; they are blocked on lab access, lab access plus checkpoint generation, or hardware deployment that is later in the project. The mapping claims marked "Discrepancy noted" are findings to surface to a supervisor as part of a faithful reproduction effort, not as defects in the codebase.

The single execution claim that does qualify as an active blocker is the dynamics checkpoint dependency for `Finetune-v0`, which is documented in [Checkpoint and Finetune Status](checkpoint-and-finetune-status.md) along with the path forward.

The RWM-U row in the execution table is currently a structural placeholder. The corresponding paper analysis and implementation analysis will be added under a parallel `uncertainty-aware/` folder once that work begins.