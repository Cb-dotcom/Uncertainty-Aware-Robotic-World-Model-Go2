# Paper Analysis: Uncertainty-Aware Robotic World Model

This page is the paper-faithful description of *Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots* by Li, Krause, and Hutter (ETH Zurich, arXiv:2504.16680v3, dated 8 January 2026 on the PDF cover page). The paper's GitHub citation lists the year as 2025, reflecting the original v1 posting; the v3 PDF currently in this repository is the January 2026 revision used as the source for this page. It uses the paper's notation verbatim and includes the equations and quantitative results from the paper.

The implementation as found in the upstream codebase is described separately on the [Implementation Analysis](implementation-analysis.md) page (forthcoming). The mapping between paper claims and code (including discrepancies and the inactive paths that activate it) is on the [Paper-to-Code Synthesis](paper-to-code-synthesis.md) page (forthcoming). This page makes no claims about implementation; it describes only what the paper says.

This page records the current reading of the paper. Exact figure and table values should be rechecked against the paper PDF before they are used as final reproduction targets.

The base RWM method this paper builds on is documented on the [RWM paper analysis](../world-model/paper-analysis.md). RWM-U inherits the dual-autoregressive training scheme from RWM and adds uncertainty estimation on top. A reader unfamiliar with RWM should read that page first.

## 1. Problem and motivation

### 1.1 What changes between RWM and RWM-U

The original RWM paper trained world models *online*: the policy collected fresh real-environment interactions at every iteration, the dynamics model was updated on those interactions, and the policy was optimized through imagined rollouts in the updated model. The online interaction loop was what kept the model from drifting too far from reality, because every iteration brought in fresh data from the policy's current state distribution.

RWM-U targets the *offline* setting: the dynamics model and the policy are trained entirely from a fixed dataset of pre-collected transitions, with no further environment interaction allowed. This is the regime where the data is what it is, the policy cannot probe the environment to fix gaps, and any model error in regions outside the dataset's coverage cannot be corrected by collecting more data.

The offline setting is appealing for robotics because real-world data is expensive, slow, and often risky to collect. If you have a dataset of past robot interactions, the offline pipeline lets you train new policies from it without needing additional hardware time. But it introduces a problem that online MBRL did not have:

- *Compounding errors* in the world model can no longer be corrected by additional interaction.
- *Distribution shift* between the dataset's state-action coverage and the regions the policy wants to explore is unbounded; the policy can drive the model into regions where it has no training data, producing nonsense predictions.

The paper's proposal: extend RWM with *uncertainty estimation*, so that the world model not only predicts the next observation but also reports how confident it is in that prediction. Then use the uncertainty signal during policy optimization to penalize trajectories that pass through high-uncertainty regions. The policy is thereby steered toward regions of the state space where the model's predictions are reliable.

### 1.2 Stated contributions

The paper makes three contributions:

1. *RWM-U*: an extension of RWM that quantifies epistemic uncertainty via bootstrap ensembles applied to the prediction heads on top of a shared recurrent base, while preserving the dual-autoregressive training scheme.
2. *MOPO-PPO*: a policy optimization method that adapts the MOPO uncertainty-penalized framework (Yu et al., 2020, originally built on top of SAC) to the on-policy PPO algorithm, replacing the reward with a penalized version $\tilde r = r - \lambda u$.
3. *Empirical demonstration on real hardware*: deployment of policies trained entirely from offline datasets on ANYmal D and Unitree G1, including configurations that mix simulated and real-world data. The paper positions this as a first demonstration of offline model-based RL producing deployable policies on real legged hardware (Section 1, Section A.5).

The paper also positions itself against the typical offline RL benchmark setting in Section A.5, arguing that off-policy value-based methods (CQL, IQL, COMBO) do not transfer well to large-scale legged locomotion, and that on-policy PPO with model-based augmentation is the right design choice for this domain.

## 2. Foundational concepts

The RWM paper analysis already covers POMDPs, world models, autoregressive prediction, and privileged information. This page assumes those are understood. The new concepts that RWM-U introduces are listed here.

### 2.1 Offline reinforcement learning

In offline RL, the agent has access to a fixed dataset $\mathcal{D} = \{(o_t, a_t, r_t, o_{t+1})\}$ collected by one or more *behavior policies* (which may be very different from the policy the agent wants to learn). Once the dataset is collected, no further environment interaction is permitted. Every component of the learning system, including dynamics modeling, long-horizon rollouts, and policy optimization, must operate entirely on the static dataset.

The agent cannot probe the environment to fill gaps in the dataset. If the dataset under-covers some region of the state space, that region remains under-covered for the entire training run. This is what the literature calls *distribution shift*: the policy may want to take actions whose consequences are not represented in the dataset, and any model trained on that dataset will produce uncertain or wrong predictions in those regions.

Section A.6 of the paper makes this precise: in offline RL, "the dataset may contain only a limited number of behaviors, be biased toward safe or suboptimal trajectories, and lack critical transitions such as failures or recovery maneuvers that are naturally observed during online exploration."

### 2.2 Epistemic versus aleatoric uncertainty

These two terms come from Bayesian statistics and capture qualitatively different types of uncertainty.

*Aleatoric* uncertainty is the inherent stochasticity of the environment. Even with perfect knowledge of the system dynamics, the next observation is not deterministic, because of sensor noise, actuator imprecision, micro-dynamics not captured by the state, etc. Aleatoric uncertainty is *irreducible*: collecting more data does not reduce it.

*Epistemic* uncertainty is uncertainty about the model itself. The model has been trained on finite data, so its parameters are not pinned down exactly. Different training runs would produce different models. Epistemic uncertainty is *reducible*: collecting more data, or choosing a better model class, reduces it. In the limit of infinite data, epistemic uncertainty goes to zero.

For RWM-U, the distinction matters because:

- *Aleatoric* uncertainty is captured by each ensemble member's predicted standard deviation $\sigma$. The model says "I think the next observation is $\mu$, with this much noise around it." This is the per-member uncertainty.
- *Epistemic* uncertainty is captured by the *disagreement across ensemble members*. If all members predict similar means, the model is confident; if they disagree, the model knows it is operating in a region where its parameters are not well-constrained.

The penalty in MOPO-PPO uses *epistemic* uncertainty, because it is the signal of "this region was not well-covered by the training data." Aleatoric uncertainty is irreducible noise and does not indicate dataset gaps.

### 2.3 Bootstrap ensembles

A bootstrap ensemble is a set of $B$ independently trained models, each trained on a slightly different version of the dataset (typically obtained by sampling with replacement, the "bootstrap" sampling procedure). The paper motivates this with the bootstrap-ensemble literature (Bickel and Freedman, 1981, cited): disagreement among independently initialized or bootstrapped heads is used as a practical proxy for epistemic uncertainty.

In modern deep MBRL, exact bootstrap sampling is often relaxed: the ensemble members may be trained on the same dataset but with different random initializations, or on bootstrapped batches drawn online during training. The key property preserved is *independence*: the ensemble members must be diverse enough that their predictions disagree in regions where the data is sparse.

RWM-U's specific implementation choice is to share the recurrent base across ensemble members and only ensemble the prediction heads. This is a computational simplification: training $B$ full networks is expensive, while sharing the base means only the heads need to be replicated. The trade-off is that the diversity across heads is bounded by what one shared base allows, so the epistemic uncertainty signal may be smaller than a fully independent ensemble would produce. The paper's empirical results suggest this is sufficient in practice, but the design choice is worth noting because it differs from textbook deep ensembles.

### 2.4 MOPO (the framework being adapted)

MOPO (Model-based Offline Policy Optimization, Yu et al., 2020) is the method RWM-U adapts. The original MOPO uses an SAC-style off-policy algorithm with a model-uncertainty penalty in the reward:

$$
\tilde r(s, a) = r(s, a) - \lambda \cdot u(s, a)
$$

where $u(s, a)$ is the model's predicted uncertainty for the state-action pair, and $\lambda$ is a tunable penalty coefficient. The policy is then optimized to maximize expected $\tilde r$ rather than $r$, which steers it toward regions where the model is confident.

RWM-U keeps MOPO's uncertainty-penalized reward structure but replaces the uncertainty estimator with variance across RWM-U ensemble means, and replaces SAC with PPO. The combination it calls MOPO-PPO.

Two practical points stressed by the paper (Section A.5):

- Standard MOPO operates on *short* one-step rollouts. RWM-U operates on *long* rollouts of up to 100 steps, in which the uncertainty penalty must be propagated and managed across the full horizon. This is a nontrivial extension.
- Standard MOPO uses SAC. The paper argues PPO is the right choice for legged locomotion based on contemporary practice in the field.

## 3. RWM-U: the world model

### 3.1 Setup and notation (inherited from RWM)

The RWM-U paper's preliminaries restate the RWM training objective to anchor the new content. From Section 3.2:

$$
L = \frac{1}{N} \sum_{k=1}^{N} \alpha^k \, L_o\!\left(o'_{t+k},\, o_{t+k}\right) \quad (1)
$$

with the autoregressive prediction:

$$
o'_{t+k} \sim p_\phi\!\left(\,\cdot \mid o_{t-M+k:t},\; o'_{t+1:t+k-1},\; a_{t-M+k:t+k-1}\right) \quad (2)
$$

These are equations 1 and 2 of the RWM-U paper (re-derived from the RWM paper), with $M$ the history horizon and $N$ the forecast horizon. $L_o$ is the per-step prediction loss (the paper does not specify a particular form here; the actual code uses sampled MSE, see implementation analysis).

### 3.2 The ensemble formulation

The new content is in Section 4.1. The paper extends RWM by introducing $B$ bootstrap ensemble members, applied to the prediction heads after a shared recurrent feature extractor (e.g., a GRU). Each ensemble member $b$ predicts a Gaussian over the next observation:

$$
o'^{\,b}_{t+k} \sim \mathcal{N}\!\left(\mu^{b}_{o_{t+k}},\; \sigma^{2,b}_{o_{t+k}}\right) \quad (3)
$$

The key architectural choice is stated explicitly in Section 4.1: *"we apply bootstrap ensembles to the observation prediction head of the model after a shared recurrent feature extractor (e.g., a GRU)."* The ensemble lives in the prediction heads, not in the entire model. The recurrent base is shared. Each ensemble member has its own mean head and standard deviation head, but they all consume the same shared base output.

The training objective is the average loss over all ensemble members, following Eq. 1. Each member is trained with the dual-autoregressive scheme inherited from RWM.

### 3.3 The two uncertainty signals at inference

At inference, the paper defines two outputs from the ensemble (Eq. 4):

$$
o'_{t+1} = \mathbb{E}_b\!\left[\mu^{b}_{o_{t+1}}\right], \qquad u_{t+1} = u_{p_\phi}(o_{t-M+1:t},\, a_{t-M+1:t}) = \mathrm{Var}_b\!\left[\mu^{b}_{o_{t+1}}\right] \quad (4)
$$

The predicted next observation is the mean across ensemble members. The epistemic uncertainty $u$ is the variance across ensemble means.

The paper writes the uncertainty at the observation level, so $u$ in Eq. 4 is vector-valued (one variance per observation dimension). For policy optimization, this vector-valued disagreement must be reduced to a scalar penalty per transition, since rewards are scalar quantities. The exact reduction (sum across dimensions, mean, max, or some norm) is left implicit in the paper and should be checked in the implementation analysis.

The paper also computes aleatoric uncertainty separately as the mean of the predicted standard deviations across ensemble members (this is mentioned in Section 5.1 but not given a separate equation), but only epistemic uncertainty is consumed by MOPO-PPO. The aleatoric signal is reported in figures for diagnostic purposes.

### 3.4 Why this design quantifies what we want

Three properties together justify the design:

1. *Epistemic reducibility.* Variance across ensemble means goes to zero in regions where the dataset has heavy coverage (all members converge to the same prediction) and grows in regions where data is sparse (members extrapolate differently). This matches the intended interpretation of epistemic uncertainty.

2. *Aleatoric per-member.* Each member's predicted $\sigma$ captures the noise the member thinks is present at this state. Averaging across members gives a stable estimate of the noise without confusing it with model disagreement.

3. *Separation.* Variance of means (epistemic) and mean of variances (aleatoric) are mathematically distinct quantities. A model with low epistemic but high aleatoric is confident in its predictions but says the environment itself is noisy here. A model with high epistemic but low aleatoric is unsure of its own predictions, regardless of any environment noise.

### 3.5 Figure 1 in the paper

Figure 1 is the high-level pipeline diagram. It shows three stages left-to-right:

- *Offline datasets* (left): tuples $(o_t, a_t, o_{t+1})$ as the input to the pipeline.
- *Uncertainty-aware imagination* (middle): RWM-U receives a window $o_{t-\cdots}, o_{t-1}, o_t$ and rolls forward autoregressively to $o'_{t+1}, o'_{t+2}, \ldots, o'_{t+T}$. The figure shades the imagined trajectories to indicate growing uncertainty over the rollout horizon. MOPO-PPO consumes these uncertain rollouts and applies the penalty $\tilde r = r - \lambda u$.
- *Policy learned offline* (right): the trained policy $\pi$ deploys without simulation.

The shading in the middle panel is the diagrammatic representation of epistemic uncertainty growing with rollout horizon. The rest of the paper makes this picture quantitative.

### 3.6 Figure 2 in the paper

Figure 2 is the training-time diagram, showing two consecutive autoregressive rollouts of length $T - M$ each, both conditioning on the same model RWM-U. The figure makes three points visually:

- Each roll consumes a history $o_{t-M+1}, \ldots, o_t$ and produces predictions $o'_{t+1}, o'_{t+2}, \ldots$.
- The rolls are shifted by one timestep to illustrate the sliding-window training.
- For each predicted observation, three quantities are produced: the next observation $o'$, the epistemic uncertainty $u$, and the penalized reward $\tilde r$.

The bottom of the figure shows MOPO-PPO consuming $(o'_{t+k}, u_{t+k}, \tilde r_{t+k})$ for the policy update.

## 4. MOPO-PPO: policy optimization

### 4.1 The penalized reward

From Section 4.2, the central equation:

$$
\tilde r(o_t, a_t) = r(o_t, a_t) - \lambda \cdot u_{p_\phi}\!\left(o_{t-M+1:t},\, a_{t-M+1:t}\right) \quad (5)
$$

The reward $r$ is the same as in any RL algorithm; the new term is the penalty $-\lambda u$. With $\lambda > 0$, high-uncertainty transitions reduce the reward signal, steering the policy toward regions the model is confident about.

The paper notes that $\lambda$ is the *only new hyperparameter* introduced relative to standard PPO. Everything else (clip range, KL target, GAE, mini-batches) inherits from standard PPO.

### 4.2 Algorithm 1

Algorithm 1 in the paper has the following structure:

```text
1: Input: Offline dataset D
2: Initialize policy π_θ, world model p_φ
3: Train the world model p_φ autoregressively using offline dataset D according to Eq. 1
4: for learning iterations = 1, 2, ... do
5:     Initialize imagination agents with observations sampled from D
6:     Roll out uncertainty-aware trajectories using π_θ and p_φ for T steps according to Eq. 4
7:     Compute uncertainty-penalized rewards for each imagination transition according to Eq. 5
8:     Update π_θ using PPO or another RL algorithm
9: end for
```

The structural difference from MBPO-PPO (the RWM paper's algorithm):

- Step 3 is a *one-shot* offline training of the world model on the fixed dataset $\mathcal{D}$. It happens once, before any policy updates begin.
- Steps 4-9 do not include any real-environment interaction. The whole loop runs entirely in imagination, against the world model trained in step 3.
- Step 7 introduces the uncertainty penalty.

This is not the same as MBPO-PPO from the RWM paper, where the world model continues to be updated using fresh real-environment data collected by the current policy at every iteration.

### 4.3 The choice of PPO over SAC

The paper devotes Section A.5 to justifying the use of PPO (rather than SAC) for the policy update. Briefly:

- Modern legged locomotion practice favors PPO over SAC empirically. On-policy methods with massively parallel simulators are more robust on locomotion tasks than off-policy critics.
- PPO's trust-region behavior pairs well with imagination-based training because the policy never wanders too far from the data distribution it was trained on, which mitigates the risk of policies driving the dynamics model into regions it cannot model.
- SAC's actor-critic formulation is sensitive to value function errors that compound when the underlying dynamics model is uncertain. The paper notes that "off-policy value-based methods also tend to struggle with the distributional shift characteristic of locomotion control."
- Tuning SAC for large-scale legged locomotion is acknowledged as an open problem (citing Raffin, 2025).

### 4.4 The role of $\lambda$

The penalty coefficient $\lambda$ trades off exploration against model reliability:

- $\lambda \to 0$: the penalty vanishes, the policy ignores uncertainty, MOPO-PPO degenerates to MBPO-PPO. The policy can exploit hallucinated regions of the model.
- $\lambda \to \infty$: the penalty dominates the reward, the policy stays in regions of zero uncertainty, which usually means staying near the dataset's most-covered states, which can mean staying still. The policy becomes overly conservative.
- A well-tuned $\lambda$: the policy explores the model in regions where the model is confident, avoiding the high-uncertainty fringes.

The paper identifies $\lambda = 1.0$ as the well-tuned value for ANYmal D and Unitree G1 in their experiments. Section 5.2 sweeps $\lambda$ across multiple values per robot and shows the performance curve.

## 5. Empirical results

### 5.1 Autoregressive uncertainty estimation (Figure 3)

Section 5.1 trains RWM-U on offline data collected from a velocity-tracking policy on ANYmal D, with $M = 32$, $N = 8$, ensemble size $B = 5$. The model runs at 50 Hz control frequency.

Figure 3 visualizes long-horizon rollouts of approximately 200 steps. The model conditions on ground-truth observations until $t = 32$ (the history horizon), then rolls out autoregressively. Three quantities are plotted in the right panel:

- *Prediction error* (grey): the L1 difference between the predicted observation and the ground truth at each step. Grows over the autoregressive rollout, as expected.
- *Epistemic uncertainty* (dark blue): the variance across ensemble means. Tracks the prediction error closely. This is the central empirical claim of the section.
- *Aleatoric uncertainty* (light blue): the mean of predicted standard deviations across members. Stays low and flat, reflecting that the simulation environment has small inherent stochasticity.

The strong correlation between epistemic uncertainty and prediction error is the property RWM-U is designed to produce: the model knows where it is uncertain, in a way that aligns with where it is actually wrong. This is the trust signal MOPO-PPO consumes.

The left panel of Figure 3 shows the actual predicted state evolution against ground truth for linear velocity, angular velocity, and joint position, with solid lines (ground truth) and dashed lines (predicted) closely aligned in the early portion of the rollout and diverging at later times (consistent with the growing uncertainty in the right panel).

### 5.2 Penalty coefficient sweep (Figure 4)

Section 5.2 reports policy optimization with $\lambda$ varied across multiple values. Figure 4 shows imagination reward (left) and epistemic uncertainty (right) over training iterations, for two robots (ANYmal D and Unitree G1). The final policy evaluation on real dynamics is overlaid as dots on the imagination-reward curves.

The $\lambda$ values used in the figure differ between robots:

- ANYmal D: $\lambda \in \{0.0, 0.2, 0.5, 1.0, 2.0, 5.0\}$.
- Unitree G1: $\lambda \in \{0.0, 0.1, 0.5, 1.0, 5.0, 10.0\}$.

The qualitative findings:

- *Small $\lambda$ ($\leq 0.5$):* high imagined reward, but the policy exploits model errors and fails on real dynamics. The robot shows unstable gaits or fails to walk.
- *Large $\lambda$ ($\geq 2.0$ on ANYmal D, $\geq 5.0$ on Unitree G1):* low imagined reward, policy stays in low-uncertainty regions which often means staying near a stationary configuration. The robot exhibits overly cautious or stationary behavior.
- *Well-tuned $\lambda = 1.0$:* imagination reward grows steadily, epistemic uncertainty stays bounded, and the deployed policy works well on real dynamics.

This is the classic exploration-exploitation curve, but with the controlling parameter being the *uncertainty penalty* rather than a learning rate or entropy bonus.

### 5.3 Generality across environments (Figure 5)

Section 5.3 compares MOPO-PPO with RWM-U against four baselines on three robotic tasks (Reach-Franka manipulation, Velocity-ANYmal-D quadruped, Velocity-Unitree-G1 humanoid) with four dataset types:

- *Random*: trajectories from a randomly initialized policy.
- *Medium*: trajectories from a partially trained PPO policy.
- *Expert*: trajectories from a fully trained PPO policy.
- *Mixed*: combination of replay buffer data from PPO training at various stages.

Baselines:

- *CQL*: Conservative Q-Learning, model-free with value-conservatism (Kumar et al., 2020).
- *MBPO*: Model-Based Policy Optimization with SAC (Janner et al., 2019), trained offline.
- *MBPO-PPO*: the RWM paper's method, the uncertainty-unaware ablation (RWM with PPO).
- *MOPO*: original MOPO with SAC (Yu et al., 2020).

Per Table S9 in the paper's appendix, the factorial structure of the comparison:

| Method | Model architecture | Policy optimization |
|---|---|---|
| MBPO | RWM | SAC |
| MBPO-PPO | RWM | PPO |
| MOPO | RWM-U | SAC |
| MOPO-PPO | RWM-U | PPO |

This is a clean factorial: ensemble (RWM vs RWM-U) crossed with policy optimizer (SAC vs PPO). MOPO-PPO is the RWM-U + PPO combination.

Figure 5 reports normalized episodic rewards. Key findings:

- MOPO-PPO consistently outperforms baselines across all environments and dataset types.
- The advantage is most pronounced on locomotion (ANYmal D and Unitree G1) versus manipulation (Reach-Franka), where the dynamic complexity makes uncertainty handling more critical.
- The Mixed dataset gives the best performance, comparable to or exceeding Expert. The paper attributes this to broader state-action coverage, which reduces distribution shift in the model.
- MBPO and MBPO-PPO (the uncertainty-unaware ablations) overfit to model errors and produce unsafe locomotion.
- CQL performs well on manipulation but struggles on locomotion, consistent with the broader observation that off-policy critics scale poorly to high-dimensional locomotion (see Section A.5).

### 5.4 Real-world data and hardware deployment (Table 1)

Section 5.4 trains policies entirely from offline datasets that mix simulated and real-world data, then deploys on ANYmal D hardware.

The data collection setup: a velocity-tracking policy is deployed on the real robot, and trajectories are collected with an autonomous safety-bounded sampling pipeline. The pipeline defines a rectangular safety boundary relative to the odometry origin (computed by fusing legged odometry with LiDAR data); commands that would drive the robot outside this region are rejected and resampled, with a fallback that drives the robot back toward the origin after several rejections. Approximately one hour of robot operation produces approximately 200K state transitions of real data.

Table 1 reports normalized policy performance on ANYmal D under different data mixtures. The data-collecting policy used for offline collection scores 0.79 (the baseline). The online model-free PPO baseline scores 0.88.

The MOPO-PPO results:

| Sim transitions | Real transitions | Performance |
|---:|---:|:---|
| 1M | 0 | 0.82 ± 0.02 |
| 800K | 200K | **0.91 ± 0.03** |
| 600K | 400K | 0.78 ± 0.02 |
| 400K | 600K | 0.59 ± 0.04 |
| (data-collecting policy) | | 0.79 ± 0.02 |
| (online model-free) | | 0.88 ± 0.01 |

Three observations:

1. *Pure simulation (1M sim, 0 real)* already exceeds the data-collecting policy (0.82 vs 0.79). Offline MBRL extracts a stronger policy than the one that collected the data.
2. *Best performance is at 800K sim + 200K real (0.91 ± 0.03)*. Adding real data improves over pure-simulation training. This result also exceeds the online model-free PPO baseline (0.88), which is a strong claim.
3. *Performance degrades when real data dominates the mixture* (600K and 400K real transitions). The paper attributes this to limited diversity in real data: the safety-bounded collection pipeline does not sample failure modes (falls, collisions), and these informative-but-risky transitions are exactly what world model and policy learning need.

The paper's Figure 7 illustrates the same penalty-coefficient effect on hardware: insufficient $\lambda$ produces unstable gaits and collisions on the real robot; excessive $\lambda$ produces stationary or ineffective motion; $\lambda = 1.0$ produces stable locomotion comparable to the sim-to-real baseline.

## 6. Reward formulation

The RWM-U paper inherits the reward formulation from the RWM paper. The reward terms ($r_{v_{xy}}, r_{\omega_z}, r_{v_z}, r_{\omega_{xy}}, r_{q\tau}, r_{\ddot q}, r_{\dot a}, r_{f_a}, r_c, r_g, r_{f_c}, r_{q_d}$) are equation-by-equation identical to the RWM paper's Section A.1.2, since both papers operate on the same locomotion benchmarks.

The MOPO-PPO modification is a single change at the reward-summation level: the per-step reward is summed as in RWM, then the uncertainty penalty $-\lambda u_{p_\phi}$ is subtracted, producing $\tilde r$. The penalty is per-step, not per-term: it does not modify any individual reward equation, only the total scalar reward at each imagined transition.

For the term-by-term equations and weight values, see the [RWM paper analysis Section 6](../world-model/paper-analysis.md#6-reward-formulation).

## 7. Architecture and hyperparameters

### 7.1 RWM-U architecture (Table S7)

Identical to RWM, with the addition of $B$-fold ensemble heads:

| Component | Type | Hidden Shape | Activation |
|---|---|---|---|
| base | GRU | 256, 256 | − |
| heads | MLP | 128 | ReLU |

### 7.2 Policy and value architecture (Table S8)

Identical to RWM:

| Network | Type | Hidden Shape | Activation |
|---|---|---|---|
| policy | MLP | 128, 128, 128 | ELU |
| value function | MLP | 128, 128, 128 | ELU |

### 7.3 RWM-U training parameters (Table S10)

| Parameter | Symbol | Value |
|---|---|---:|
| step time seconds | $\Delta t$ | 0.02 |
| max iterations | − | 2500 |
| learning rate | − | $1\times10^{-4}$ |
| weight decay | − | $1\times10^{-5}$ |
| batch size | − | 1024 |
| ensemble size | $B$ | 5 |
| history horizon | $M$ | 32 |
| forecast horizon | $N$ | 8 |
| forecast decay | $\alpha$ | 1.0 |
| transitions in training data | − | 6M |
| approximate training hours | − | 1 |
| number of seeds | − | 5 |

### 7.4 MOPO-PPO training parameters (Table S11)

| Parameter | Symbol | Value |
|---|---|---:|
| imagination environments | − | 4096 |
| imagination steps per iteration | − | 100 |
| uncertainty penalty weight | $\lambda$ | 1.0 |
| step time seconds | $\Delta t$ | 0.02 |
| buffer size | $\lvert\mathcal{D}\rvert$ | 1000 |
| max iterations | − | 2500 |
| learning rate | − | 0.001 |
| weight decay | − | 0.0 |
| learning epochs | − | 5 |
| mini-batches | − | 4 |
| KL divergence target | − | 0.01 |
| discount factor | $\gamma$ | 0.99 |
| clip range | $\epsilon$ | 0.2 |
| entropy coefficient | − | 0.005 |
| number of seeds | − | 5 |

The paper reports total training time on an NVIDIA RTX 4090: approximately 1 hour for the world model (RWM-U). MOPO-PPO training time is not explicitly stated.

The "imagination steps per iteration = 100" entry in Table S11 is paired with the paper's repeated mention of "100-step episodic rollouts" in Section 1 and Section A.5. The paper's claim is unambiguous: the autoregressive rollout horizon during MOPO-PPO training is 100 steps. (This contrasts with the active code path documented on the implementation analysis page, where the runner uses 24-step rollouts; this is one of the paper-to-code discrepancies tracked in the synthesis page.)

## 8. Limitations stated by the paper

Section 6 of the paper acknowledges the following limitations.

### 8.1 Dataset bound on policy quality

Like all offline RL, the policy is bounded by what the dataset covers. State-action regions absent from the dataset cannot be fully recovered by uncertainty-aware modeling. Compounding model errors over very long horizons remain a challenge that MOPO-PPO mitigates but does not eliminate.

### 8.2 Real-world data lacks risky transitions

Real-world data collected under safety constraints lacks the failure modes (falls, collisions, edge cases) that are essential for the model to learn what *not* to do. The paper frames this as a trade-off: real data has the right distribution but is limited in coverage; simulated data has full coverage but is biased by simulator-reality gap. The authors recommend mixing.

This limitation is empirically visible in Table 1: increasing the real-data share past 200K transitions (to 400K and 600K) degrades performance. The paper attributes this to the missing failure-mode coverage in the safety-bounded real-data pipeline.

### 8.3 PPO's on-policy nature

Although not stated as a limitation explicitly, the choice of PPO inherits PPO's on-policy data requirements. In the imagination loop this is satisfied because each imagined rollout is on-policy with respect to the current $\pi_\theta$, so the constraint is internal rather than external. But this means MOPO-PPO cannot easily leverage off-policy data sources beyond what fits inside the world model's training set $\mathcal{D}$.

## 9. Connection to RWM and relationship to other methods

### 9.1 RWM as a special case of RWM-U

Conceptually, RWM can be viewed as the special case of RWM-U with a single ensemble member ($B = 1$) and no uncertainty penalty ($\lambda = 0$). With $B = 1$, the ensemble disagreement is identically zero, so the epistemic uncertainty signal vanishes; with $\lambda = 0$, the penalized reward $\tilde r$ collapses to the original reward $r$. Algorithmically, the uncertainty-aware part collapses to uncertainty-unaware imagination with RWM-style rollouts. It does not by itself recover the full online RWM training loop, because the offline fixed-dataset assumption remains different from the online data-collection loop in the original RWM paper.

How this special-case relationship is realized in code (whether the same code path implements both methods, or whether the codebase has separate scripts for online RWM and offline RWM-U) is documented on the [Implementation Analysis](implementation-analysis.md) and [Paper-to-Code Synthesis](paper-to-code-synthesis.md) pages.

### 9.2 MOPO-PPO versus MOPO

Original MOPO (Yu et al., 2020) uses SAC. MOPO-PPO replaces SAC with PPO. The uncertainty-penalty mechanism is identical (Eq. 5 is structurally the same as MOPO's penalized reward).

The paper argues PPO is the right policy optimizer for legged locomotion specifically, on practical grounds (Section A.5): on-policy methods with massively parallel simulation are empirically more robust on locomotion tasks, and PPO's trust-region update provides additional stability when training against imagined data.

### 9.3 RWM-U versus PETS / MBPO

Probabilistic Ensembles with Trajectory Sampling (PETS, Chua et al., 2018) and MBPO (Janner et al., 2019) both use ensembles of dynamics models. The differences from RWM-U:

- PETS and MBPO use *fully independent* ensemble members (each member is a separate network).
- PETS uses MPC-style trajectory sampling for action selection, not policy optimization.
- MBPO interleaves real-environment interaction with imagined-rollout training (online setting).

RWM-U's shared-base design is a computational simplification that preserves the variance-of-means epistemic uncertainty signal while reducing parameter count. The paper does not directly compare its shared-base ensemble against a fully independent ensemble.

## 10. Summary of paper claims

Five claims targeted by the experimental section, in increasing order of significance:

1. RWM-U's epistemic uncertainty signal correlates with actual prediction error in long autoregressive rollouts. (Section 5.1, Figure 3.)
2. The penalty coefficient $\lambda$ has a clear optimum around $\lambda = 1.0$ on ANYmal D and Unitree G1. (Section 5.2, Figure 4.)
3. MOPO-PPO with RWM-U outperforms MBPO, MBPO-PPO, MOPO, and CQL on three benchmark tasks across four dataset types. (Section 5.3, Figure 5.)
4. Policies trained entirely on offline data deploy on ANYmal D and Unitree G1 hardware, with qualitative deployment results visualized in Figures 6 and 7 across both robots. The quantitative data-mixture study in Table 1 is reported for ANYmal D only; whether equivalent quantitative numbers exist for Unitree G1 should be rechecked against the PDF before being used as a final reproduction target. (Section 5.4.)
5. The best mixed-data configuration (800K sim + 200K real) achieves $0.91 \pm 0.03$ on ANYmal D, exceeding both the data-collecting policy ($0.79$) and the online model-free PPO baseline ($0.88$). (Table 1.)

These claims are the targets against which any project-side reproduction will be measured. Per-claim verification status will be tracked on the [Reproduction Status](../validation/reproduction-status.md) page once the corresponding implementation analysis is complete.