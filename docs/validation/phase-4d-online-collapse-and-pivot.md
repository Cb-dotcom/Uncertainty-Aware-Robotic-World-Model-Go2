# Phase 4D: The Online RWM-U Collapse, and the Pivot to Offline

*Engineering / validation log. Continues Phase 4C. Work spans 2026-06-13 → 2026-06-16 on the lab
workstation (`rwmu-cogar-cb`), RTX 6000 Ada, shared with a root-owned vLLM (~41.5 GB).*

---

## 0. Scope

Phase 4C ended with the RWM-U comparison *launched but not interpreted*: an ensemble-5 pretrain done
and two finetunes (`pen0`, no penalty; `pen1`, penalty −1.0) queued from the same checkpoint. This
phase reports what those runs actually showed, the additional horizon experiment they motivated, and
the conclusion that ended the online line of attack entirely. It closes with the realization, from a
close re-reading of both papers, that the project's *named* method (uncertainty-aware RWM) is the
**offline** RWM-U / MOPO-PPO pipeline, not the online MBPO pipeline Phases 4B-4C had been exercising.

The headline: **the online finetune collapse is real, mechanistically understood, and largely
penalty-independent.** It is the motivation for the offline contribution, not the contribution itself.

---

## 1. The pen0 / pen1 comparison: both arms collapse

Both finetunes ran to completion from the shared `2026-06-12_13-39-03_pretrain_ens5/model_2000.pt`.
Read at *matched iterations* (last-50 windows, not snapshots), the primary metric, `base_contact`
termination rate, tells the story:

| iter | h256 `pen0` | h256 `pen1` |
|---|---|---|
| 2100 (≈ shared start) | 0.028 | 0.028 |
| 2500 | 0.523 | 0.557 |
| 3000 | 0.421 | 0.527 |
| 3998 | 0.587 | 0.511 |

Both arms diverge from the *identical* good state at iter ~2100 (reward 20.7, `base_contact` 0,
tracking 0.91) and both collapse to a ~50-60% fall rate. The penalty changed *which fiction the policy
exploits*, not *whether* it collapses:

- `pen0` (no penalty) collapses into **slide-and-fall**: `feet_slide` drifts to ≈ −0.29 (vs −0.06
 baseline), i.e. the policy re-finds the old skate optimum, and falls anyway.
- `pen1` (−1.0) collapses into **freeze-and-fall**: action noise-std collapses (0.087 vs 0.26) and
 entropy goes sharply negative, the penalty crushed exploration without reducing falls.

A telling detail: `pen1`'s epistemic uncertainty *rose* over training (0.80 → 1.25) while being
penalized for it. The penalty was supposed to steer the policy toward low-disagreement regions; instead
the policy moved into higher-uncertainty regions and bled reward. **A −1.0 penalty is mis-scaled
relative to the unnormalized imagined task reward (~5.7): the penalty contribution (~1.2) is ~20% of the
objective, enough to hijack it.**

> **Correction recorded for honesty.** An early read of this table called `pen0` "healthier" (higher
> reward, better tracking, no entropy collapse). That read was half-wrong on the metric that matters:
> on `base_contact`, `pen0` is *worse*. Its better tracking is partly the slide-cheat returning; its
> "positive imagined reward" is definitional (no penalty term), not a health signal.

---

## 2. The horizon experiment: and why it failed to rescue the control

The shared collapse from an identical state pointed at **long-horizon model exploitation**: the
imagination rolls out `max_episode_length = 256` steps against a world model whose autoregressive error
is 0.6-1.0, i.e. fiction well before step 256. The hypothesis: shorten the horizon and the no-penalty
control should stabilize, giving an interpretable baseline.

`finetune_ens5_pen0_h32` (`max_episode_length = 256 → 32`, all else identical) was run. Confirmation the
lever engaged: `mean_episode_length_imagination` dropped from ~256 to ~31.4.

**Result, at matched iterations:**

| iter | h256 `pen0` | **h32 `pen0`** | h256 `pen1` |
|---|---|---|---|
| 2500 | 0.523 | **0.697** | 0.557 |
| 3000 | 0.421 | **0.647** | 0.527 |
| 3998 | 0.587 | **0.688** | 0.511 |

**h32 is the *worst* arm at every matched point.** Shortening the horizon made falling worse, not
better. Its one real effect was less sliding (`feet_slide` −0.18 vs −0.29), the two failure modes
trade off, but neither is fixed. An apparent mid-run "h32 is better" reading was an artifact of comparing
h32's *transient* against h256's *endpoint*; matched, the horizon story dies.

**Conclusion: horizon is necessary context, not the lever.** The decisive cross-check came later (Phase
4E): the *working* ANYmal-D finetune used the **same** `max_episode_length = 256`. So 256 is the
established working value, not a Go2-specific bug, and the Go2 collapse is not caused by the horizon.

---

## 3. The termination-head class-imbalance finding

A consistent signature across all three online arms explained the falling collapse mechanistically:

- `Train/mean_episode_length_imagination` pins at the 256 cap while real `base_contact` climbs to
 0.5-0.7, **imagined episodes stop terminating exactly as real falling rises.**
- `System Dynamics/termination_loss ≈ 0.000` throughout.

Base-contact falls are *rare per-step events* (one positive per ~200-step episode, <1%). An unweighted
BCE termination head minimizes its loss by predicting "never terminate", scoring ~0 loss while being
useless. So the policy trains in an imagination where it essentially **never falls**, gets no gradient
pressure to avoid falling, and then falls half the time in reality. This class-imbalance blindness
becomes a central design target in the offline work (Phase 4F, `pos_weight`).

---

## 4. The code-level bridge: why online optimization *must* exploit

Reading the MBPO runner (`mbpo_on_policy_runner.py`) settled why the collapse is structural rather than a
tuning accident. After the world-model warmup:

- every iteration collects one real env step and trains the world model;
- the policy update is **gated**: during finetune it always takes `self.alg.update(imagination=True)`;
- the real-data policy update path runs **only** when imagination is off (i.e. in pretrain).

So **after warmup the policy's entire reward gradient comes from imagined rollouts**; real data only
trains the world model and seeds imagination, and never enters the policy objective. A policy optimized
purely against an imperfect model, for thousands of iterations, *will* find and exploit the model's
errors, that is a guarantee, not a surprise. The paper's MBPO-PPO budget is ~5 minutes of
policy-optimization; our 2000-iteration runs continue long past the point where imagination still yields
real gains. **We reproduced the method at its budget (best-checkpoint tracking ~0.92, matching the
paper's 0.90) and then kept going, exposing the exploitation regime the paper documents but does not
characterize.**

---

## 5. The pivot: the named method is *offline*

Both papers were re-read carefully at this point:

- **Paper 1 (RWM, MBPO-PPO)** is the *online* pipeline: world model + policy trained with environment
 interaction. Its own limitations section reports model-exploitation collisions ("more than 20 times on
 average during online learning") and explicitly proposes uncertainty-aware world models as *future
 work requiring architectural modification*. The deployed policy is a selected checkpoint; the
 exploitation is described as transient.
- **Paper 2 (RWM-U, MOPO-PPO)**: *"Uncertainty-Aware Robotic World Model Makes Offline Model-Based RL
 Work on Real Robots"*, is the **offline** pipeline: the policy is optimized entirely on a fixed
 dataset with no environment interaction, against an ensemble world model whose epistemic disagreement
 penalizes the policy (MOPO). It is the first demonstration of uncertainty-penalized offline MBRL on
 full-scale physical robots.

The thesis title is *Uncertainty-Aware Robotic World Model*, i.e. **Paper 2's method**. Phases 4B-4D
exercised Paper 1's *online* pipeline and rediscovered Paper 1's known exploitation problem. That is
valuable as **motivation**: an independent, mechanistic characterization of exactly the failure RWM-U
exists to prevent, but it is not the contribution.

**Decision: pivot to the offline RWM-U / MOPO-PPO pipeline (`scripts/.../model_based/`), with Go2 as the
target robot.** The online collapse study is reframed as the motivating negative result. The standing
caveat, recorded so it is never overstated: in this codebase "offline" means *the policy is optimized
against a frozen world model with no env interaction*, the world model itself is still produced by an
upstream training stage. This is a defensible, standard offline-MBRL setup, but it is not offline-data
world-model fitting in the strictest sense until Phase 4F builds exactly that.

---

## 6. Status at end of Phase 4D

- Online RWM-U on Go2: **characterized, closed.** Both penalty arms collapse from a shared good state;
 the failure is largely penalty-independent (no-penalty control degrades comparably), `−1.0` is
 mis-scaled, the termination head is class-imbalance-blind, and horizon is not the lever.
- This is a clean **negative-with-mechanism** result and the motivation for the offline contribution.
- Next (Phase 4E): build the offline Go2 pipeline, build the real-environment evaluation harness, and
 validate the whole apparatus on ANYmal-D against the authors' own assets before trusting any Go2
 number.
