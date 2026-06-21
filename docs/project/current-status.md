# Current Status

*A one-page snapshot of where the project actually is. For the per-claim ledger see
[Reproduction Status](../validation/reproduction-status.md); for the phase-by-phase narrative see the
Validation section (Phases 4A–4F).*

## One-line state

A command-following **offline** RWM-U / MOPO-PPO Go2 walker has been achieved in simulation — weak,
conservative, slip-heavy, and seed-fragile (a minority basin), at parity with the offline ANYmal
baseline on tracking reward. The apparatus is validated; the frontier is robustness and world-model
coverage.

## The arc, compressed

1. **Online RWM/MBPO on Go2 (Phases 4B–4C).** Fixed the scoot (`feet_slide`), got a clean walking
   pretrain, then plain MBPO finetune **collapsed** via model exploitation.
2. **Online RWM-U collapse + pivot (Phase 4D).** Both penalty arms collapse from a shared good state;
   failure is largely penalty-independent; horizon is not the lever; the termination head is
   class-imbalance-blind. Re-reading the papers established that the thesis method is the **offline**
   RWM-U pipeline (Paper 2), not the online one. The online collapse is reframed as **motivation**.
3. **Offline pipeline + ANYmal gate (Phase 4E).** Built the offline Go2 pipeline and a real-environment
   eval harness; validated the whole apparatus on ANYmal using the authors' assets (penalty −1.0 →
   0.417; penalty 0 collapses). Found Bug #1 (policy run against an untrained WM). A Go2 policy against
   an online-pretrain WM does not walk — a coverage frontier (exploit ↔ freeze).
4. **Curated WM + offline walker (Phase 4F).** Built a coverage-broadened curated dataset (noise on a
   competent policy → goal-directed falls), fit a Go2 world model (held-out termination AUC 0.978, but
   bimodal/half-covered), found and fixed Bug #2 (normalizer-consistency freeze), and produced the
   offline walker.

## What is solid

- The two-sided **exploit ↔ freeze conservatism frontier** and the **normalizer-consistency** requirement
  that gates any of it being measurable — the structural contribution.
- The **ANYmal gate** validating method + pipeline + harness.
- Two diagnosed root-cause **bugs** and a **confound ledger** ([Engineering Findings](../validation/engineering-findings.md)).
- A deployable online `ens5` walker (the hardware-demo artifact, separate from the offline result).

## What is open

- Seed **walk-rate** over a larger set; a rendered video to confirm trot-vs-shuffle gait quality.
- The **ANYmal-WM diagnostic** — build an ANYmal WM with *our* fit and compare to 0.417 — to localize
  whether the remaining gap is the WM-fitting pipeline or dataset coverage.
- If coverage: the **diverse-data rebuild** (random/medium/expert behavior policies, Paper 2's recipe).
- Hardware deployment (Phase 4G), gated on robustness + safety, using the online `ens5` walker.

## Honest framing for write-up / defense

Report the offline walker as *demonstrated but seed-fragile, at offline-ANYmal parity* — not as
paper-parity locomotion. Lead the contribution with the **frontier + the normalizer lesson + the two
bugs**, which are solid regardless of the walker's strength, and present the walker as the
existence-proof at the narrow operating point between exploitation and freezing.
