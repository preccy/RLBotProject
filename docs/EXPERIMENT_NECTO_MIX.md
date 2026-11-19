# Necto-mix experiment (pre-registered; one bounded comparison)

Written and committed before the smoke check and before either arm trains. Results are appended under "Results";
nothing above it is edited once training starts.

## Question
Does giving about 15% of training episodes to the frozen external bot Necto V1 (docs/BENCH_NECTO.md) improve
IL-12900? Two separate questions, reported separately:
1. **Against Necto:** does exposure improve the goal margin against Necto itself?
2. **Transfer:** does any gain carry over to other opponents (the control arm, frozen IL-12900, the standing
   regression references)?
One seed per arm and 1,000 iterations: a first, uncertain read. An inconclusive comparison is reported as
inconclusive, not as "no harm".

## Arms
Both resume from the same full checkpoint `runs/reference/S4I-iter12900-checkpoint` (read-only), with the same run
seed. Rewards, imitation term and demonstrations (coef 0.02, `runs/demos/s1.npz` matches 0-1), action selection
(sampled), scenario mixture, architecture, action table and PPO settings are identical.
* CTRL `runs/s5-ctrl`, `configs/experiments/necto-control.toml`: the configuration that produced IL-12900,
  continued. 12 workers x 6 envs = 72 self-play envs = 144 learner slots.
* NECTO `runs/s5-necto`, `configs/experiments/necto-exp.toml`: identical plus `[self_play] necto_fraction = 0.15`.
  That gives 13 workers x 6 envs = 78 envs. 12 envs (15.4%) put the learner against frozen Necto V1, with Necto's
  side alternating orange/blue across envs; the other 66 are normal mirror self-play. Learner slots are
  66 x 2 + 12 x 1 = 144, the same as CTRL.
  * Resumed with `--allow-config-change`; the only learning-relevant diff is `necto_fraction`.
  * Necto runs inside the workers from the game state (`src/botcore/necto.py`, the adapter checked in
    docs/BENCH_NECTO.md). Its car is an EXTERNAL slot and never enters PPO.
  * Necto envs use the same scenario mixture (kickoff 50%, open play, shot, save, ...); Necto plays argmax, as
    shipped.
  * The env share is fixed; the actual episode share depends on episode lengths and is logged each iteration.
* **Budget:** 1,000 iterations per arm (target 13,900). That is 73,728 learner transitions per iteration in both
  arms, so 73,728,000 learner transitions per arm, the same as each IL-pilot arm. Only learner transitions count.
  The PPO schedule is also identical: 3 batches of 24,576 x 2 epochs = 6 updates per iteration.
* The env seeds differ between arms by construction (78 vs 72 envs).
* The arms run one after the other, with Rocket League closed.

## Smoke check (10 iterations in `runs/necto-smoke` from the same checkpoint; discarded, neither arm starts from it)
Pass requires all of:
* no worker error;
* Necto envs finish episodes, and Necto scores most goals in them (the benchmark had 12900 conceding 81%);
* exactly 73,728 learner steps per iteration (the trainer checks learner slots = 144 at start);
* finite losses and imitation term;
* approx_kl < 0.02;
* throughput practical: an arm can finish in under about 1.5 h (>= ~14,000 learner steps/s).

The exclusion of Necto's car from PPO is also covered by `tests/test_necto_training.py`. On failure: stop and
report; no setting is tuned.

## Evaluation (our bots sample their actions throughout; fresh seed offsets)
1. **Against Necto** (`scripts/necto_mix_eval.py`, same match rules and adapter as the benchmark): CTRL-final and
   NECTO-final each play the same 40 colour-swapped seed pairs (offsets 18,100-18,139). IL-12900 on the same seeds
   is a descriptive reference only.
   * Primary: paired difference in goal margin per match, NECTO minus CTRL, with a bootstrap 95% CI over pairs.
   * Also reported: the difference in goals scored and in goals conceded, and W-D-L.
2. **Transfer:**
   * NECTO vs CTRL head-to-head: 100 pairs (offsets 19,000-19,099).
   * Each final vs frozen IL-12900: 100 pairs (offsets 19,200-19,299, same seeds for both arms).
   * Regression checks vs R0 and M2REF: 40 pairs (seeds 0-39).

## Decision rules (fixed now)
* **Against Necto:** exposure **helped** if the CI lower bound of the paired goal-margin difference is > 0;
  **hurt** if the upper bound is < 0; otherwise **inconclusive**.
* **Transfer:** NECTO is **better** than CTRL if the head-to-head CI lower bound is > 0.5, **worse** if the upper
  bound is < 0.5, otherwise **inconclusive**. A gain against Necto is said to carry over only if NECTO is better
  head-to-head. The comparisons with frozen 12900 are reported for each arm, with no further threshold.
* **Promotion (standing rule):** a final may replace IL-12900 as best_verified only with:
  * 95% CI lower bound > 0.5 against IL-12900 on 100 fresh pairs;
  * >= 0.45 in both regression checks.

  If both finals qualify, NECTO is chosen only if it is better head-to-head; otherwise CTRL. A promoted checkpoint
  is installed in `bot/model` only after the fixed 4-match All-Star series shows no runner faults. Otherwise
  IL-12900 stays best_verified and deployed. IL-12900 is preserved either way (read-only reference copies).
* No extension, no second fraction and no further runs.

---

## Results
