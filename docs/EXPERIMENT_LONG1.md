# Long run 1: gated two-hour continuation (pre-registered)

Written and committed before training starts. Results are appended below "Results" afterwards; nothing above it is
edited once the run begins.

## Starting point and why

* Start: M2 run **D-s202, iteration 160** (11,796,480 learner steps, 960 model updates), with its full training
  state: actor/critic weights, both Adam states, return normaliser, counters, RNG.
  Resumed from a byte-identical read-only copy, `runs/reference/M2-D-s202-iter160-checkpoint` (the original
  `runs/m2-D-s202` is left untouched).
* Why this candidate:
  * Cell D is the configuration that stays the provisional default under the M2 pre-registered rule, so continuing
    it keeps the reward/learning configuration unchanged.
  * D-s202 is the strongest M2 policy on evidence that held up: it beat C-s202 0.800 [0.713, 0.875] and B-s202
    0.838 [0.750, 0.912] head-to-head, and E60 0.738 [0.650, 0.825], with the most developed behaviour
    (airborne 27%, 4.5 touches/min).
  * C-s202's higher 40-pair score vs random did not replicate on 100 fresh pairs.
* Reference policy **M2REF** = D-s202 iteration 160 exported (`runs/reference/M2-D-s202-iter160`, weights
  `6a46adbeb590`). Random baseline **R0** = `runs/reference/R0-random-init` (`b3e66cfbd006`).

## Training

* Config `configs/experiments/long1.toml`, generated from the checkpoint's own saved config. Only `run.name`,
  `run.total_iterations` and the checkpoint cadence differ; `resume` without `--allow-config-change` refuses any
  learning-relevant difference, so these are unchanged: obs contract obs-v1, action table rlgym-lookup-90-v1,
  512x3 MLP, rewards (goals 3/3, touch 0.5, velocity_to_ball 0.5, ball_to_goal 1.5, boost 0.1), PPO settings,
  scenario mix, seed 202.
* Budget: **2,103 additional iterations x 73,728 = 155,049,984 additional learner steps.** `total_iterations` is an
  absolute target, hence 2,263 = 160 + 2,103. Expected ~2 h at the measured ~77M steps/h.
* Checkpoints every 50 iterations (~3.7M steps), all kept. The run stops at the budget; no further run is started.

## Evaluations (paired seeds, colours swapped within each pair, 300 s + up to 120 s overtime, stochastic both sides)

* **Midpoint (diagnostic only)**: checkpoint iteration 1,200 (~76.7M added steps) vs R0 and vs M2REF, 40 pairs
  each on the M2 diagnostic seed set (offsets 0-39). Never used for the decision.
* **Final, diagnostic**: final checkpoint (iteration 2,263) vs R0 and vs M2REF on the same 40-pair diagnostic set,
  comparable with the M2 table.
* **Final, decision (fresh seeds, never used before)**: offsets 1,000-1,149, **150 pairs (300 matches)** each:
  1. FINAL vs M2REF;
  2. FINAL vs R0;
  3. M2REF vs R0 (same seeds), needed for the own-goal comparison.
* Reported for each: W-D-L, score with 95% percentile bootstrap CI over seed pairs, goals for/against, own goals per
  match, touches per minute, airborne fraction.

## Decision rules (fixed now)

* **Improvement vs M2** (primary): FINAL vs M2REF score has 95% CI lower bound > 0.5 **and** point estimate >= 0.60.
* **Beats random**: FINAL vs R0 score has 95% CI lower bound > 0.5.
* **Own-goal reduction**: on the fresh seeds vs R0, per seed pair compute (FINAL own goals per match - M2REF own
  goals per match), averaging the two colour-swapped matches of each pair. Reduction is established if the 95%
  bootstrap CI of the mean difference lies entirely below 0. Own-goal share (own goals conceded / all goals
  conceded) is reported as well.
* **Verdict**:
  * **demonstrated improvement** if "improvement vs M2" holds;
  * **regression** if FINAL vs M2REF has CI upper bound < 0.5;
  * otherwise **inconclusive**.
* **Continuation gate** (whether a further run of this configuration is justified): continue if **beats random**
  holds, or if **improvement vs M2** and **own-goal reduction** both hold. Otherwise stop changing nothing but
  steps and investigate (own-goal sequences and their reward contributions) before proposing a change.
* **Promotion**: FINAL becomes the first `best_verified` only if **beats random** holds (150 fresh pairs). If only
  "improvement vs M2" holds it is marked `candidate`, not promoted. Reward curves and point estimates with
  inconclusive intervals never promote anything.
* Live Rocket League validation only if the verdict is "demonstrated improvement".

---

## Results (appended 2026-11-16 after the run and all evaluations)

**Training.** Resumed exactly (counters continued from 11,796,480; entropy continued from D-s202's last value);
2,103 iterations, 155,049,984 added steps, final total 166,846,464; 2.16 h at 19.9k steps/s; stopped at the
budget. Runtime log-prob alignment guard max 3.1e-5. Training diagnostics (not used for decisions): touch reward
per step 0.0048 -> 0.0097, entropy 3.44 -> 3.09.

**Decision evaluations (fresh seeds 1,001,000-1,001,149; 150 pairs = 300 matches each):**

| Comparison | W-D-L | Score [95% CI] | Goals | Own goals / match (A) | Touches / min (A) | Airborne (A) |
|---|---|---|---|---|---|---|
| FINAL vs M2REF | 300-0-0 | 1.000 [1.000, 1.000] | 3,401-101 | 0.18 | 64.1 | 0.10 |
| FINAL vs R0 | 298-0-2 | 0.993 [0.983, 1.000] | 2,435-182 | 0.60 | 45.8 | 0.04 |
| M2REF vs R0 | 165-56-79 | 0.643 [0.597, 0.690] | 396-223 | 0.73 | 4.7 | 0.27 |

Own-goal difference vs R0, FINAL - M2REF, per match: **-0.133 [-0.26, -0.013]** (150 pairs).

* improvement vs M2: **yes**; beats random: **yes**; own-goal reduction: **yes, but small** (0.73 -> 0.60 per match;
  own goals are still ~99% of what the random opponent is credited with).
* **Verdict: demonstrated improvement.** Continuation gate: continue. **Promoted**: `runs/s1-long` iteration 2,263
  (weights `8d55b78c86c7`) is the first `best_verified`; it passed the standing promotion rule.

**Diagnostics (40 pairs on the old diagnostic seeds; not decision inputs):** midpoint (iter 1,200) beat R0 80-0-0
(740-19) and M2REF 80-0-0 (993-59); final beat R0 80-0-0 (658-36) and M2REF 80-0-0 (892-20); **final vs midpoint
0.850 [0.775, 0.912], 68-0-12, 418-189**, so the second hour still added strength (no plateau at the budget).

**Sanity checks on the size of the effect:** goals are not a kickoff trick (vs M2REF: median 21 s from kickoff to
goal, 11% within 5 s); final vs itself is balanced (46-39 goals), so the evaluation has no side bias; vs M2REF the
final's goals were 106 own shots and 0 opponent own goals in 10 matches. All opponents here are weak (random and
earlier versions of the same lineage): this measures progress along the lineage, not absolute strength.

The own-goal inspection (scripts/own_goal_inspect.py) was prepared but not needed for the decision.

**Live Rocket League validation** of the promoted policy: docs/LIVE_TEST_2.md (goals, resets, demolition; one
reset bug found and fixed).
