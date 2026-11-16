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
