# Experiment M2: learning rate and goal weight (pre-registered)

Written and committed **before** any M2 run started. Results are appended in a separate section afterwards;
nothing above "Results" is edited after the runs begin.

## Question

Milestone 1 changed two things at once after probing: the learning rate (1e-4 -> 3e-4) and the goal reward
(10 -> 3). Do they each help this MLP learn to *win matches* (not just collect shaping reward), and which one
matters? Both settings are provisional until this is answered.

## Design: 2 x 2 factorial, 2 seeds per cell, everything else fixed

| Cell | actor/critic lr | goal_scored / goal_conceded | Config |
|---|---|---|---|
| A | 1e-4 | 10 | `configs/experiments/m2/A.toml` |
| B | 3e-4 | 10 | `configs/experiments/m2/B.toml` |
| C | 1e-4 | 3 | `configs/experiments/m2/C.toml` |
| D | 3e-4 | 3 (current default) | `configs/experiments/m2/D.toml` |

Fixed across cells: `configs/default.toml` otherwise (512x3 MLP, 12 workers x 6 envs, 73,728 steps/iteration,
batch 24,576, 2 epochs, ent 0.01, gamma 0.99, lambda 0.95, stage-1 scenario mix, shaping weights touch 0.5,
velocity_to_ball 0.5, ball_to_goal 1.5, boost_economy 0.1).

* Seeds: run seeds **101** and **202** for every cell (the same two seeds in every cell, so seed effects are
  shared across conditions).
* Budget: **160 iterations = 11,796,480 learner steps** per run (about 11 minutes at the measured ~19k steps/s).
  8 runs. Checkpoints every 40 iterations (kept).
* Evaluation set: seeds 1,000,000-1,000,039 (40 colour-swapped pairs = 80 matches), 300 s regulation + up to 120 s
  overtime, stochastic (sampled) inference on both sides, RocketSim. Same seed set for every comparison.
* Fixed opponents (read-only copies in `runs/reference/`):
  * **R0** = random initial policy (`R0-random-init`, weights `b3e66cfbd006`).
  * **E60** = Milestone-1 probe after 4.4M steps (`E60-probe-4.4M`, weights `a75feaa9cfb1`).

## Measurements

1. Each run's iteration-160 policy vs R0 and vs E60 (40 pairs each). Primary metric: score vs R0
   (win 1, draw 0.5, loss 0), 95% percentile bootstrap over seed pairs.
2. Factor head-to-heads at iteration 160, per seed (40 pairs each):
   learning rate: D vs C (at goal 3), B vs A (at goal 10); goal weight: D vs B (at lr 3e-4), C vs A (at lr 1e-4).
3. Diagnostics only (not used to judge strength): training entropy, KL, clip fraction, touch rate, speed towards
   the ball, episode goal fraction; evaluation behaviour (touches/min, mean speed, ball distance, airborne
   fraction, boost).

## Decision rules (fixed now)

* A run **beats random** if its score vs R0 has 95% CI lower bound > 0.5.
* A factor's effect is **established** only if all four of its head-to-heads (2 seeds x 2 levels of the other
  factor) favour the same level (point estimate > 0.5 for that level) **and** at least two of them have a CI
  excluding 0.5. Otherwise it is **not established**, whatever the reward curves look like.
* The default config changes only for an established effect. If neither effect is established, the current
  default (cell D) stays, labelled provisional.
* Best-verified model: the iteration-160 policy with the highest mean score vs R0 is re-evaluated vs R0 over
  **100 pairs** (seeds 1,000,100-1,000,199, not used above); it becomes the first `best_verified` only if the CI
  lower bound is > 0.5. Later candidates must pass the standing promotion rule (100 pairs vs best_verified).
* No rank, MMR or "skill level" will be inferred from any of this.

## Known limits of this design

Two seeds per cell detect only large effects; 11.8M steps is early training, so a factor that matters later
(e.g. goal weight once play is coherent) may not show here. A null result means "not established at this
budget", not "no effect".
