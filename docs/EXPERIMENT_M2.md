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

---

## Results (appended 2026-11-16, after all runs and evaluations)

All 8 runs completed as specified (9.1-9.3 min each, 11,796,480 learner steps; runtime log-prob alignment guard
<= 1.5e-6 throughout). Reports: `runs/m2/eval/*.json`, summary `runs/m2/summary.json`.

| Run | lr | goal | vs R0: score [95% CI], W-D-L, goals | vs E60: score [95% CI], goals | Eval behaviour: touches/min, speed, ball dist, airborne |
|---|---|---|---|---|---|
| A-s101 | 1e-4 | 10 | 0.500 [0.438, 0.562], 19-42-19, 24-23 | 0.519 [0.419, 0.619], 58-51 | 0.7, 494, 1882, 0.82 |
| A-s202 | 1e-4 | 10 | 0.544 [0.475, 0.619], 19-49-12, 23-16 | 0.519 [0.431, 0.606], 64-59 | 0.5, 486, 1962, 0.82 |
| B-s101 | 3e-4 | 10 | 0.481 [0.400, 0.562], 24-29-27, 37-37 | 0.463 [0.362, 0.562], 46-66 | 1.2, 501, 1504, 0.83 |
| B-s202 | 3e-4 | 10 | 0.469 [0.381, 0.556], 23-29-28, 30-35 | 0.600 [0.481, 0.719], 88-68 | 1.1, 510, 1640, 0.78 |
| C-s101 | 1e-4 | 3 | 0.525 [0.444, 0.600], 28-28-24, 42-39 | 0.525 [0.425, 0.625], 70-65 | 1.7, 474, 1245, 0.79 |
| C-s202 | 1e-4 | 3 | 0.656 [0.581, 0.725], 39-27-14, 64-29 | 0.562 [0.469, 0.656], 91-75 | 1.5, 527, 1384, 0.57 |
| D-s101 | 3e-4 | 3 | 0.519 [0.419, 0.619], 32-19-29, 60-55 | 0.631 [0.525, 0.738], 117-87 | 2.7, 491, 940, 0.77 |
| D-s202 | 3e-4 | 3 | 0.562 [0.456, 0.662], 38-14-28, 93-69 | 0.738 [0.650, 0.825], 178-77 | 4.5, 717, 993, 0.27 |

R0 itself in these matches: 0.14 touches/min, mean ball distance 5,023 uu, airborne 0.83.

Head-to-heads (score of the first-named cell, 40 pairs):

| Factor | seed 101 | seed 202 | Verdict |
|---|---|---|---|
| lr 3e-4 vs 1e-4 at goal 3 (D vs C) | 0.531 [0.450, 0.613] | **0.800 [0.713, 0.875]** | |
| lr 3e-4 vs 1e-4 at goal 10 (B vs A) | 0.463 [0.369, 0.556] | 0.406 [0.319, 0.500] | **not established** (signs disagree) |
| goal 3 vs 10 at lr 3e-4 (D vs B) | 0.550 [0.456, 0.644] | **0.838 [0.750, 0.912]** | |
| goal 3 vs 10 at lr 1e-4 (C vs A) | 0.494 [0.400, 0.594] | 0.481 [0.400, 0.562] | **not established** (signs disagree) |

Decisions under the pre-registered rules:

* **Neither effect is established.** The data suggest an interaction (the combination lr 3e-4 + goal 3, cell D, wins
  both of its seed-202 head-to-heads by a wide margin; each change alone does not help), with large seed variance.
  Per the rules the default stays cell D, **labelled provisional**.
* **Beats random:** only C-s202 met the criterion on the 40-pair set. Its pre-registered 100-pair confirmation on
  unseen seeds (1,000,100-1,000,199) gave **0.525 [0.468, 0.583]** (68-74-58, goals 112-91): **not confirmed**. The
  0.656 was a selection effect (highest of eight noisy estimates). **No best_verified model is set.**
* **Behaviour (diagnostic):** training does reduce the air-heavy play of the near-random start: D-s202 is airborne
  27% of the time (R0: 83%), faster, and closer to the ball, with ~30x R0's touch rate.
* **Why near-0.5 against a random opponent (diagnostic, `scripts/goal_attribution.py`, 12 matches each):** R0 scored
  0 real goals; every goal it is credited with was an **own goal** by the learner (D-s202 and C-s202: 12 scored / 7
  own goals and 10 / 7; the two runs printed unlabelled, so which is which is not recorded). The built-in
  attribution on 16 further matches gives D-s202 1.06 own goals per match vs R0's 0. The learners score about one
  goal per 5-minute match but put ~0.6-1.1 per match into their own net. Undirected contact (the stage-1 touch reward pays for hard hits in any direction) is the specific weakness.
  Own goals per match are now reported by `botcore evaluate` (behaviour.own_goals_per_match).

Interpretation limits: 11.8M steps is very early; two seeds; one budget. "Not established" means not at this budget.
