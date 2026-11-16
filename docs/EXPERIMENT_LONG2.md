# Long run 2: bounded four-hour continuation (pre-registered)

Written and committed before the pre-training All-Star matches and before training. Results are appended below
"Results" afterwards; nothing above it is edited once the run begins.

## Starting point

* Start: **long run 1 final, `runs/s1-long` iteration 2,263** (166,846,464 learner steps), the current
  `best_verified` policy (weights `8d55b78c86c7`), with its full training state (weights, both Adam states, return
  normaliser, counters, RNG).
* Saved unchanged: byte-identical, read-only copies in `runs/reference/S1L-iter2263-checkpoint` (checkpoint) and
  `runs/reference/S1L-iter2263` (export bundle); training resumes from the reference copy. This bundle is **START**.
* Why: it passed the standing promotion rule and long run 1 was still improving at its budget (final beat its own
  midpoint 0.850 [0.775, 0.912]).

## Training

* Config `configs/experiments/long2.toml`: identical to long run 1 except `run.name` (`s1-long2`) and
  `run.total_iterations`. The observation contract (obs-v1), action table (rlgym-lookup-90-v1), architecture
  (512x3 MLP), rewards, PPO settings, scenario mix and seed are unchanged; `resume` refuses any learning-relevant
  difference.
* Budget: **3,850 additional iterations x 73,728 = 283,852,800 additional learner steps**, ~3.94 h at the long run 1
  rate (3.69 s/iteration). `total_iterations` is absolute: 6,113 = 2,263 + 3,850. Hard wall-clock ceiling 4 h 10 min:
  if reached first, training is stopped and the last saved checkpoint is FINAL.
* Checkpoints every 50 iterations, all kept. Stop at the budget; no further block is started, and nothing in the
  learning configuration is changed.

## Evaluations (existing harness: paired seeds, colours swapped within a pair, 300 s + up to 120 s overtime, stochastic)

* **Decision (fresh seeds, never used before): FINAL vs START, offsets 4,000-4,149 = 150 pairs (300 matches).**
  Reported: W-D-L, score with 95% CI over seed pairs, goals for/against, own goals per match for each side, own-goal
  share of goals conceded, touches per minute, airborne fraction.
* **Regression checks (diagnostic, old diagnostic seeds 0-39, 40 pairs):** FINAL vs R0 and FINAL vs M2REF.
* **Visible play:** behaviour metrics of both sides in the decision matches; HTML replays of FINAL vs START in
  RocketSim (fixed seeds); the All-Star matches below.

## Decision rules (fixed now)

* **Improvement continued** if FINAL vs START has 95% CI lower bound > 0.5 (150 pairs; this is also the standing
  promotion rule: >= 100 pairs, lower bound > 0.5, and FINAL must score >= 0.45 in the regression checks).
* **Regression** if the CI upper bound < 0.5. Otherwise **inconclusive**.
* **Own goals (reported, not a gate):** per seed pair, FINAL own goals per match minus START own goals per match in the
  same decision matches, with a 95% bootstrap CI. Each side faces a different opponent here, so this is descriptive.
* **Retention:** FINAL becomes `best_verified` (and is installed in `bot/model`) only if improvement continued and the
  regression checks pass. Otherwise START stays `best_verified` and stays in `bot/model`; FINAL is kept as the run's
  `latest` only. The latest checkpoint is never promoted automatically.
* **CI method:** the pair bootstrap, except that a sample in which every pair is a clean sweep gets the exact
  Clopper-Pearson bound over pairs (e.g. 150/150 -> [0.976, 1]) rather than [1, 1] (fix in `evaluate.bootstrap_ci`,
  made before this run).

## External diagnostic: Psyonix All-Star in Rocket League (not a rank estimate)

* Fixed set (`scripts/allstar_series.py`): 4 offline 1v1 matches, standard 5 minutes on Stadium_P, BotCore blue,
  orange, blue, orange vs the built-in All-Star bot; BotCore samples stochastically, CPU inference, current runner
  (`bot.toml`). A match tied at the end of regulation is stopped (Rocket League overtime is unlimited) and counted as
  a draw.
* **Before training:** START (`--label pre`). **After training:** FINAL (`--label post`), same setup.
* Recorded: scorelines, Rocket League scoreboard shots/saves/own goals for both cars, touches, airborne fraction,
  speed, boost, runner health (fallbacks, latency, kickoff resets); Rocket League replay files (Demos folder) and an
  HTML top-down replay per match.
* Four matches against one fixed opponent is a small diagnostic. It is never used for promotion.

---

## Results

**Pre-training All-Star series (START, 4 matches):** 1-0-3, goals 15-16 (3-4, 4-6, 5-1, 3-5); BotCore own goals 2.
Details: `runs/allstar/pre/summary.md`. Rocket League saved no replay files (`skip_replays = true` loads the map with
`?noreplay`); the packet recordings and HTML top-down replays (`runs/allstar/pre/m*.html`) are the gameplay record.

**Pause at the user's request (2026-11-16, ~21:40):** training stopped at iteration 3,648; last checkpoint
iteration 3,600 (265,420,800 total steps; 98.5M of the 283.9M budget). Interim diagnostic (old seeds 0-39, 40 pairs,
not a decision input): iteration 3,600 vs START 66-0-14, score 0.825 [0.725, 0.912], goals 485-257; own goals per
match 0.39 (iter 3,600) vs 1.01 (START) in the same matches.

**Interim All-Star series (iteration 3,600, same fixed setup, user-requested; diagnostic only):** 3-1-0, goals 21-5
(7-0, 6-3, 6-0, 2-2 stopped at regulation); BotCore own goals 0; BotCore shots 38 vs All-Star 7 (pre: 25 vs 17).
Details: `runs/allstar/interim-iter3600/summary.md`.
