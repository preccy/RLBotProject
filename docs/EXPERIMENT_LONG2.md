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

## Overnight continuation (authorised 2026-11-17 ~01:29; rules written before it started)

* **Start:** `runs/s1-long2` iteration 3,600 (265,420,800 total steps), the latest valid checkpoint; frozen byte-identical
  read-only copies `runs/reference/S2L-iter3600-checkpoint` and `runs/reference/S2L-iter3600` (weights `7b35dd6a3e77`).
  Untouched: best_verified (`runs/s1-long` iteration 2,263, `8d55b78c86c7`) and the deployed `bot/model` (same weights).
* **Config** `configs/experiments/long2-overnight.toml`: identical learning setup; only `total_iterations` (9,900 =
  3,600 + 6,300) and `keep_last` (200, so nothing from this run is pruned) differ.
* **Budget:** at most 7 h from the instruction (hard end 08:29). 6,300 iterations (464.5M added steps) ~6.5 h at
  3.7 s/iteration, so it should end by itself. Watchdog: if still running at 08:20 a STOP file is created, which
  checkpoints the current iteration and exits cleanly (no force-kill). NaNs / policy faults / checkpoint errors raise
  and stop training; they are reported, nothing is restarted.
* **~2-hourly diagnostics** (checkpoints ~5,550, ~7,500, ~9,450): 40 pairs vs S2L-iter3600 on seeds 0-39 (baseline:
  iteration 3,600 vs itself 0.537 [0.463, 0.613]); plus training entropy and log-prob guard from metrics.jsonl.
  **Deterioration flag:** score < 0.40, or own goals/match more than 0.5 above the start's in the same matches, or
  touches/min below 70% of the start's, or policy entropy < 2.0 (collapse / repetitive play). Lower airtime or boost
  alone is not a flag. A flag is confirmed with 40 more pairs on seeds 7,000+; if confirmed, training is stopped with
  the STOP file.
* **Final (fresh seeds):** final vs S2L-iter3600, 150 pairs, offsets 4,000-4,149; final vs best_verified S1L-iter2263,
  150 pairs, offsets 5,000-5,149; regression checks vs R0 and M2REF (40 pairs, seeds 0-39); final vs S2L-iter3600 on
  seeds 0-39 for comparability with the diagnostics.
* **Earlier candidate:** if an intermediate checkpoint's diagnostic score beats the final's (seeds 0-39) by > 0.10, it
  is played against FINAL on seeds 6,000-6,149; if it wins (lower bound > 0.5) it replaces FINAL as the candidate and
  is evaluated against best_verified on independent seeds 8,000-8,149.
* **Promotion:** the candidate becomes best_verified only with 95% CI lower bound > 0.5 vs best_verified (150 fresh
  pairs) and >= 0.45 in both regression checks. Otherwise S1L-iter2263 stays best_verified and deployed.
* **If promoted:** the fixed All-Star series is repeated with it (`--label night-<iter>`); it is installed in
  `bot/model` (old model kept in `bot/model.previous` and `runs/reference`) only if the live runner shows no faults.
* Afterwards every training/test process is stopped; Windows is left running.

## Overnight results (appended 2026-11-17 ~07:40)

**Training.** Resumed from S2L-iter3600 at 01:33; reached the 9,900-iteration target and stopped by itself at 06:53:34
(~5 h 20 min, inside the 7 h budget; the 08:20 watchdog never fired). 6,300 iterations = **464,486,400 added learner
steps** (total 729,907,200), ~24k steps/s. No NaNs, policy faults, checkpoint errors or tracebacks; all checkpoints
complete. Entropy ~3.07 -> ~3.1 (no collapse); runtime log-prob guard max ~1.2e-5.

**~2-hourly diagnostics** (40 pairs vs S2L-iter3600, seeds 0-39; not decision inputs; no deterioration flag at any):

| Iteration | W-D-L | Score [95% CI] | Goals | Own goals / match (cand. vs start) | Touches / min (cand. vs start) | Airborne |
|---|---|---|---|---|---|---|
| 5,550 | 65-0-15 | 0.812 [0.725, 0.887] | 485-285 | 0.46 vs 0.83 | 42.5 vs 39.3 | 0.13 |
| 7,500 | 60-0-20 | 0.750 [0.637, 0.863] | 480-308 | 0.44 vs 0.73 | 33.7 vs 39.1 | 0.15 |
| 9,450 | 74-0-6 | 0.925 [0.863, 0.975] | 579-193 | 0.25 vs 0.86 | 40.7 vs 36.3 | 0.15 |
| 9,900 (final) | 72-0-8 | 0.900 [0.825, 0.963] | 552-215 | - | - | - |

Earlier-candidate rule: no intermediate checkpoint beat the final's 0.900 by > 0.10, so FINAL (iteration 9,900) is
the only candidate.

**Final evaluations (fresh seeds):**

| Comparison | Pairs | W-D-L | Score [95% CI] | Goals | Own goals / match (A vs B) |
|---|---|---|---|---|---|
| iter 9,900 vs S2L-iter3600 (offsets 4,000-4,149) | 150 | 281-0-19 | 0.937 [0.907, 0.963] | 2,200-815 | 0.29 vs 0.97 |
| iter 9,900 vs best_verified S1L-iter2263 (offsets 5,000-5,149) | 150 | 298-0-2 | 0.993 [0.983, 1.000] | 2,707-570 | 0.23 vs 1.35 |
| iter 9,900 vs R0 (seeds 0-39) | 40 | 80-0-0 | 1.000 [0.912, 1.000] | 1,317-14 | - |
| iter 9,900 vs M2REF (seeds 0-39) | 40 | 80-0-0 | 1.000 [0.912, 1.000] | 1,307-7 | - |

Own-goal difference per match (descriptive): vs S2L-iter3600 -0.68 [-0.81, -0.55]; vs S1L-iter2263 -1.13 [-1.26, -0.99].
Behaviour vs S1L-iter2263 in the same matches: touches/min 40.4 vs 34.4, airborne 0.14 vs 0.08, mean boost 8.4 vs 5.7,
supersonic 3.0% vs 2.2%.

**Promotion: yes.** Lower bound 0.983 > 0.5 vs best_verified on 150 fresh pairs; regression checks 1.000 and 1.000
(>= 0.45); `check_promotion` reports no problems. `runs/s1-long2` pointers: candidate = best_verified =
`checkpoints/iter_009900` (weights `d4ee480068c2`). Frozen read-only copies: `runs/reference/S2N-iter9900-checkpoint`
and `runs/reference/S2N-iter9900` (sha256-verified).

**All-Star series (iteration 9,900, same fixed setup; diagnostic only, not a rank estimate):** **4-0-0, goals 40-4**
(9-0, 13-1, 9-1, 9-2); BotCore own goals 0; shots 59 vs 9; All-Star saves 13, BotCore saves 0. Runner: fallbacks 0,
missed kickoff resets 0, p99 inference latency <= 1.6 ms, no tracebacks; rlgym-rlbot tick-recovery warnings (12 missed
step ticks over four matches) at the level of the earlier series (8 and 21). Details: `runs/allstar/night-9900/`.

| Series | Policy | W-D-L | Goals | BotCore own goals | Shots (BotCore vs All-Star) |
|---|---|---|---|---|---|
| pre | S1L-iter2263 | 1-0-3 | 15-16 | 2 | 25 vs 17 |
| interim | S2L-iter3600 | 3-1-0 | 21-5 | 0 | 38 vs 7 |
| night-9900 | S2N-iter9900 | 4-0-0 | 40-4 | 0 | 59 vs 9 |

**Deployed:** the runner showed no faults, so iteration 9,900 was installed in `bot/model` (`d4ee480068c2`); the previous
deployed model S1L-iter2263 (`8d55b78c86c7`) is in `bot/model.previous` and `runs/reference/S1L-iter2263`.

**Not tested / caveats:** all simulator opponents are earlier policies of the same lineage plus a random policy, so the
scores measure progress along the lineage, not absolute strength. The All-Star series is four matches against one
fixed built-in bot. Rocket League saved no replay files (`skip_replays`); the gameplay record is the packet recordings
and HTML replays. No HTML replays of iteration 9,900 vs the start in RocketSim were made overnight. The four-hour
block's originally planned FINAL-vs-S1L decision on offsets 4,000-4,149 was superseded by the overnight plan above.
