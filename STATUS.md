# Project status

Last updated 2026-11-17 (long run 2 + overnight continuation). Nothing here implies a rank or Rocket League skill level; all match results
are in RocketSim against random or earlier versions of the same lineage unless stated otherwise.

## Current state (long run 2 + overnight, docs/EXPERIMENT_LONG2.md)

* **best_verified = `runs/s1-long2` iteration 9,900** (weights `d4ee480068c2`, 729.9M total steps; frozen copy
  `runs/reference/S2N-iter9900`), installed in `bot/model`. Previous deployed best S1L-iter2263 (`8d55b78c86c7`) is in
  `bot/model.previous`. Same configuration as long run 1 throughout (obs-v1, lookup-90, 512x3, unchanged rewards).
* Fresh-seed decisions (150 pairs each): vs S1L-iter2263 298-0-2, 0.993 [0.983, 1.000], own goals 0.23 vs 1.35 per
  match; vs overnight start S2L-iter3600 281-0-19, 0.937 [0.907, 0.963]. Regression checks vs R0 / M2REF 80-0-0 each.
* All-Star diagnostic (4 offline matches, not a rank estimate): pre 1-0-3 (15-16), interim 3-1-0 (21-5),
  iteration 9,900 4-0-0 (40-4), runner fault-free.
* No further training block has been started.

## Previous state: long run 1 (docs/EXPERIMENT_LONG1.md, docs/LIVE_TEST_2.md)

* best_verified (then) = `runs/s1-long` iteration 2,263 (weights `8d55b78c86c7`, 166.8M steps), also installed in
  `bot/model` (previous probe kept in `bot/model.previous`). It is a continuation of M2 run D-s202 for 155.0M steps
  with the configuration unchanged.
* Pre-registered decision, 150 fresh seed pairs each:
  * vs M2 reference: 300-0-0 (goals 3,401-101), score 1.000;
  * vs random: 298-0-2, 0.993 [0.983, 1.000];
  * own goals vs random: 0.73 -> 0.60 per match (difference -0.133 [-0.26, -0.013]).
  * Verdict: **demonstrated improvement**. It still beats its own midpoint 0.850 [0.775, 0.912], so it had not
    plateaued.
* Live Rocket League tests with this policy:
  * it scored natural goals;
  * goals, replays, kickoffs and a staged demolition were all handled;
  * 0 fallbacks; p99 callback 1.6 ms; 0 over 6 ms.
  * A missed kickoff reset after a 440-tick countdown was found and fixed (verified offline on the real recording and
    live 6/6).
* Training command for the next block (not started): see "Next experiment" in the report / EXPERIMENT_LONG1.md.
  Resume with `python -m botcore resume --run runs\s1-long --config <config with a higher total_iterations>`.

## Milestone 2 summary (details: docs/EXPERIMENT_M2.md, docs/LIVE_TEST_1.md)

**Collector audit (real RocketSim workers, `tests/test_collector_audit.py`).**
* Goals and resets: 14 goals produced 28 terminal trajectories, each with +1/-1 exactly once on the terminal step.
  Every terminal final observation shows the ball past the goal line (min 5,218 uu vs 5,216), on the correct side of
  each agent's own frame, so it is captured before the auto-reset. Truncated trajectories never carry goal reward.
* Continuity: no own-car jump above 105 uu within a trajectory (limit 280).
* Alignment and policy version: stored vs recomputed log-probs differ by exactly 0.0 over 1,500 pipelined
  transitions with frozen opponents; frozen slots never reach the learner (a confident, different frozen net
  differs by 0.65 nats on average, so leakage would show).
* Reward assignment: rewards land on the agent that earned them (blue +1.62 vs orange -1.62 in a controlled test).
* Runtime guard: every iteration recomputes 4,096 log-probs before the update and stops if they differ by more
  than 1e-3. Observed: <= 1.5e-6 on GPU over 1,280 iterations.

**First live Rocket League test (2026-11-15).** BotCore vs BotCore via RLBot v5, both colours.
* Bundle load: OK; 0 fallbacks, 0 policy faults.
* Ticks: 99.85% of in-play frame steps were single ticks.
* Callback latency: p99 1.7 ms; 2 of ~49,000 callbacks were over 6 ms.
* Controls: 100% of applied inputs were exact action-table rows.
* Orientation: correct for both colours; observations finite.
* Behaviour: the same as in RocketSim for the same policy.
* Found and fixed: rlgym-rlbot skipped the environment reset at overtime kickoffs. Verified by replaying the live
  recording through the real runner offline (1 -> 2 resets) and covered by a regression test.
* Not covered: a goal, a demolition, a competent policy.

**Visualisation.** `python -m botcore replay-html` writes a self-contained top-down HTML replay of a simulated
match or a live recording. Its rendering was checked with a headless Edge screenshot. RLViser 0.9.1 (the version
matching the installed rlviser-py 0.6.13) is downloaded and its hash verified, but running it was blocked in this
session; steps in README.

**Experiment M2 (pre-registered 2x2: lr x goal weight, 2 seeds, 11.8M steps each).**
* Neither effect is established. The combination (current default, cell D) looks best but with large seed variance.
* The default stays cell D, provisional.
* No run is confirmed to beat the random policy: the best 40-pair result (0.656) shrank to 0.525 [0.468, 0.583]
  on 100 fresh pairs.
* Training clearly changes behaviour: airborne time 83% -> 27% in the best run, ~30x more ball touches, and it beats
  the 4.4M probe (0.738 [0.650, 0.825]).
* Specific weakness: own goals (~0.6-1.1 per match); the random opponent scored no real goals at all.

**Action representation: kept.** No deployment problem was found, and trained policies do leave the near-random
air-heavy behaviour. The kickoff first-tick jump caveat is documented (docs/DESIGN.md section 2/8).

## Verdict and long-run plan

**Ready for a gated sustained run; not for an open-ended one.** The training, evaluation, checkpoint and deployment
paths are verified (see above). What is not yet shown is any match-level improvement over a random policy. That is
the question the next run must answer, with a stop rule:

* Command (PowerShell, project root):
  `python -m botcore train --config configs\default.toml --name s1-long --iterations 2100`
  This is cell D: lr 3e-4, goals 3, stage-1 shaping, 512x3 MLP, seed 1.
* Expected duration: measured 9.1-9.3 min per 11.8M steps over 74 min of continuous training, i.e. ~77M learner
  steps/hour (~21k steps/s). 2,100 iterations = 155M steps, about 2 hours, at ~80% CPU. Disk: ~60 MB/hour of
  opponent snapshots plus 5 x 16 MB checkpoints.
* Checkpoints and stopping:
  * A checkpoint is written every 20 iterations (~1.5M steps, ~70 s); the last 5 plus any pointer targets are kept.
  * Ctrl+C or creating `runs\s1-long\STOP` checkpoints after the current iteration (verified: ~3 s).
  * `python -m botcore resume --run runs\s1-long` restores weights, Adam state, return normaliser, counters and RNG
    exactly (verified). Environments restart with fresh episodes.
* **Gate 1 at ~155M steps** (all 300 s matches, stochastic). Evaluate the latest checkpoint:
  * vs R0 on 100 fresh pairs (`--seed-offset 200`);
  * vs E60 and vs `runs\m2-D-s202\exports\iter_000160` on 40 pairs each;
  * plus own goals per match.
  * Continue if vs R0 has CI lower bound > 0.5, or if it beats D-s202 with CI lower bound > 0.5 and its own goals
    per match are below D-s202's 1.06. A passing R0 result is then promoted (`export --mark-candidate`, `promote
    --initial`).
  * Otherwise stop and revise stage-1 rewards before spending more compute. The first change to test is a
    direction-aware touch reward (pay for ball velocity towards the opponent goal, not any hard hit).
* Remaining uncertainties: whether match-level gains appear by 155M steps; seed variance (one long seed); behaviour
  in game of a competent policy (goals, demos, kickoff jumps); RocketSim vs Rocket League physics differences.

## Milestone 1 record

## Machine and environment (verified)

Native Windows 11 Home 10.0.26200 (not WSL), Ryzen 5 8400F 6C/12T, 32 GB RAM, RTX 5070 12 GB (driver 610.88,
sm_120), Python 3.11.9 venv at `.venv`, torch 2.7.0+cu128 (CUDA forward/backward/Adam verified), RocketSim 2.2.1,
RLGym 2.0.1, rlgym-learn-algos 0.4.2, rlgym-rlbot 0.3.0 / rlbot 2.0.0b52 / rlgym-compat 2.3.6. RLBot v5
installed (RLBotServer 1.0.0+11d5d86); Rocket League installed (Steam, D:). Full pins: `requirements-lock.txt`;
`python -m botcore doctor` passes 11/11 checks (warning: optional `rlviser.exe` absent).

## Decisions (details in docs/DECISIONS.md and docs/DESIGN.md)

* PPO maths from rlgym-learn-algos 0.4.2 (PPOLearner + Rust GAE, verified against a reference); collection by
  botcore's own spawn-worker collector because rlgym-learn 2.0.0's process layer fails on native Windows (mio
  double registration); rlgym-ppo not used (truncation bootstrap bug, no frozen opponents).
* obs-v1: 276 floats, team-relative by rotation, 3v3 slots with presence masks, fixed scales, previous action,
  pad timers. Action table: RLGym 90-entry lookup (SHA-pinned). Action repeat 8 (15 Hz), RLBot-style 1-tick delay.
* Stage-1 rewards: goals 3/3, touch 0.5, velocity-to-ball 0.5/s, ball-to-goal 1.5/s, boost potential 0.1;
  lr 3e-4; 512x3 MLP; 12 workers x 6 envs; 73,728 fresh steps and 6 Adam steps per iteration.
* Deployment through rlgym-rlbot (timing) + botcore bundle checks, pad-timer/flip reconstruction, zero-action
  fallback, latency logging, optional packet recording. CPU inference by default.

## Verified results

### Tests
`python -m pytest`: **140 passed** (obs contract and team inversion/parity, presence masks, NaN safety, action
table identity and jump semantics, reward single-counting and time scaling, scenario validity for every scenario
and team size, GAE terminal-vs-truncation bootstrapping, rollout boundaries, frozen-opponent isolation, pipelined
= lockstep data, checkpoint atomicity/retention/RNG, export round trip and every rejection path, config
validation, simulator-vs-RLBot-packet observation parity in 1v1/2v2 with both colours, pad tracker, flip
expiry, packet recording/replay, evaluation reproducibility and scoring, promotion rule, end-to-end smoke).
`python -m botcore smoke-test`: all 15 checks pass in ~10 s (train, exact resume of weights/Adam/normaliser/
counters, export, RLBot runner load + fallback, evaluation).

### Training throughput (`python -m botcore benchmark`, 4 iterations each, first excluded, includes collection, GAE, PPO update and checkpoint writes; batch 16,384)

| Workers x envs | Learner steps/s | Env steps/s | Physics ticks/s | x real time | CPU (12 threads) | RAM (process tree) | VRAM peak |
|---|---|---|---|---|---|---|---|
| 4 x 6 | 8,907 | 4,453 | 35,627 | 297 | 51% | 2.0 GB | 328 MB |
| 6 x 6 | 11,994 | 5,997 | 47,975 | 400 | 57% | 2.2 GB | 344 MB |
| 8 x 6 | 14,513 | 7,257 | 58,053 | 484 | 66% | 2.3 GB | 344 MB |
| 10 x 6 | 16,983 | 8,491 | 67,930 | 566 | 72% | 2.4 GB | 339 MB |
| **12 x 6** (default) | **19,092** | 9,546 | 76,370 | 636 | 81% | 2.6 GB | 374 MB |

Learner steps count both cars of a 1v1 (2 per env step). Collection is CPU-bound (PPO update ~0.1-0.2 s of a ~4 s
iteration). Sustained over the 120-iteration probe: 16.7k-20.9k learner steps/s (~1.65M steps/min).

### Models (`python -m botcore benchmark-models`, RTX 5070 / one CPU thread)

| | Actor params | CPU inference, batch 1 | GPU rollout batch 96 | GPU update, batch 16,384 | Peak VRAM (update) | End-to-end training |
|---|---|---|---|---|---|---|
| MLP 512x3 (default) | 713,306 | 0.09 ms | 0.64 ms | 15.7 ms | 323 MB | ~19k steps/s |
| Entity encoder (d=128, 2 layers, 4 heads) | 500,314 | 0.58 ms | 2.48 ms | 288 ms | 6.66 GB | ~8.7k steps/s (with 4-way gradient accumulation) |

The entity encoder trains and exports correctly but has **not** been compared for playing strength.

### Deployment latency (`python -m botcore latency`, synthetic packets, 2,950 decisions, 512x3 MLP)
Full per-decision path (packet -> rlgym-compat state, pad/flip reconstruction, observation, policy, 8
ControllerStates), excluding sockets: **CPU p50 0.34 ms, p99 0.67 ms, max 1.54 ms**; GPU p50 0.75 ms, p99 1.42 ms,
max 3.03 ms. 0 decisions over the 6 ms budget on either. CPU is the default.

### Learning probes (bounded, 1v1, stage-1 scenario mix)

| Run | Steps | Result |
|---|---|---|
| goals 10, guide-like shaping, lr 1e-4 | 8.85M | No measurable learning (entropy 4.500 -> 4.488, KL ~1e-5) |
| velocity-to-ball only, lr 1e-4 / 3e-4 | 2.2M each | Learned; 3e-4 ~3x faster (speed towards ball 0 -> +0.11 of max; KL 0.003-0.004, clip 0.03-0.05) |
| goals 10, lr 3e-4 | 5.2M | Still no measurable learning (goal noise dominates) |
| **stage-1 weights, lr 3e-4** (`runs/s1w-probe`) | 4.4M | Learned contact: entropy 4.50 -> 4.16, touch signal x7, speed towards ball 0 -> ~+0.1 of max, episodes ending in a goal 22% -> 43% |
| Evaluation of that policy vs its iteration-0 self | 16 pairs, 5-min matches | score 0.469 (95% CI 0.344-0.594) sampled; 0.453 argmax. **No measurable strength gain yet.** |

These show the loop learns; they say nothing about Rocket League skill.

## Not verified (needs you or later work)

* **Any live Rocket League / RLBot match.** The bot folder, match files and runner are built and offline-tested
  (RLBot's own config loader parses them; the runner loads bundles and handles faults), but the game was not
  launched. rlbot 2.0.0b52 has not been run against the installed RLBotServer build.
* Observation parity with real packets (only synthetic packets from RocketSim states), real timing at 120/240 FPS,
  missed-packet behaviour, kickoff/goal-reset behaviour in game.
* RLViser window (`watch`), since `rlviser.exe` is not installed; `watch --dry-run` works.
* Story Mode: not available for RLBot v5 ("planned ... after the beta"); not attempted.
* Playing strength of anything; team play; the entity encoder's value; long-run stability.

## Commands (PowerShell, `C:\Users\TomPr\RLBotProject`, after `.\.venv\Scripts\Activate.ps1`)

```
python -m botcore doctor
python -m pytest
python -m botcore smoke-test
python -m botcore train --config configs\default.toml --name s1-a          # long run: your call to start
python -m botcore resume --run runs\s1-a
python -m botcore export --checkpoint runs\s1-a --to-bot
python -m botcore evaluate --a runs\s1-a --b runs\s1-a\opponents\iter_000000 --pairs 50
python scripts\start_match.py bot\match_vs_allstar.toml                      # or use the RLBot GUI
```

## Next actions

1. Start the first long stage-1 run (hours), evaluate every few hours vs early snapshots, move to stage 2 by the
   gate in docs/DESIGN.md section 4.
2. First real-game check: `export --to-bot`, cap RL at 120 or 240 FPS, run `match_vs_allstar.toml` with
   `$env:BOTCORE_RECORD="runs\rec\m1.bcp"`, then `python -m botcore replay --recording runs\rec\m1.bcp --policy bot\model`.
   Report the bot's status log lines (latency, fallback ticks) and anything odd.
3. MLP vs entity encoder at equal wall-clock (docs/DESIGN.md section 9).

Probe runs kept for reference: `runs/s1w-probe` (stage-1 weights, 4.4M steps), `runs/s1-probe-lr3e-4`,
`runs/diag-v2b-lr1e-4`, `runs/diag-v2b-lr3e-4`, `runs/probe-entity`; benchmark JSON in `runs/benchmarks/`,
evaluation reports in `runs/eval/`.
