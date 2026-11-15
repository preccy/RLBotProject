# Project status

Last updated 2026-11-15. Milestone 1 (complete training-to-deployment loop) is built and tested in simulation.
**No policy has been trained beyond short probes; nothing has been run in Rocket League yet.**

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
