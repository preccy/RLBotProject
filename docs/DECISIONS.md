# Research and decision record

Machine: Windows 11 Home 10.0.26200 (native, not WSL), AMD Ryzen 5 8400F (6C/12T), 32 GB RAM, RTX 5070 12 GB
(driver 610.88), Python 3.11.9 in `.venv`. Sources accessed 2026-11-14/15. Everything marked *verified* was run
or read in source on this machine; everything else is labelled.

## 1. Sources

| Source | What was used / found |
|---|---|
| https://wiki.rlbot.org/v5/ | v5 is "pre-release and nearly feature-complete" (beta). Bots connect to RLBotServer over sockets (FlatBuffers); TOML configs; `run_command` replaces `python_file`. |
| https://wiki.rlbot.org/v5/community/story-mode/ | "RLBot Story Mode for v5 is planned, but will not be finished until after the beta has ended." Story Mode exists only in the v4 RLBotGUI. It is a campaign for a *human* player; nothing documents a bot replacing that player. **Not pursued.** |
| https://wiki.rlbot.org/v5/botmaking/config-files/ | `bot.toml` / `*.bot.toml`, `[settings] agent_id, name, run_command, root_dir, loadout_file`; `run_command` runs from `root_dir`. `match.toml` sections `[rlbot] [match] [mutators] [[cars]]`. |
| https://wiki.rlbot.org/v5/botmaking/tick-rate/ | Tick rate = min(FPS, 120); VSync limits FPS. v5 expects 120 Hz bots. |
| https://wiki.rlbot.org/v5/botmaking/machine-learning-faq/ | General pointers (RocketSim, RLGym); no deployment specifics. |
| https://github.com/RLBot/python-interface @ `8a0850f` | PyPI package `rlbot`; latest 2.0.0b56 needs `rlbot_flatbuffers~=0.19.0`; Python >= 3.11; MIT. `MatchManager`, `load_match_config` read from source. |
| https://github.com/RLBot/core @ `11d5d86` | Matches the installed `RLBotServer.exe` product version `1.0.0+11d5d863...` (*verified*). |
| https://github.com/RLBot/flatbuffers-schema @ `e27de77` | 2026-10-04 changed the documented meaning of `BoostPadState.timer` from "seconds since pickup" to "seconds remaining". botcore therefore reconstructs pad timers from `is_active` edges. Schema changes after flatbuffers 0.18 (ConsoleCommand variant, env vars, game info) are additive. |
| https://github.com/RLGym/rlgym-rlbot @ `c7477a3` (PyPI 0.3.0) | RLBot **v5** + RLGym **v2** wrapper; pins `rlbot==2.0.0b52`, `rlbot_flatbuffers>=0.18.8`, `rlgym-compat>=2.0.0`; Python <3.13; Apache-2.0. Read in full: per-tick action map, `StepOffset(-1, ACTION_END)` for `RocketSimEngine(rlbot_delay=True)`, countdown handling (obs from the 440th countdown tick), missed-tick recovery, ">6 ms" warning, guidance "capped 120 or 240 FPS, not 360". README example passes `above_240_fps_mode`, which does not exist in the 0.3.0 code; the code was followed. |
| https://github.com/JPK314/rlgym-compat @ `50b70c8` (PyPI 2.3.6) | Builds RLGym-v2-like `GameState` from `GamePacket`. Read in full; found the stale `air_time_since_jump` after `dodge_timeout` returns to -1 (fixed in botcore) and the 6-tick `on_ground` heuristic during `AirState.Jumping` (documented limitation). |
| https://github.com/JPK314/rlgym-learn @ `2503793` (PyPI 2.0.0) | Rust process layer + config system. **Fails on native Windows** (section 3). |
| https://github.com/JPK314/rlgym-learn-algos @ `466b8e3` (PyPI 0.4.2) | PPO learner, Rust GAE processor, experience buffers. Read `ppo_learner.py`, `gae_trajectory_processor*.py`, `experience_buffer*.py`, `ppo_agent_controller.py`, multi-agent controller. Apache-2.0. |
| https://github.com/AechPro/rlgym-ppo @ `4ffd2e9` (installed 1.3.13) | Read `learner.py`, `batched_agent_manager.py`, `torch_functions.compute_gae`, `ppo_learner.py`. |
| https://rlgym.org/Rocket%20League/training_an_agent/ | Official example uses rlgym-ppo: action repeat 8, 30 s no-touch / 300 s timeouts, batch 100k, minibatch 50k, 2 epochs, lr 1e-4, ent 0.01, exp buffer 300k, 2048-2048-1024-1024 nets, rewards InAir 0.002 / SpeedTowardBall 0.01 / VelocityBallToGoal 0.1 / Goal 10. Expectation stated: entropy "should settle around 2". No skill/time claims. |
| https://github.com/RLGym/rlgym (installed 2.0.1) | Source of `RLGym`, `RocketSimEngine`, `LookupTableAction`, `KickoffMutator`, done conditions, `RLViserRenderer` read from the installed package. Apache-2.0. |
| https://github.com/RLGym/rlgym-tools @ `0c63b2a` | README only (category list). Not installed; nothing from it is used. |
| https://github.com/ZealanL/RocketSim @ `c2baacb` (installed 2.2.1) | MIT. Pad order, tick rate (120) and bindings checked in the installed wheel. Its README claim "accurate enough to train ML bots to SSL level" is a **maintainer claim, unverified here**. |
| https://github.com/ZealanL/RLGym-PPO-Guide @ `cef4469` | Only the landing README was retrieved (it targets RLGym-PPO); the detailed pages were not read this session. |
| https://pytorch.org/get-started/locally/ | cu128 wheels; installed `torch 2.7.0+cu128` contains `sm_120` kernels (*verified* arch list) and trains on the RTX 5070 (*verified* forward/backward/Adam). |
| EARL-pytorch `1706f48`, rocket-learn `976e255`, replay-pretraining `8c15e84`, Necto `2e6ed7d` | HEAD commits recorded; contents **not reviewed** this session. All are RLGym v1-era designs: use as ideas (entity attention, distributed self-play, replay pretraining), not as drop-in code. |
| https://github.com/tokio-rs/mio (`src/sys/windows/mod.rs`) | `IoSourceState::register` returns `ErrorKind::AlreadyExists` if the source is already registered (confirms the rlgym-learn diagnosis). |

## 2. Installed environment (preserved, extended)

Pre-existing and kept unchanged: Python 3.11.9 venv, `torch 2.7.0+cu128`, `numpy 1.26.4`, `rlgym 2.0.1`,
`rlgym-rocket-league 2.0.1`, `rocketsim 2.2.1`, `rlgym-ppo 1.3.13`. Added: `rlgym-learn 2.0.0`,
`rlgym-learn-algos 0.4.2`, `rlgym-rlbot 0.3.0` (+ `rlbot 2.0.0b52`, `rlbot_flatbuffers 0.18.11`, `rlgym-compat 2.3.6`,
`psutil`), `rlviser-py 0.6.13`, `pytest 9.2.0`, and botcore itself (editable, `--no-deps`).
`rlviser-py 0.6.14` was skipped because it would have upgraded numpy to 2.x, which `rlgym-rocket-league 2.0.1`
does not allow (`numpy<2`). `pip check` is clean. Full pins: `requirements-lock.txt`.

RLBot v5 (GUI, launcher, RLBotServer 1.0.0+11d5d86 and the bot pack) and Rocket League (Steam, `D:\SteamLibrary`)
were already installed. Neither was launched in this session.

## 3. Learner choice

| | rlgym-ppo 1.3.13 (installed) | rlgym-learn 2.0.0 + algos 0.4.2 | **Chosen: botcore collector + rlgym-learn-algos PPO** |
|---|---|---|---|
| Native Windows multiprocess | Works (Python sockets) | **Fails**: `FileExistsError: entity already exists` at process init. In `env_process_interface.rs` each parent UDP socket is registered with a handshake `mio::Poll` and again with the multiplexer without deregistering; mio's Windows backend rejects the second registration (`AlreadyExists`), Linux epoll allows it. Reproduced with 4 processes. | Works: spawn workers + Pipes, verified start/step/shutdown, 2-12 workers |
| Truncation handling | **Bug**: `compute_gae` bootstraps truncated steps with `values[step+1]`, the value of the *next trajectory's first state* in the concatenated buffer | Correct: Rust GAE uses each trajectory's own `final_val_pred` when truncated, 0 when terminal (*verified* against a reference, `tests/test_ppo.py`) | Same Rust GAE |
| Custom models | Fixed MLP classes | Any `ActorCritic` | Any `ActorCritic` (MLP and entity encoder) |
| Frozen opponents / multiple policies | No | MultiAgentController (needs the broken process layer) | Per-env role map; frozen agents produce no learner data (tested) |
| Data reuse | 300k FIFO buffer reuses earlier iterations (stale data) | Configurable | Buffer cleared every iteration: strictly on-policy |
| Checkpoint completeness | Weights, optimisers, return/obs stats; no RNG | Weights, optimisers, buffer; **return stats not restored** (`ExperienceBuffer.load` does not pass the checkpoint folder to the trajectory processor) | Weights, optimisers, return stats (restored explicitly), counters, RNG, config, contract, opponent pool; atomic |
| Maintenance | Last commit is the installed one | Active (2.0.0 released 2026-08-29) | Depends on algos only for PPO/GAE |

Trade-off: botcore owns ~400 lines of collection code instead of using rlgym-learn's faster Rust IPC, but every
line runs on this machine and the PPO maths remains upstream code. Re-evaluate rlgym-learn when a release fixes
Windows socket registration (or try 1.0.5, which used Python selectors; untested).

## 4. Other decisions

* **Observation contract obs-v1** (`docs/OBSERVATION_SPEC.md`): 276 floats, explicit car slots for up to 3v3 with
  presence masks, team-relative by 180-degree rotation (never reflection), fixed scales (no running normaliser
  to export), previous action, boost-pad timers. Inversion/parity/NaN/mask tests in `tests/test_obs.py`.
* **Action table**: RLGym's established 90-entry `LookupTableAction`, pinned by SHA-256
  (`86baa15c...`), so an upstream edit fails loudly instead of remapping indices.
* **Action repeat 8** (15 decisions/s) as in the RLGym guide; configurable. `rlbot_delay=True` in RocketSim, matching
  rlgym-rlbot's `StepOffset(-1, ACTION_END)`.
* **PPO settings** start from upstream: ent 0.01, 2 epochs, clip 0.2, grad-norm 0.5, gamma 0.99, lambda 0.95
  (RLGym guide / algos defaults) and **lr 3e-4** (the default of both rlgym-ppo and rlgym-learn-algos). The guide's
  1e-4 was tried first: on a velocity-to-ball diagnostic it learned ~3x slower (KL ~0.001 vs ~0.003-0.004).
  Deviations: batch 24,576 with 73,728 fresh steps per iteration (6 Adam steps/iteration, like the guide's
  100k batch over a 300k buffer) without reusing old iterations; 512x3 networks instead of 2048-wide (collection
  is CPU-bound; inference must fit a 6 ms budget). Stage-1 reward weights deviate from the guide (goals 3 instead
  of 10) for the measured reason in DESIGN.md section 3.
* **Workers**: 12 x 6 envs chosen by benchmark (STATUS.md). 4-worker `modest.toml` for using the PC meanwhile.
* **Deployment**: rlgym-rlbot for timing (it is the maintained RLGym v2 -> RLBot v5 path); botcore adds bundle
  checks, pad-timer and flip-window reconstruction, zero-action fallback and latency logging.
* **Licences**: RLGym, rlgym-learn(-algos), rlgym-rlbot, rlgym-compat: Apache-2.0; RocketSim: MIT; rlbot: MIT.
  Reused with attribution: RLGym's lookup table (generated from the installed package) and kickoff spawn values.
