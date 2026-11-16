# Live integration test 1 (2026-11-15, real Rocket League via RLBot v5)

Setup: `scripts/live_check.py` with `bot/match_live_selfplay.toml`: BotCore (blue) vs BotCore (orange), both
running the Milestone-1 probe bundle `a75feaa9cfb1` (4.4M steps), CPU inference, sampled actions, 5-minute
match, Stadium_P. RLBot started Rocket League through Steam; RLBotServer `1.0.0+11d5d86` with the Python interface
`rlbot 2.0.0b52` (the first time this combination ran). Data: `runs/live/20261115-232517/` (logs, `.bcp`
recordings, summaries, `live_report.json`); viewable with `runs/viz/live-20261115-232517.html`.

The match was regulation 0-0 and went into Rocket League's unlimited overtime; it was stopped after ~6.8 minutes
of play (the bots retired cleanly and wrote their summaries). `live_check.py` now stops at overtime by default
and the self-play match file caps overtime as a backstop.

## Measured (both bots unless stated)

| Check | Result |
|---|---|
| Bundle load, contract check, both colours initialised | yes, no load errors |
| Packets received | 49,365 per bot, 121 packets/s over 408 s wall (includes a 12 s game pause) |
| Tick continuity during Kickoff/Active | 45,960 of 46,030 frame steps were exactly 1 tick (99.85%); 70 gaps of 2-6 ticks |
| Full callback time (what rlgym-rlbot's 6 ms window refers to) | p50 0.30 ms, p99 1.7 ms, p99.9 2.3 ms; max 3.8 ms (blue), 7.1 ms (orange); **over 6 ms: 0 (blue), 2 (orange)** of ~49,350 |
| Policy inference per decision | p50 0.58 ms, p99 1.04 ms, max 2.9 ms (CPU) |
| Fallback ticks / policy faults | **0 / 0** |
| Timing recoveries logged by rlgym-rlbot | per bot: 26 "incorrect action on a previous tick", 14 "missed step tick", 13-14 "missed action" (~0.9% of 5,767 decisions); identical counts on both bots, matching the 70 game-side frame gaps |
| Environment resets | 1 per bot (opening kickoff); 0 on-the-fly resets. **Finding:** the overtime kickoff (Active -> Countdown at frame 37,137) did not reset the environment in rlgym-rlbot 0.3.0. Fixed in BotCoreBot; replaying this recording through the real runner offline reproduces the live run exactly (1 reset, 5,767 decisions) without the fix and gives 2 resets with it |
| Applied inputs (`last_input` reported by the game) | 100.0% of 46,033 in-play ticks are exact rows of the action table; 2% all-zero |
| Observations | 0 non-finite values; max abs value 1.87; 30-31 of 276 features had some values outside the 0.1-99.9% range of a 8,000-sample simulator reference of the same policy, at most 2.7% of decisions (ball angular velocity z, two big-pad timers, opponent relative position/velocity y) |
| Orientation | kickoff spawn world y: blue -2560/-3840, orange +2560/+3840 (consistent with team); team-frame observations put both cars in their own half at kickoff |
| Kickoff behaviour | after 1 s: blue 1,300 uu/s and 845 uu closer to the ball; orange 1,035 uu/s, 726 uu closer. Orange chose a jump action as its first kickoff decision once, which rlgym-rlbot turns into a held jump (documented rlgym-rlbot caveat) |
| Behaviour vs simulator, same policy | jump actions 14.3% / 15.1% of decisions live vs 14.8% in RocketSim; airborne 74% / 76% live vs 80% sim; flip used 55-58% vs 55% |

## What this does and does not verify

Verified live: the RLBot v5 runner starts from `bot.toml`, connects with rlbot 2.0.0b52 to the installed server,
loads and contract-checks the bundle, plays both colours with correct team-relative observations, keeps
processing far inside the 6 ms window on CPU, and the game applies exactly the table controls the policy chose.
The random-looking flipping seen in the match is the policy's own behaviour (same rates in the simulator); this
4.4M-step policy is near-random and does not beat a random policy in simulation either.

Not verified: per-feature equality between the live observation and a simulator ground truth (no ground truth
exists for the real game; only synthetic-packet parity is tested), behaviour of a competent policy, kickoff
timing with a policy that jumps at kickoff, goal/replay resets (no goal was scored), demolitions, other
frame-rate settings (the in-game cap was not read; the measured tick continuity was 99.85%), GPU inference
in game. RLBot's on-screen overlay briefly reported "76 Hz, Poor" while the Windows Start menu was open over
the game window at startup; the recording shows ~120 Hz for the rest of the match.
