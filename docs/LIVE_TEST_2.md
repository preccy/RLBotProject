# Live integration tests 2 and 3 (2026-11-16): promoted policy, goals, resets, demolition

Policy: `best_verified` = `runs/s1-long` iteration 2,263 (weights `8d55b78c86c7`), installed with
`export --to-bot`, CPU inference, sampled actions; BotCore vs BotCore. `scripts/live_check.py --events
--max-play-seconds 90` with `bot/match_live_events.toml` (state setting enabled): after 8 s of play a demolition is
staged (blue at 2,300 uu/s into a stationary orange car), then a goal into each net is staged; everything else is
ordinary play. Data: `runs/live/20261116-160904/` (test 2) and `runs/live/20261116-161625/` (test 3).

| | Test 2 (before fix) | Test 3 (after fix) |
|---|---|---|
| Score (2 staged goals each) | 4-1 (3 natural goals) | 3-2 (3 natural goals) |
| Kickoffs / environment resets per bot | 6 / **5** | 6 / **6** |
| Demolition of orange; observation flagged `is_demoed` | 1; 44/44 decisions | 1; 45/45 decisions |
| Callback p99 / max; over 6 ms | 1.5-1.6 / 3.0 ms; 0 | 1.6 / 2.5 ms; 0 |
| Fallbacks / policy faults | 0 / 0 | 0 / 0 |
| Applied inputs that are table rows | 100% | 100% |
| Kickoff side; facing the ball (cos) | correct both colours; 0.99 | correct; 0.99 |
| rlgym-rlbot timing warnings per bot | 10 | 0 |
| First kickoff action held jump | 0 of 6 | 0 of 6 |

**Bug found in test 2:** the countdown after the staged goal into the blue net lasted exactly 440 ticks (others
473-841). rlgym-rlbot 0.3.0 resets on the 440th countdown tick, assuming >= 479, so the kickoff was played without a
reset (stale previous action and pre-goal environment state; no fallback). **Fix (BotCoreBot):** reset 60 ticks
earlier (380th countdown tick; cars are frozen during the countdown) and, as a safety net, force a reset if a
kickoff ever starts without one (counted as `missed_kickoff_resets`). Evidence: replaying the test-2 recording
through the real runner offline reproduces the live run exactly without the fix (5 resets, 1,574 decisions) and gives
6 with it; test-1's recording gives 2/2 with all fixes; regression tests for 480-, 440- and 300-tick countdowns;
test 3 live: 6/6 resets at the 380th tick. Test 3 had no short countdown, so the 440-tick case itself is verified via
the replayed real recording, not a new occurrence.

Still not verified live: a competent external opponent (Psyonix bots, humans), naturally occurring demolitions,
overtime with this policy, GPU inference in game.
