# External benchmark: Necto (evaluation only, 2026-11-18)

**Opponent.** Necto V1, the RLGym community bot (https://github.com/Rolv-Arild/Necto, CC BY-NC-SA 4.0; its README
puts it at "around Diamond level"). It was chosen over Nexto ("approximately GC1") as the opponent our bot is more
likely to contest, and because it needs the least integration: one TorchScript model, a 24-feature-per-entity
observation and plain controls every 8 ticks. It plays frozen, as shipped in RLBot: `necto-model.pt` unchanged,
argmax actions, no hard-coded kickoff. Code lives in `external/Necto` (git-ignored).

**Adapter** (`scripts/necto_bench.py`, no change to `src/`):
* It re-implements Necto's V1 observation and action mapping.
* A subclass of our parser passes Necto's 8 controls through the same RocketSimEngine. Both cars decide every
  8 ticks on the current state, with the same one-tick RLBot input delay.
* Boost pads are re-ordered from RocketSim's order to Necto's by position.
* Timers follow Necto's training builder. The RLBot port's per-tick demo-timer cycling was not reproduced.
* Checks:
  * `selftest`: compared against Necto's own `necto_obs.py`/`agent.py` on 80 real states, both colours. Max
    observation error 0.0 (excluding that demo column) and 0 action mismatches.
  * Mirror check: Necto vs Necto from a symmetric kickoff stays mirror-symmetric to 0.005 uu.
  * Smoke pair: Necto won 12-2 as orange and 8-1 as blue, with about 13 dodges per minute.
  * Necto's learned kickoff is slow. From back-centre it reaches the ball in 3.1 s, driving mostly without boost.
    From a diagonal it dodges in at 2.3 s.

**Design.** 20 fresh colour-swapped seed pairs (offsets 18,000-18,019), the same seeds and settings for both
checkpoints. Our policies sampled their actions, as in the imitation pilot. Matches were 300 s, with overtime to the
next goal capped at 120 s. Results are in `runs/necto-bench/` (`report.json`, per-match files, smoke and self-test).

## Results
| | W-D-L | Goals | Goal diff / match [95% CI] | Kickoff first touch (ours / Necto) | Touch share | Own goals / match |
|---|---|---|---|---|---|---|
| S3C-11900 | 0-0-40 | 59-431 | -9.3 [-10.2, -8.4] | 397 / 44 (90%) | 41.0% | 0.30 |
| S4I-12900 | 0-0-40 | 96-408 | -7.8 [-8.9, -6.7] | 479 / 0 (100%) | 42.8% | 0.58 |

Paired differences, 12900 minus 11900 (20 pairs, bootstrap over pairs):

| Quantity | Difference [95% CI] |
|---|---|
| Match score | 0.0. Uninformative: both lost every match |
| Goal difference per match | **+1.5 [+0.3, +2.8]** |
| Goals for | +0.9 [+0.4, +1.5] |
| Goals against | -0.6 [-1.6, +0.5] |
| Kickoff first-touch share | +10.5 points [+7.1, +13.9] |
| Touch share | +1.8 points [-1.5, +5.4] (inconclusive) |
| Own goals per match | +0.28 [-0.03, +0.58] (inconclusive) |

Kickoff first touch and the defender context below are new counts added for this report. They use the same
touch signal as the evaluator; there was no existing kickoff metric.

**What Necto punishes** (heuristic context at Necto's scoring touch; both checkpoints alike):
* **Not kickoffs.** No goal came within 5 s of a kickoff, and the median goal came 16-17 s after it. We win almost
  every kickoff first touch, and it does not turn into goals for either side.
* **Dodge shots.** Necto was airborne at 89-90% of its scoring touches and dodging at 74-77%. Only about 20% of
  those touches were on a ball above 300 uu, so these are mostly ground and low-ball flip shots, not aerials.
* **Our car out of position and empty.** We were goal-side at only 39% of Necto's scoring touches, about 1,175 uu
  from the ball, with about 5 boost on average. Over the whole match our average boost was 8, against Necto's 23.
* **Rarely lost 50/50s.** We had touched the ball within 2 s before only 8-14% of Necto's scoring touches. Goals
  come from Necto's possession while we are not in the play, not from lost challenges.
* Necto had more touches per minute (35 vs 24-27), dodged 14-15 times a minute (12900: 2.8, 11900: 0.02), and spent
  more time in the air (0.36 vs 0.14-0.21).

**Reading.** Against an opponent from outside our lineage, 12900 is still clearly beaten but loses by about 1.5 fewer
goals per match than 11900, mainly because it scores more. This is a smaller and more certain-to-matter gap than the
0.74 head-to-head against 11900 might suggest. One opponent, RocketSim only, one seed per training arm.

**Recommendation: keep Necto as a benchmark for now, not a training opponent.**
* At 0 wins in 80 matches and about 8 goals conceded per match, games against it would give sparse winning signal,
  and they would cost throughput in the collector.
* The goal margin is a sensitive, cheap, outside-the-lineage metric. It detected a +1.5 goals-per-match
  difference with 20 pairs.
* Revisit a training share once a checkpoint takes a meaningful share of goals against it, for example a goal
  difference above about -3 per match.
