# Imitation-learning feasibility test (no training)

2026-11-17. Data: one recorded human-vs-bot match (`runs/live/human-argmax-11900/`, human blue, 18-1).
Question: can obs-v1 and the 90-row / 8-tick action interface faithfully represent ordinary human dodges, kickoffs
and grounded movement? Scripts: `scripts/il_feasibility/` (label_audit, il_map, il_sim, il_extra; output dir as
argument). Nothing was trained; iteration 11900 stays deployed.

## Labels (PlayerInfo.last_input in every recorded packet)
* All 8 controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake) for the human car, every packet.
  Values were only -1/0/+1 in this match (digital input), so analog quantisation was not tested.
* Alignment: a jump / dodge / boost press appears in the same packet as its physical effect (lag 0 in 123 / 96 / 218
  cases): `last_input` is the input applied to produce that packet.
* Gaps: 51,702 of 51,767 packets consecutive; 40 one-tick gaps, 5 of 3-5 ticks (forward-filled); long gaps are goal
  replays.
* The game sums yaw + roll for the dodge side component (pitch -1, yaw -1, roll -1 -> dodge_dir (0.45, -0.89)).

## Observations from the human's perspective
* `CompatObsPipeline` (the live bot's path) rebuilt the bot's own 5,248 recorded observations exactly (max diff 0).
* The same pipeline for the human's player id: 5,202 decisions, finite, self position exact (outside demolition).
* Reconstructed rather than read: previous action (the previous mapped table row, not raw input), pad timers
  (PadTimerTracker), flip availability/window (rlgym-compat + dodge-timeout fix), on_ground (AirState heuristic).

## Action mapping (state-aware: ground uses throttle/steer/jump/boost/handbrake, air uses pitch/yaw/roll/jump/boost)
* Tick agreement, event-aligned mapping (naive sample-and-hold in brackets): throttle 98% (97), steer 94% (90),
  pitch 92% (88), yaw 84% (81), roll 89% (88), jump 97% (96), boost 98% (96), handbrake 98% (98).
* 98 ground jump -> release -> dodge sequences: human jump-to-dodge median 20 ticks (min 13; 10% under the 16-tick
  minimum the interface allows). Event-aligned mapping keeps 98/98 patterns, dodge press shifted -4..+6 ticks;
  naive mapping loses 3/98.
* Dodge direction: 67% of 103 human dodges are one of the table's 8 directions; the rest (yaw and roll held
  together, ~63 deg) map to 45 deg, an 18 deg error. Table jump rows have no yaw: a real, small limitation.

## Re-simulation (RocketSim, initial state from packets, bot car replays its own recorded inputs)
| Segment | Variant | Dodge executed | Dodge tick shift | Angle error | Other |
|---|---|---|---|---|---|
| 19 dodge kickoffs | RAW human inputs | 19/19 | 0 | 0 deg (max 3) | arrival (<220 uu) +2 ticks, speed -11 uu/s; <10 uu error for ~115 ticks |
| | EVENT mapping | 19/19 | -4..+4 | 15 deg | reaches <220 uu of the ball in 7/20; closest approach median 477 uu (recorded 168) |
| | EVENT + exact human dodge stick (not expressible) | 19/19 | -4..+4 | 1 deg | still misses: closest approach median 587 uu |
| 6 clean in-play dodges | RAW | 6/6 | 0 | 0 deg | no >100 uu divergence in ~1.1 s |
| | EVENT mapping | 6/6 | -3..+8 | 14 deg | no >100 uu divergence; speed 1 s later within ~60 uu/s |

* First mapped-vs-raw control difference in kickoffs: steering (18/20, median tick 25). 59% of pre-jump kickoff
  steering pulses are shorter than 8 ticks (21% are 1-3 ticks). Open-loop, this heading drift (not dodge timing or
  direction) is what makes mapped kickoffs miss the ball.
* Raw-replay divergence after ~2 s is mostly contact-driven (11 of 17 after the first ball touch; cars within
  200 uu in 14/20 kickoffs).
* A greedy block-by-block tracking mapper was worse (state restores add noise: raw inputs with per-block restores
  arrive +6 ticks); it neither shows nor rules out a table sequence that reproduces the kickoff path.

## Verdict: partial pass
* Usable now: labels, observations, grounded movement, in-play dodges (with 45-deg-snapped diagonal labels).
* Not demonstrated: faithful kickoffs. Open-loop mapped kickoffs miss the ball; whether a closed-loop policy on this
  interface can reproduce the kickoff outcome is untested.
* Exclude/flag: demolished ticks, goal replays/countdowns, packets after 2+ tick gaps, dodges with yaw+roll held
  together (label direction approximate), trajectory comparisons after car/ball contact.
* Replay files (correction): Rocket League .replay files do contain partial control information (replicated
  throttle/steer/handbrake and boost/jump/dodge activity, dodge direction), but not the full per-tick input, and at
  network rate; the rest needs reconstruction from physics. Not tested here (no parser installed).
