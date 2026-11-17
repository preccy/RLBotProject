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

## Observations from the human's perspective: what is verified and what is reconstructed
* Verified: `CompatObsPipeline` re-run on the recording reproduces the bot's own 5,248 logged observations exactly.
  This shows the offline path equals the live path; it is a consistency check of the code, NOT independent proof
  that any feature is physically correct, and it says nothing extra about the human's features.
* Verified for the human: self position equals the packet position (exact outside demolition); the pipeline is
  updated packet by packet and each observation is built before the next packet is read (causal by construction).
* Reconstructed for the human, not independently verified: previous action (the previous mapped table row, a
  modelling choice, not the raw input); pad timers (inferred from is_active edges and fixed cooldowns);
  flip availability / flip window (derived from has_jumped / dodge_timeout); on_ground (AirState heuristic, can
  disagree for a few ticks around take-off and landing).

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

## Supervised-label timing (`scripts/il_feasibility/build_demos.py`)
* Training runs RocketSimEngine with `rlbot_delay=True`: the first tick of each decision still uses the previous
  controls, so an action chosen on the state at tick t controls ticks t+2..t+9. Since `last_input` at packet k
  produced packet k's state, the label for the observation at frame f is the human input of frames f+2..f+9, and
  the previous-action feature is the row for frames f-6..f+1 (the action already committed).
* Labels are causal: a jump state machine decides each block's jump bit from that block's window and earlier ones
  only; a press in the first 4 ticks of a window starts in that block (up to 3 ticks early), a later press in the
  next block (up to 4 ticks late), and a released block is forced between presses. Measured offsets on the recorded
  match: -3..+6 ticks. Every in-grid dodge (60/60) becomes a fresh airborne jump label within one block.
* Leak checks: prev_action never has jump set at a jump onset (0%); no unflagged first-jump label whose observation
  already shows the jump (0%); labels whose press began before the observation are flagged `press_inflight`
  (~40% of first-jump onsets) and should be excluded from an imitation loss (the bot itself can never be in that
  state). Yaw is moved onto roll only at a dodge press (where the game sums yaw+roll); elsewhere jump rows drop yaw.
* Worked example (frames 2035-2067): ground -> first jump (label front-pitched jump) -> hold -> release with air
  control -> dodge press labelled forward-right diagonal (pitch -1, roll +1 from the human's yaw +1) one block before
  the observation shows has_flipped; prev_action equals the previous label throughout.
* Baselines on the one match (4,130 labelled blocks, flagged ones excluded): repeat-previous-action 53.8%
  (state-equivalent rows); iteration 11900's argmax is in the human's equivalence class 9.8% of the time (ground
  driving 17%, air control 2%), mean log P = -7.90, and ~0 probability at the human's first jumps and dodge presses.
