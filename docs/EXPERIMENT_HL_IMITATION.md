# High-level 1v1 replay imitation pilot (pre-registered; one bounded comparison)

Written before any replay is selected or converted and before either arm trains. Results are appended under
"Results". Nothing above it is edited once training starts, except the label-check outcome and the label scheme,
which are recorded in the marked slot before training.

## Question
Do demonstrations from competitive human 1v1 replays broaden CTRL-13300's open-play decisions, especially first
touches, challenges and boost routing, compared with continuing on the user's own KBM demonstrations? Match strength
is measured. Behaviour is described, with no threshold.

## Data source
* The requested name `Rolv-Arild/rl-high-level-dataset` is not accessible: Hugging Face returns 401 and the author's
  public listing has no such dataset.
* The author's public **`Rolv-Arild/hlrlrd-parsed`** is "The High-Level Rocket League Replay Dataset, parsed into
  rlgym-tools ReplayFrame objects". It is the same dataset, already passed through the existing converter
  (rlgym-tools `replay_to_rlgym`, RocketSim interpolation). We use it.
* Each `.npz` holds one replay: `frames` (serialised ReplayFrames, 30 Hz, 4 ticks apart) and `metadata`
  (replay name, map, date, players with platform online_id).
* The converter source, rlgym-tools 2.6.5 (Apache-2.0), is unpacked in git-ignored `external/rlgym-tools-2.6.5` and
  imported only by the replay scripts. The training environment is not changed.
* The original replays' licence is not stated on the page we could read. The data is used locally for training only,
  and no player names appear in any write-up.

## Selection (`scripts/hl_replays/select_matches.py`)
* Candidates are screened by reading only each file's metadata (HTTP range request), in a seeded random order over
  the whole dataset.
* Keep: 2 players, "Ranked Duel" in the replay name, Online, at least 8,500 replay frames (not forfeits), at most 12
  matches per map.
* No player (online_id) appears in two selected matches. Train and held-out matches therefore share no players.
* 100 matches in total. 15 are held out at random (seeded) before any conversion. Only the 100 selected files are
  downloaded.

## Label construction (`scripts/hl_replays/convert.py`)
* **Decision grid.** Contiguous gameplay stretches are cut into 8-tick blocks of two replay frames. The observation
  is taken at the block's first frame (tick t). The label covers ticks t..t+7. Training applies a decision to ticks
  t+1..t+8, so the label is one tick (8 ms) earlier.
* **Policy inputs use only what is known at tick t.**
  * The ball comes from frame t, which the replay sets every frame.
  * Each car comes from its own latest true replay update at or before t (`update_age == 0`), not from the
    converter's interpolated state. Interpolated car states are simulated with inferred controls, so they contain
    future motion.
  * That snapshot is extrapolated to t: position by velocity (plus gravity in the air), rotation by angular
    velocity. Its age must be at most 0.1 s.
  * Pad timers come from frame t.
  * `prev_action` is the previous block's label, as in the existing demos.
  * The observation is the unchanged obs-v1 builder (`botcore.obs.builder.build_obs`, 276 values), built for both
    players. Orange is mirrored by the builder, as in training.
* **Labels may use future motion.** They come from the converter's per-frame controls: analog throttle and steer
  from the replay, pitch/yaw/roll inferred from later angular velocity, jump/boost/handbrake from replay flags. Each
  frame's controls are spread over its 4 ticks, then mapped with the existing rules (`build_demos.py`:
  `JumpMachine`, `block_row`, dodge-direction handling, `approx_dir`/`dodge_lost`/`press_inflight` flags).
* **Analog sticks.** Throttle, steer, pitch, yaw and roll take the block mean. Two schemes are compared in the checks:
  * nearest: round to -1/0/+1;
  * dithered: +-1 with probability |mean|, otherwise 0, seeded per match. The expected value equals the analog value.

  The check below picks one. A dodge's press direction keeps the existing rule.
* **Excluded:**
  * kickoffs: from each period's start until 1.0 s after the ball first leaves the centre spot;
  * non-gameplay gaps, and frames whose spacing is not 4 ticks;
  * either car demolished;
  * car snapshots older than 0.1 s;
  * blocks in a stretch where the converter's own controls do not reproduce the replay: re-simulating the acting car
    in RocketSim from one true update to the next with the converter's controls ends more than 150 uu or more than
    20 degrees from the replay;
  * `press_inflight` and `dodge_lost` blocks (dropped by the existing loader).

## Initial checks on 4 converted training-split matches (all must pass, or stop and report the concrete failure)
* **C1 conversion:** 2 cars; 276 finite features. Under 0.1% of values fall outside the per-feature range of the
  existing demos (`runs/demos/s1.npz`) widened by 10% of that range.
* **C2 perspectives and causality:**
  * For blue and for orange separately, the correlation between the block's analog steer and the car's subsequent
    local yaw-rate change is > 0.5, and the two teams differ by < 0.15.
  * The correlation between throttle and forward acceleration (grounded, below 1,400 uu/s) is > 0.3 for both teams.
  * Corrupting every frame after t leaves the observation at t bit-identical.
* **C3 mapping loss.** From true-update states, each block is re-simulated in RocketSim for 0.5 s (about 7 blocks)
  three ways: raw converter controls, nearest labels, dithered labels. The opponent replays its raw controls.
  * Grounded steering with the chosen scheme: median position divergence from the raw-control simulation <= 100 uu,
    and median heading divergence <= 10 degrees.
  * The boost bit agrees with the replay's block-majority boost in >= 95% of blocks.
  * Ordinary dodges: mapped dodge direction within 45 degrees of the replay's flip direction in >= 90%, and a fresh
    jump label within one block of each dodge in >= 90%.
  * **Scheme choice:** dithered if its median 0.5 s position divergence on grounded blocks is lower than nearest's,
    otherwise nearest.
* **C4 coverage:** at least 50% of post-kickoff gameplay blocks remain usable.
* **C5 inspection** (descriptive): short passages of grounded steering, boost use and ordinary dodges, both
  perspectives, printed with analog controls, mapped rows and the car's response.

**Label-check outcome and chosen scheme:** _(filled in before training)_

## Arms (one seed each, run one after the other, Rocket League closed)
Both arms resume from the same full checkpoint `runs/reference/S5C-iter13300-checkpoint` (read-only) with the same
run seed. Self-play setup, rewards, scenario mixture, architecture, action table and PPO settings are those of the
run that produced 13300 (`runs/s5-ctrl`). Both arms use imitation coef 0.02 with 2,048 demonstration samples per
actor update (6 updates per iteration), for 1,000 iterations = 73,728,000 learner transitions each. Necto is not used
in training.
* **CTRL** (`runs/s6-ctrl`, `configs/experiments/hl-control.toml`): the user's demos `runs/demos/s1.npz`, matches
  0 and 1, exactly as before.
* **HL** (`runs/s6-hl`, `configs/experiments/hl-exp.toml`): each 2,048-sample batch is 1,024 drawn from the same KBM
  demos plus 1,024 from the 85 training replays (`runs/demos/hl1.npz`). Resumed with `--allow-config-change`; the
  only learning-relevant difference is the demonstration mix.

## Candidate selection (identical for both arms, fixed now)
1. **Screen** the checkpoints at 250, 500, 750 and 1,000 iterations (13,550 / 13,800 / 14,050 / 14,300). Each plays
   frozen 13300 on 40 colour-swapped diagnostic pairs, offsets 21,000-21,039 (the same seeds for all 8 checkpoints).
   The candidate is the highest score, then the higher goal difference, then the later checkpoint.
2. **Confirm** each candidate on untouched seeds: 100 pairs vs frozen 13300, offsets 21,100-21,199 (the same seeds
   for both arms).

## Evaluation (our bots sample their actions throughout)
* Head-to-head HL vs CTRL candidate: 100 pairs, offsets 21,300-21,399.
* Against Necto V1 (argmax, as in the benchmark): HL candidate, CTRL candidate and 13300 on the same 40 pairs,
  offsets 21,500-21,539. Reported as paired goal-margin, goals-scored and goals-conceded differences against 13300
  and between the arms.
* Regression checks: 40 pairs each vs R0 and M2REF (seeds 0-39).
* Held-out imitation fit (descriptive): -log P(class) on the 15 held-out replays and on KBM match m03, for 13300 and
  both candidates.
* Behaviour (descriptive): from simulated matches of each candidate vs 13300 on fresh seeds (offsets 21,700+):
  * first touch when the defender closes versus backs off;
  * challenges;
  * zero-boost time share and pad pickups.
  Replay sequences (HTML) cover successes and failures.
* Full-match results are reported first.

## Decision rules (fixed now)
* **Promotion (standing rule):** a candidate replaces 13300 as best_verified only with:
  * 95% CI lower bound > 0.5 on its 100-pair confirmation vs 13300;
  * >= 0.45 in both regression checks.

  If both qualify, HL is chosen only if its head-to-head CI lower bound is > 0.5; otherwise CTRL.
* **External regression flag:** a promoted candidate whose paired goal margin vs 13300 against Necto has a CI upper
  bound < 0 is not deployed. The trade-off is reported.
* Deployment of a promoted, unflagged candidate only after a fault-free 4-match All-Star series. Otherwise 13300
  stays deployed. The 13300 checkpoint and bundle stay preserved (read-only references).
* No extension, no second mix ratio and no further runs.

---

## Results
