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

**Amendment during converter development, before the checks were run.** Building the converter on two training
matches showed only 102 of 122 dodges keeping a fresh jump label within one block. There were two causes, both
fixed:
1. **The existing jump rule merged fast dodges into one hold.** High-level players hold the first jump about 12 ticks,
   release for one replay frame (4 ticks), then press. The re-press then starts on the next block's first tick, and
   `JumpMachine` reads it as a continued hold. This contradicts its own documented intent, that press-release-press
   patterns survive. The replay conversion uses `HLJumpMachine`: while holding, a block keeps the jump bit only if
   the jump is still held at the block's end. The KBM demos and `build_demos.py` are unchanged.
2. **The dodge-kept count could not judge some dodges.** It counted a dodge as lost when the block after it was
   excluded for another reason. It now judges only dodges inside three contiguous usable blocks.

The thresholds above are unchanged.

**Label-check outcome and chosen scheme:** **FAILED (C2). The pilot stopped here: no training and no
evaluation.** Details are under Results.

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

Run 2026-11-19. Selection `external/hlrlrd/selection.json` (git-ignored data). Checks `scripts/hl_replays/checks.py`,
output `runs/hl-check/checks.json` and `runs/hl-check/passages.txt`.

### Selection
* 1,372 files screened by metadata; 245 were eligible full-length ranked duels.
* 100 selected: 36 maps, dates 2021-08 to 2025-08, 200 distinct players. 15 are held out.
* 250 MB downloaded (the selected files only).

### Initial checks (training-split matches 0-3, both perspectives)
| Check | Result | Pass? |
|---|---|---|
| C1 conversion | 25,899 blocks, 276 finite features, 0.007% of values outside the widened KBM-demo range | pass |
| C2 steer vs next yaw rate | blue 0.910, orange 0.736 (both > 0.5, same sign as the KBM demos); **difference 0.174 >= 0.15** | **FAIL** |
| C2 throttle vs forward acceleration | blue 0.633, orange 0.400 (both > 0.3) | pass |
| C2 causality | corrupting every later frame left 60/60 observations bit-identical | pass |
| C3 grounded steering, 0.5 s re-simulation vs raw controls | nearest: median 14.1 uu / 2.4 deg (p90 47 uu / 7.4 deg). Dithered: 18.5 uu / 3.2 deg. **Nearest chosen** | pass |
| C3 boost | label bit matches block majority 99.8%; matches the replay's boost consumption 99.2% | pass |
| C3 dodges | direction within 45 deg in 369/369 (median 10 deg); fresh jump label kept 342/342 | pass |
| C4 coverage | 94.0% of post-kickoff blocks usable (reproduction excluded 4.0%) | pass |

The converter's own controls reproduce the replay between true updates to a median 15-17 uu and 1.4 deg. About 1%
of segments exceed 150 uu or 20 deg.

### The concrete failure (C2)
Steer vs next-block yaw-rate correlation, per match and car:

| Match | Car | Our observations | Raw replay physics (no obs builder) | Median speed |
|---|---|---|---|---|
| 0 | blue | 0.923 | 0.947 | 1,384 |
| 0 | orange | 0.763 | 0.865 | 1,410 |
| 1 | blue | 0.922 | 0.945 | 1,459 |
| 1 | orange | 0.757 | 0.841 | 1,484 |
| 2 | blue | **0.658** | 0.770 | 1,072 |
| 2 | orange | 0.948 | 0.959 | 1,318 |
| 3 | blue | 0.922 | 0.947 | 1,340 |
| 3 | orange | **0.659** | 0.760 | 1,553 |

* **It is not tied to colour.** In match 2 blue is the low one, so the mirroring of the orange perspective is not
  the cause.
* **It is not replay staleness.** Both cars get true updates on 42-46% of frames, with the same mean snapshot age
  (about 25 ms) in every match.
* **It is present in the raw physics.** One player per match couples steer to yaw rate less than the other (0.95 vs
  0.77-0.87). This is a difference between the two humans' driving. Each "team" in C2 is a different person, so the
  0.15 team-difference threshold measured player style as well as perspective handling.
* **Our observations do lower the correlation.** They are 0.02-0.11 below the raw physics, most for the lower player.
  This is the causal snapshot: the next block's state is the latest true update, up to 0.1 s old and extrapolated.

As instructed, a failed initial check stops the pilot. No arm was trained and CTRL-13300 stays deployed. A
perspective test that separates player from colour would compare each player's correlation with itself in raw
physics, or swap colours by mirroring a whole match. It was not pre-registered, so it has not been substituted
here.

---

## Replacement check (requested after the C2 failure; written and committed before it is run)
The C2 result above stands and stays in the report. C2's team-difference threshold mixed two things: the
converter's handling of the two colours, and the fact that each colour is a different human. The replacement
separates the two.

### How the original pooled C2 scores were computed (for reproducibility)
* Data: training-split matches 0-3, converted as above. Both observation and analog-steer values are independent of
  the stick scheme.
* Pairs: every pair of usable blocks (i, j) of the same car with j starting exactly 8 ticks after i ("usable" means
  after the converter's exclusions, before the loader drops `press_inflight`/`dodge_lost`).
* Kept: `self.on_ground` = 1 in both observations, and local forward speed at i
  (`self.lin_vel_local[0]` x `LIN_VEL_SCALE`) > 300 uu/s.
* x is the block mean of the converter's analog steer over block i. y is `self.ang_vel_local[2]` in block j's
  observation.
* Pooling: all pairs of all blue cars of the 4 matches form one sample (n = 6,128), all orange cars the other
  (n = 6,166). Each pooled sample therefore mixes 4 different players.
* r is Pearson's correlation: blue 0.9099, orange 0.7362, difference 0.1737. The new script recomputes and prints
  these numbers.

### R1 rotation equivalence (same player, same frames)
* **Matches:** training-split matches 4, 5 and 6, never checked before. The 15 held-out matches stay untouched.
* **Transform** of every ReplayFrame before conversion: rotate the world 180 degrees about the vertical axis and swap
  teams.
  * Positions, linear and angular velocities of both cars and the ball: (x, y, z) -> (-x, -y, z).
  * Orientations: R -> diag(-1, -1, 1) R.
  * `team_num` 0 <-> 1.
  * Boost pad timers: permuted to the rotated pad positions. The permutation is computed from BOOST_LOCATIONS; it
    equals reversal.
  * Unchanged: car-relative quantities (flip torque, flags, timers, wheel contacts), controls, update ages. A
    rotation keeps the player's own left/right.
* **Rebuild:** the original and the rotated version both go through the unchanged converter (`hlconv.extract`,
  nearest scheme, including the RocketSim reproduction filter).
* **Comparison**, keyed by (player, block tick):
  * the set of usable blocks;
  * the flags `press_inflight`, `dodge_lost`, `approx_dir` and `dodge`;
  * mapped label rows and `prev_action` rows;
  * equivalence masks;
  * the 276 team-relative observation values;
  * the per-player steering metric (the C2 statistic computed per player).
* **Pass:**
  * every discrete item agrees exactly;
  * every observation value within 1e-4 (absolute, normalised units);
  * each per-player steering metric within 1e-4.
  * A usable-set difference counts as **borderline**, not a failure, only if the reproduction error that set it lies
    within 1 uu / 0.1 degree of the 150 uu / 20 degree threshold on both sides (RocketSim float noise). Such cases
    are counted and reported.
* **On failure:** report the first divergent block (match, player, tick) and the first divergent item or feature.

### R2 absolute steering fidelity per player (same matches 4-6, original orientation)
**Aligned grounded segments:** block start frame a where:
* the player has a true replay update at a and at a + 4 (16 ticks later);
* the car is grounded at both;
* forward speed > 300 uu/s;
* the blocks at a and a + 2 (and the neighbours used for shifting) are usable;
* no jump label.

From the true state at a, RocketSim re-simulates 16 ticks. The player uses (i) the converter's raw controls,
(ii) the two mapped labels, (iii) the labels shifted one block earlier, (iv) one block later. The opponent uses raw
controls throughout. The result is compared with the replay's true state at a + 4.

**Pass, for every player separately:**
* mapped labels: median position error <= 40 uu and median heading error <= 5 degrees against the replay;
* timing: the median heading error with correctly aligned labels is lower than with either shifted version;
* inputs: the per-player C2 statistic from our observations is within 0.15 of the same statistic from raw replay
  physics. Raw physics is the converter's car state at the next block's start frame, angular velocity projected on
  the car's up axis.

Raw-control errors are reported as the converter's own floor.

**If R1 and R2 both pass,** the pilot proceeds exactly as planned above: label scheme nearest (from C3), 1,000
iterations per arm, the predeclared screening and confirmation, fresh-seed evaluations and the existing promotion
rules. If either fails, the pilot stops and the first divergence is reported.

### Replacement check results (2026-11-19, `scripts/hl_replays/rotation_check.py`, `runs/hl-check/replacement.json`)
**Original C2 reproduced exactly:** blue r = 0.9099 (n 6,128), orange r = 0.7362 (n 6,166).

**R1 rotation equivalence: PASS** on training-split matches 4, 5 and 6.
* The original and rotated rebuilds are identical for every block: 4,323, 7,533 and 8,278 blocks respectively.
  * The usable-block sets, label and prev rows, the four flags and the equivalence masks agree exactly.
  * The maximum observation difference is 0.0, bit-identical.
  * There were no borderline cases.
* Per-player steering metrics are identical: 0.9313 / 0.7412, 0.7006 / 0.9117, 0.9396 / 0.6385.
* The converter treats the two perspectives identically.

**R2 per-player steering fidelity: FAIL**, on the timing criterion for one player in every match.

| Match | Player | Segments | Heading error: aligned | One block earlier | One block later | Position (aligned) | Steer metric obs / raw | Pass |
|---|---|---|---|---|---|---|---|---|
| 4 | blue | 44 | 3.46 deg | 4.28 | 3.53 | 21.0 uu | 0.931 / 0.954 | yes |
| 4 | orange | 49 | 2.41 deg | 3.23 | **2.24** | 24.5 uu | 0.741 / 0.837 | **no** |
| 5 | orange | 136 | 2.91 deg | 3.65 | **2.41** | 23.4 uu | 0.701 / 0.807 | **no** |
| 5 | blue | 139 | 2.55 deg | 2.88 | 2.88 | 24.4 uu | 0.912 / 0.938 | yes |
| 6 | blue | 175 | 2.56 deg | 2.78 | 3.12 | 23.9 uu | 0.940 / 0.963 | yes |
| 6 | orange | 155 | 4.33 deg | 4.72 | **3.87** | 29.5 uu | 0.639 / 0.772 | **no** |

**First divergence:** match 4 (selection index 4), the orange player. Labels shifted one block later fit the
replay's true motion better than the aligned labels: median heading error 2.24 vs 2.41 deg. Position and heading
magnitudes, and the obs-vs-raw gap (at most 0.133), passed for every player.

**Cause (diagnosis after the check, no criterion changed).** The correlation between steer and the car's yaw rate
was computed at true-update frames, grounded, above 300 uu/s, with steer taken L frames (4 ticks each) earlier.
The lag with the highest correlation was:

| Match | Recorder (named in the replay title) | Recorder's best lag | Opponent's best lag |
|---|---|---|---|
| 0 | orange | 4 frames | 2 |
| 1 | orange | 4 | 2 |
| 2 | **blue** | 5 | 1 |
| 3 | orange | 6 | 2 |
| 4 | orange | 5 | 2 |
| 5 | orange | 4 | 2 |
| 6 | orange | 6 | 2 |

In every match, the player who recorded the replay has controls leading their car's motion by 4-6 frames
(130-200 ms, varying between matches). The opponent's lead is 1-2 frames, consistent with the label timing. The
only blue recorder (match 2) is the only match where blue was the low side in C2. The asymmetry therefore belongs to
the replay data (the recorder's own inputs against the server's version of their car), not to the converter or the
colour.

**Outcome:** a material, systematic label-timing error affects half of the demonstration data. Per the
pre-registration, **the pilot stops here: no arm is trained.** CTRL-13300 stays best_verified and deployed, and its
references are untouched.

---

## Retained-side pilot (option 1; written and committed before its checks are run)
Requested after the R2 failure. Everything above stays as recorded. Only the non-recorder's side of each replay is
used, for training and for the held-out evaluation. Everything not changed here stays as pre-registered above:
* the converter (`hlconv.py`, nearest scheme, causal inputs);
* the kickoff and other exclusions;
* the arms, budget, candidate selection, evaluation seeds and decision rules.

### Recorder identification (`scripts/hl_replays/recorder.py`)
* The recorder is the player named in the replay title, `<date> <name> Ranked Duel [Win|Loss]`.
* The title is a file name, so characters that are not file-name safe are stripped from `<name>`. Names are therefore
  compared using only their ASCII letters and digits, case-insensitively.
* A match is **ambiguous and skipped entirely** unless all of the following hold:
  * the title parses;
  * the title's name key is non-empty and equals the key of exactly one player;
  * the two players' keys differ;
  * a result suffix (W/L), if present, agrees with the final score.
* Applied to the selection: 84 of 85 training matches and 15 of 15 held-out matches are identified. Training match
  99 is skipped: its title names someone who matches neither player.
* On training matches 0-6 the rule names the same player as the timing diagnosis above (the player with the 4-6
  frame lead) in 7 of 7.
* The retained player is the other player.

### Retained-side checks (all must pass, or stop and show the first failing sequence)
**Matches:** training-split matches 7, 8 and 9, the next unchecked ones in selection order. Their retained players
are orange, blue and orange respectively. The 15 held-out matches stay untouched until the final evaluation.

**N1 R2 fidelity and timing, existing criteria, per retained player.** The aligned grounded segments, the four
re-simulations (raw, mapped, one block earlier, one block later), the opponent on raw controls, and the comparison
with the replay's true state 16 ticks later are all exactly as in R2. Pass requires:
* more than 20 segments;
* mapped labels: median position error <= 40 uu and median heading error <= 5 deg;
* the median heading error with aligned labels is lower than with labels shifted one block earlier and one block
  later;
* the per-player steer metric from our observations is within 0.15 of the raw-physics value.

**N2 steering changes vs sustained turns.** Each N1 segment is classified by the steer component of the four mapped
labels it uses, at ticks t-8, t, t+8 and t+16:
* **change:** the four steer values are not all equal. A shifted label sequence differs from the aligned one only
  in these segments.
* **sustained turn:** all four are equal and non-zero. Shifting changes little here, so these segments can make a
  shifted label look accurate.
* **straight:** all four are zero.

Pass requires:
* **Change segments, per retained player:** median heading error with aligned labels lower than with labels one
  block earlier and with labels one block later. The same must hold for median position error. If a player has
  fewer than 20 change segments, the check uses the change segments pooled over the three retained players, and the
  per-player values are reported.
* **Sustained turns**, pooled over the three players: median position error <= 40 uu and heading error <= 5 deg.
  The shifted errors are reported.

**N3 control lead.** This is the quantity that failed.
* Method: the correlation between the converter's analog steer L frames earlier (L = 0..8, 4 ticks each) and the
  car's yaw rate (angular velocity on the car's up axis). It is computed at the retained car's true-update frames,
  grounded, with forward speed > 300 uu/s, with all frames from k-L to k in the same gameplay period at 4-tick
  spacing.
* Pass: the best L is <= 3 frames for every retained player.
* The recorder's best L is reported alongside. The diagnosis above found 4-6 for recorders and 1-2 for opponents.

**N4 jump/dodge timing and boost, the existing C3 criteria on the retained side:**
* ordinary dodges: mapped direction within 45 deg of the replay's flip direction in >= 90%;
* a fresh jump label within one block of each dodge (judged inside three contiguous usable blocks) in >= 90%;
* boost bit equal to the block majority in >= 95%.

**On failure:**
* stop before training;
* show the first failing sequence: for the first failing player, a segment of the failing type with its tick, its
  analog steer per frame, the mapped labels and the re-simulation errors;
* CTRL-13300 stays deployed.

### Data counts (reported before training)
For the retained side of all identified training and held-out matches, the following are reported:
* matches and players;
* loader-usable blocks (after `press_inflight`/`dodge_lost`);
* usable dodges: dodge-labelled blocks kept by the loader, and has_flipped dodges judged/kept in the grid.

### Changes to the arms (the only ones)
* HL demonstrations: `runs/demos/hl1.npz`, retained side of the 84 identified training matches. The HL arm draws
  1,024 of each 2,048-sample batch from them and 1,024 from the KBM demos, as planned.
* Held-out imitation fit: retained side of the 15 held-out matches (`runs/demos/hl1-holdout.npz`).
* All other arm, screening, confirmation, evaluation and promotion rules are as pre-registered above, including:
  * seeds 21,000-21,039 (screen), 21,100-21,199 (confirm), 21,300-21,399 (head-to-head), 21,500-21,539 (Necto),
    21,700+ (behaviour);
  * regression seeds 0-39.

### Retained-side check results (2026-11-19, `scripts/hl_replays/retained_check.py`, `runs/hl-check/retained.json`)
Recorder identification: 84 of 85 training matches and 15 of 15 held-out matches identified. Training match 99 was
skipped as ambiguous. There is one retained player per match.

| Match | Retained player | Segments (change / sustained / straight) | Heading, all segments: aligned / earlier / later | Change segments: aligned / earlier / later | Position at changes (aligned / earlier / later) | Steer metric obs / raw | Best lag retained (recorder) | Dodges kept, direction | Pass |
|---|---|---|---|---|---|---|---|---|---|
| 7 | orange | 109 (53 / 22 / 34) | 2.21 / 2.45 / 3.13 deg | 2.21 / 2.89 / 4.82 | 24.2 / 25.7 / 26.5 uu | 0.935 / 0.955 | 2 (5) | 45/45, 51/51 within 45 deg | yes |
| 8 | blue | 56 (24 / 18 / 14) | 2.44 / **2.11** / 2.72 deg | 3.14 / **2.53** / 5.26 | 24.1 / 26.6 / 25.9 uu | 0.940 / 0.959 | 2 (4) | 63/63, 67/67 | **no** |
| 9 | orange | 191 (85 / 42 / 64) | 1.94 / 1.95 / 2.58 deg | 2.66 / 2.94 / 4.40 | 21.5 / 24.1 / 23.8 uu | 0.929 / 0.948 | 2 (6) | 51/51, 53/53 | yes |

* All three retained players pass the magnitude criteria (median position 21.8-27.8 uu, heading 1.9-2.4 deg), the
  obs-vs-raw steer metric (gap <= 0.02), N3 (best lag 2 frames, against 4-6 for the recorder in the same match) and
  N4 (boost bit >= 99.7%).
* Pooled N2 passes. Change segments (162): aligned 2.65 deg vs earlier 2.88 and later 4.97. Sustained turns (82):
  23.6 uu, 3.14 deg.
* **Failure:** match 8, retained player (blue), on N1 timing and N2 change timing. Labels shifted one block
  **earlier** give a lower median heading error than the aligned labels: 2.11 vs 2.44 deg over all 56 segments and
  2.53 vs 3.14 over the 24 change segments. The position criterion at changes passes (24.1 uu aligned vs 26.6
  earlier).

**First failing sequence** (match 8, retained blue player, segment at tick 7062):
```
  frame  tick   analog throttle  analog steer   (converter controls, 4 ticks per frame)
   1281  7054   +1.00            -0.43
   1282  7058   +1.00            -0.15
   1283  7062   +1.00            -0.12   <- segment start
   1284  7066   +1.00            -0.33
   1285  7070   +1.00            -0.58
   1286  7074   +1.00            -0.86
   1287  7078   +1.00            -1.00
   1288  7082   +1.00            -1.00
  mapped rows (thr, steer) at t-8, t, t+8, t+16: (+1,+0), (+1,+0), (+1,-1), (+1,-1)
  16-tick re-simulation raw     : position   48.4 uu, heading  3.90 deg
  16-tick re-simulation mapped  : position   48.5 uu, heading  4.34 deg
  16-tick re-simulation earlier : position   57.1 uu, heading  1.56 deg
  16-tick re-simulation later   : position   48.2 uu, heading  6.54 deg
```
* This is a steering onset: the analog steer ramps from -0.12 to -1.00 over four frames. The nearest scheme puts the
  onset into the block at t+8 (mean -0.72).
* The car turned sooner than the converter's own raw controls predict (3.90 deg error even with raw controls).
  Labels one block earlier match the heading best (1.56 deg) but miss position by more (57 vs 48 uu).

**Descriptive follow-up (after the check, no criterion changed).**
* Paired per segment, the aligned labels do not lose systematically to the earlier shift.
* Match 8, change segments, aligned vs earlier:
  * heading: aligned better in 42% of segments, earlier better in 37%, tied in 21%;
  * median paired difference 0.00 deg, 95% bootstrap interval [-1.57, +0.21];
  * position: aligned better in 67%.
* Matches 7 and 9 look the same. The pre-registered criterion compares unpaired medians on a small sample in which
  about half the segments tie, which makes it sensitive.
* This does not change the outcome.

**Data counts (computed only for the checked matches; the other 81 training matches were not converted because the
pilot stopped):**

| Match | Loader-usable blocks | Usable dodge-labelled blocks | Grid dodges kept |
|---|---|---|---|
| 7 | 3,082 | 21 | 45/45 |
| 8 | 3,882 | 40 | 63/63 |
| 9 | 4,007 | 14 | 51/51 |

**Outcome:** a pre-registered retained-side timing criterion failed, so **the pilot stops before training.**
CTRL-13300 stays best_verified and deployed, and its references are untouched. Code prepared for the HL arm, not
used: an optional second demonstration source in the imitation term (`[imitation] mix_dataset`, `mix_matches`,
`mix_batch_size`). It is smoke-tested only; the existing single-source path is unchanged.

### User authorisation to proceed (2026-11-19, recorded before conversion of the remaining matches and before training)
* The pre-registered retained-side timing check **failed** on match 8: labels one block earlier gave a lower median
  heading error for the retained player.
* The user accepted this steering-onset uncertainty for this bounded pilot and authorised proceeding on that basis.
  No additional timing gate and no global label shift are applied.
* **The HL labels are approximate at gradual turn starts.** Nearest rounding of block-mean analog steer can place a
  ramped turn onset up to one block (8 ticks) late.
* The existing non-recorder mapping and exclusions are used unchanged.
* Everything else is as pre-registered: arms, 1,000 iterations each from CTRL-13300, checkpoint screening, fresh-seed
  evaluations and promotion rules. The 15 held-out matches stay reserved for the held-out imitation fit.

### Retained demonstration sets (`scripts/hl_replays/build_hl_demos.py`, `runs/demos/hl1-counts.json`)
| Set | Matches (players) | Retained blue / orange | Blocks after exclusions | Loader-usable blocks | Usable dodge-labelled blocks | Grid dodges (fresh jump label kept) |
|---|---|---|---|---|---|---|
| Training `hl1.npz` | 84 (84) | 47 / 37 | 277,513 | **272,025** | **1,809** | 3,734 (3,734) |
| Held-out `hl1-holdout.npz` | 15 (15) | 8 / 7 | 49,272 | 48,300 | 318 | 690 (690) |

* Training match 99 was skipped as ambiguous.
* Training matches 11 and 60 contribute 0 blocks: their replay frames are not 4 ticks apart, so the existing spacing
  exclusion removes them.
* Four training matches (19, 34, 42, 64) and two held-out matches give fewer than 1,000 blocks. The median is 3,723
  per match.
* "Usable dodge-labelled blocks" counts blocks whose label is a dodge press and that the loader keeps. About half
  of all dodges have their press in the second half of an 8-tick block. The existing rule labels those in the next
  block and flags them `press_inflight`, which the loader drops, exactly as for the KBM demos. "Grid dodges" counts
  every has_flipped dodge inside three contiguous usable blocks.
* For scale: the KBM demos used by both arms (`s1.npz` matches 0-1) contain 8,410 usable blocks. In the HL
  arm, each batch takes half its samples from each source regardless of size.
