# Necto-mix experiment (pre-registered; one bounded comparison)

Written and committed before the smoke check and before either arm trains. Results are appended under "Results";
nothing above it is edited once training starts.

## Question
Does giving about 15% of training episodes to the frozen external bot Necto V1 (docs/BENCH_NECTO.md) improve
IL-12900? Two separate questions, reported separately:
1. **Against Necto:** does exposure improve the goal margin against Necto itself?
2. **Transfer:** does any gain carry over to other opponents (the control arm, frozen IL-12900, the standing
   regression references)?
One seed per arm and 1,000 iterations: a first, uncertain read. An inconclusive comparison is reported as
inconclusive, not as "no harm".

## Arms
Both resume from the same full checkpoint `runs/reference/S4I-iter12900-checkpoint` (read-only), with the same run
seed. Rewards, imitation term and demonstrations (coef 0.02, `runs/demos/s1.npz` matches 0-1), action selection
(sampled), scenario mixture, architecture, action table and PPO settings are identical.
* CTRL `runs/s5-ctrl`, `configs/experiments/necto-control.toml`: the configuration that produced IL-12900,
  continued. 12 workers x 6 envs = 72 self-play envs = 144 learner slots.
* NECTO `runs/s5-necto`, `configs/experiments/necto-exp.toml`: identical plus `[self_play] necto_fraction = 0.13`.
  That gives 11 workers x 7 envs = 77 envs. 10 envs (13.0%) put the learner against frozen Necto V1, with Necto's
  side alternating orange/blue across envs; the other 67 are normal mirror self-play. Learner slots are
  67 x 2 + 10 x 1 = 144, the same as CTRL.
  * Resumed with `--allow-config-change`; the only learning-relevant diff is `necto_fraction`.
  * Necto runs inside the workers from the game state (`src/botcore/necto.py`, the adapter checked in
    docs/BENCH_NECTO.md). Its car is an EXTERNAL slot and never enters PPO.
  * Necto envs use the same scenario mixture (kickoff 50%, open play, shot, save, ...); Necto plays argmax, as
    shipped.
  * The env share is fixed; the actual episode share depends on episode lengths and is logged each iteration.
  * **Amendment after the smoke check, before any measured training:** the first design (12 of 78 envs, 15.4%)
    gave Necto 19% of episodes, because Necto envs end sooner. Ten Necto envs is the closest layout to the
    requested 15% of episodes that keeps 144 learner slots. The smoke runs' episode shares were 19.0% and 19.3%
    (noisy, 512-847 episodes); the full run's share is reported.
* **Budget:** 1,000 iterations per arm (target 13,900). That is 73,728 learner transitions per iteration in both
  arms, so 73,728,000 learner transitions per arm, the same as each IL-pilot arm. Only learner transitions count.
  The PPO schedule is also identical: 3 batches of 24,576 x 2 epochs = 6 updates per iteration.
* The env seeds differ between arms by construction (77 vs 72 envs).
* The arms run one after the other, with Rocket League closed.

## Smoke check (10 iterations in `runs/necto-smoke`, plus 6 in `runs/necto-smoke2` for the amended layout; from the same checkpoint; discarded, neither arm starts from them)
Pass requires all of:
* no worker error;
* Necto envs finish episodes, and Necto scores most goals in them (the benchmark had 12900 conceding 81%);
* exactly 73,728 learner steps per iteration (the trainer checks learner slots = 144 at start);
* finite losses and imitation term;
* approx_kl < 0.02;
* throughput practical: an arm can finish in under about 1.5 h (>= ~14,000 learner steps/s).

The exclusion of Necto's car from PPO is also covered by `tests/test_necto_training.py`. On failure: stop and
report; no setting is tuned.

## Evaluation (our bots sample their actions throughout; fresh seed offsets)
1. **Against Necto** (`scripts/necto_mix_eval.py`, same match rules and adapter as the benchmark): CTRL-final and
   NECTO-final each play the same 40 colour-swapped seed pairs (offsets 18,100-18,139). IL-12900 on the same seeds
   is a descriptive reference only.
   * Primary: paired difference in goal margin per match, NECTO minus CTRL, with a bootstrap 95% CI over pairs.
   * Also reported: the difference in goals scored and in goals conceded, and W-D-L.
2. **Transfer:**
   * NECTO vs CTRL head-to-head: 100 pairs (offsets 19,000-19,099).
   * Each final vs frozen IL-12900: 100 pairs (offsets 19,200-19,299, same seeds for both arms).
   * Regression checks vs R0 and M2REF: 40 pairs (seeds 0-39).

## Decision rules (fixed now)
* **Against Necto:** exposure **helped** if the CI lower bound of the paired goal-margin difference is > 0;
  **hurt** if the upper bound is < 0; otherwise **inconclusive**.
* **Transfer:** NECTO is **better** than CTRL if the head-to-head CI lower bound is > 0.5, **worse** if the upper
  bound is < 0.5, otherwise **inconclusive**. A gain against Necto is said to carry over only if NECTO is better
  head-to-head. The comparisons with frozen 12900 are reported for each arm, with no further threshold.
* **Promotion (standing rule):** a final may replace IL-12900 as best_verified only with:
  * 95% CI lower bound > 0.5 against IL-12900 on 100 fresh pairs;
  * >= 0.45 in both regression checks.

  If both finals qualify, NECTO is chosen only if it is better head-to-head; otherwise CTRL. A promoted checkpoint
  is installed in `bot/model` only after the fixed 4-match All-Star series shows no runner faults. Otherwise
  IL-12900 stays best_verified and deployed. IL-12900 is preserved either way (read-only reference copies).
* No extension, no second fraction and no further runs.

---

## Results

Run 2026-11-19. Raw outputs: `runs/necto-mix-eval/` (`necto.json`, `decision.json`, one report per comparison).
Our bots sampled their actions in every match; Necto played argmax.

### Smoke check: passed
* `runs/necto-smoke` (12/78 layout, 10 iterations): no worker error. Necto scored 130 of the 161 goals in its envs.
  Exactly 73,728 learner steps per iteration; KL <= 0.0024; about 17,700 learner steps/s.
* `runs/necto-smoke2` (amended 10/77 layout, 6 iterations): same checks passed; episode share 19.3%.
* Both arms restarted from `runs/reference/S4I-iter12900-checkpoint`; nothing from the smoke runs was used.

### Training (1,000 iterations each, 73,728,000 learner transitions each)
| | CTRL | NECTO |
|---|---|---|
| learner steps/s (mean) | 22,600 | 18,400 |
| approx_kl max | 0.0035 | 0.0033 |
| NaN / non-finite losses | none | none |
| entropy start -> end | 2.78 -> 2.71 | 2.77 -> 2.68 |
| imitation loss start -> end | 0.85 -> 0.56 | 0.85 -> 0.56 |

* Resume was continuous: the first 20 CTRL iterations match the last 20 of `runs/s4-il` (entropy 2.777 vs 2.778,
  imitation loss 0.849 vs 0.864, same KL and clip fraction).
* Necto envs in NECTO: 15,801 of 83,022 finished episodes = **19.0%** (the target was 15%; the fixed env share is 13%,
  see the amendment above). In those episodes the learner scored 2,723 goals and Necto 13,051 (17% for the learner).
  There was no trend: 240/1,353 in the first 100 iterations, 200/1,362 in the last 100.

### 1. Against Necto (40 colour-swapped pairs, offsets 18,100-18,139, 80 matches per policy)
| policy | W-D-L | goals for-against | goal margin per match [95% CI] |
|---|---|---|---|
| CTRL-13900 | 0-0-80 | 102-1,057 | -11.94 [-12.64, -11.26] |
| NECTO-13900 | 0-0-80 | 92-980 | -11.10 [-11.78, -10.49] |
| IL-12900 (reference) | 1-0-79 | 209-825 | -7.70 [-8.41, -6.94] |

Paired, NECTO minus CTRL, per match:
* goal margin **+0.84 [-0.04, +1.74]**: **inconclusive** by the pre-registered rule (lower bound not > 0);
* goals scored -0.13 [-0.41, +0.16] (no difference detected);
* goals conceded -0.96 [-1.80, -0.15] (NECTO conceded fewer).

Both finals did worse against Necto than their common start. Paired against IL-12900:
* NECTO: margin -3.40 [-4.44, -2.38]; scored -1.46, conceded +1.94;
* CTRL: margin -4.24 [-5.23, -3.20]; scored -1.34, conceded +2.90.

### 2. Transfer
* NECTO vs CTRL head-to-head (100 pairs, offsets 19,000+): 130-0-70, **0.650 [0.585, 0.720]**; goals 881-671.
  By the pre-registered rule NECTO is **better** than CTRL.
* Against frozen IL-12900 (100 pairs, offsets 19,200+, same seeds for both arms), both arms regressed:
  * CTRL: 21-0-179, **0.105 [0.065, 0.150]**, goal diff -3.26 per match;
  * NECTO: 65-0-135, **0.325 [0.260, 0.385]**, goal diff -1.24 per match.
* Regression references (40 pairs, seeds 0-39): both arms 80-0-0 vs R0 and vs M2REF (pass).

### Promotion
Neither final qualifies: CI lower bounds against IL-12900 are 0.065 (CTRL) and 0.260 (NECTO), not > 0.5.
**IL-12900 stays best_verified and deployed.** `bot/model` and `runs/reference/S4I-iter12900` both still hash
`b1dbba3f4585`. No All-Star series was run, because nothing was promoted.

### Post-hoc diagnostic (not pre-registered; it does not change any verdict)
Both finals lost to their own start, so the export/eval path was checked on intermediate CTRL checkpoints
(20 pairs each vs IL-12900, offsets 19,400+, `runs/necto-mix-eval/posthoc/`):

| CTRL checkpoint | score vs IL-12900 [95% CI] |
|---|---|
| 12950 | 0.350 [0.175, 0.525] |
| 13300 | 0.688 [0.500, 0.850] |
| 13600 | 0.225 [0.100, 0.350] |
| 13900 (100 pairs) | 0.105 [0.065, 0.150] |

The path is not broken: 13300 beat IL-12900. The continued configuration swings strongly between checkpoints, and
its final snapshot landed in a weak phase. Behaviour also moved away from IL-12900 in the Necto matches:
* dodges per minute: about 3.0 for IL-12900, 8.5 for CTRL, 9.4 for NECTO;
* touches per minute: 26.9, 14.4 and 18.9 respectively.

The imitation loss kept falling throughout (0.85 -> 0.56).

### Conclusions
1. **Against Necto: inconclusive.** NECTO's margin was +0.84 goals per match better than CTRL's, but the CI touches 0.
   Any difference came from conceding fewer goals (-0.96 [-1.80, -0.15]), not from scoring more. Neither arm got
   closer to Necto than IL-12900 already was; both got further away.
2. **Transfer: by the pre-registered rule, NECTO is better than CTRL** (0.650 [0.585, 0.720]). That is not an
   improvement over the bot we already have. NECTO lost to IL-12900 (0.325), only less badly than CTRL (0.105).
   Given the checkpoint-to-checkpoint swings above, one final snapshot per arm cannot separate a Necto effect from
   where each run happened to stop. This is one seed per arm.
3. Continuing the IL-12900 configuration for 1,000 more iterations did not hold its strength at the final snapshot.
   This is a finding about the base configuration, independent of Necto.

As pre-registered, there is no extension, no second fraction and no further run.

---

## Follow-up: fresh confirmation of CTRL-13300 (plan written and committed before any match is played)
CTRL-13300 was the best of three checkpoints in the post-hoc scan: 0.688 against IL-12900 on 20 pairs, offsets
19,400-19,419. Picking the best of three on 20 pairs biases that number upward, so this follow-up repeats the test
on fresh seeds only. It is evaluation only: no training.

**Identity.** `runs/s5-ctrl/checkpoints/iter_013300` (COMPLETE marker; policy_version 13300; 980,582,400 learner
timesteps, which is the start plus 400 x 73,728). Export `runs/s5-ctrl/exports/iter_013300`, weights
`0e3fd5323e0a`, identical to a fresh re-export from the checkpoint. IL-12900 is `runs/reference/S4I-iter12900`,
weights `b1dbba3f4585`.

**Seeds.** I scanned all 2,344 result files under `runs/`. The highest evaluation offset ever used is 19,419. The
offsets below appear nowhere.
* Head-to-head: CTRL-13300 vs IL-12900, 100 colour-swapped pairs, offsets 20,000-20,099. Both sides sample their
  actions. Report W-D-L, score and the pair-bootstrap 95% CI.
* Against Necto: CTRL-13300 and IL-12900 each play the same 40 colour-swapped pairs, offsets 20,200-20,239, with
  the same adapter and rules as before. Report the paired difference (13300 minus 12900) in goal margin, goals
  scored and goals conceded, with a bootstrap 95% CI over pairs.

**Decision rules (fixed now).**
1. 13300 **qualifies** only if the head-to-head CI lower bound is > 0.5. If so, the standing regression checks run:
   40 pairs against R0 and against M2REF (seeds 0-39); each score must be >= 0.45.
2. **Clear regression against Necto** means the paired goal-margin CI upper bound is < 0. If 13300 qualifies but
   regresses clearly against Necto, IL-12900 stays deployed and the trade-off is reported.
3. Otherwise, a qualifying 13300 becomes best_verified and is installed only after a fault-free 4-match All-Star
   series. IL-12900 moves to `bot/model.previous` and keeps its read-only reference copy.
4. If 13300 does not qualify, IL-12900 stays best_verified and deployed.

This is the only candidate. The experiment ends after this comparison.

### Follow-up results (2026-11-19; `runs/necto-mix-eval/confirm-13300/`, `scripts/confirm_13300.py`)
Identity was checked by the script before play: CTRL-13300 `0e3fd5323e0a`, IL-12900 `b1dbba3f4585`. All of our
policies sampled their actions.

**Head-to-head, CTRL-13300 vs IL-12900** (100 fresh colour-swapped pairs, offsets 20,000-20,099):
* 122-1-77, score **0.6125 [0.540, 0.680]** (pair bootstrap). Qualifies: the lower bound is > 0.5.
* Goals 902-731; goal difference +0.86 per match [+0.47, +1.26].
* Own goals per match: 0.475 for 13300, 0.48 for 12900.

**Against Necto** (the same 40 fresh pairs for both, offsets 20,200-20,239):
| policy | W-D-L | goals for-against | goal margin per match [95% CI] |
|---|---|---|---|
| CTRL-13300 | 0-0-80 | 183-769 | -7.33 [-7.93, -6.69] |
| IL-12900 | 0-0-80 | 191-804 | -7.66 [-8.25, -7.09] |

Paired, 13300 minus 12900, per match:
* goal margin **+0.34 [-0.50, +1.21]**;
* goals scored -0.10 [-0.50, +0.31];
* goals conceded -0.44 [-1.08, +0.18].

There is no clear regression (the upper bound is not < 0), and no gain is shown either. The head-to-head gain is not
shown to carry over to Necto. Dodges per minute against Necto: 5.1 for 13300, 2.7 for 12900.

**Standing promotion checks:**
* `check_promotion`: no problems.
* Regression checks: R0 1.000 and M2REF 1.000 (40 pairs each, seeds 0-39).
* All-Star series (`runs/allstar/ctrl-13300/`, 4 offline matches, not a rank estimate): **4-0-0 (40-2)**.
  * BotCore own goals 0.
  * Runner clean: 0 fallback ticks, 0 policy faults, 0 unexpected or missed kickoff resets.
  * Callback p99 at most 1.6 ms; no callback over 6 ms.

**Outcome.**
* CTRL-13300 is best_verified (`runs/s5-ctrl/pointers.json`) and is installed in `bot/model` (`0e3fd5323e0a`).
* IL-12900 is in `bot/model.previous` and keeps its read-only copies, `runs/reference/S4I-iter12900[-checkpoint]`.
* New read-only copies: `runs/reference/S5C-iter13300[-checkpoint]`.

Caveats:
* One training seed.
* 13300 was found by a 3-checkpoint scan, and only its confirmation used fresh seeds.
* The strength of this configuration oscillates between checkpoints: 13900 scored 0.105 against the same opponent.

The experiment ends here.
