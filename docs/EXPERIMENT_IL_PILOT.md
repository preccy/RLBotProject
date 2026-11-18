# Imitation pilot (pre-registered; small feasibility pilot)

Written and committed before the preflight and before either arm trains. Results are appended under "Results";
nothing above it is edited once training starts.

## Question
Does adding a modest imitation loss on recorded human play to the unchanged continuation of iteration 11900 make
complete, useful dodges appear without a measurable loss of match strength? Kickoffs are excluded from the data.
Two training matches from one demonstrator, one seed per arm and 1,000 iterations can show feasibility and give a
first, uncertain read on harm; they cannot establish improvement. An inconclusive match comparison is reported as
inconclusive, not as "no harm". Recovery behaviour is reported descriptively only.

## Recorded data (2026-11-17, `runs/demos/s1.npz`, built with build_demos.py)
| Match | Score (human-bot) | Usable blocks | Ground movement blocks | Jump-release-dodge sequences | Recoveries | Approx-direction dodge blocks |
|---|---|---|---|---|---|---|
| m01 (train, index 0) | 13-1 | 4,223 | 2,063 | 30 | 11 | 30 |
| m02 (train, index 1) | 13-2 | 4,187 | 2,197 | 39 | 8 | 24 |
| m03 (held out, index 2) | 20-1 | 4,252 | 2,079 | 33 | 11 | 12 |

All three runs fault-free (0 fallbacks, 0 policy faults, 0 missed kickoff resets). Every in-grid dodge kept its
press-release-press labels except one in m03 (3 blocks flagged `dodge_lost`). Shortage: recoveries (19 train / 11
held out) are too few to assess. Labels: causal, frames f+2..f+9 for the observation at f (docs/IL_FEASIBILITY.md);
`press_inflight` and `dodge_lost` blocks are dropped from the loss; `approx_dir` dodges kept.
m03 is used for nothing but the final held-out report (no tuning, no preflight decision). The earlier argmax-opponent
match (`runs/demos/pilot0-argmax-match.npz`) informed the label design, so it is a secondary diagnostic only.

## Arms
Both resume from the same original checkpoint `runs/reference/S3C-iter11900-checkpoint` (read-only), same seed;
rewards, architecture, observation contract, action table and PPO settings unchanged. Run one after the other,
Rocket League closed.
* CTRL `runs/s4-ctrl`, `configs/experiments/il-control.toml`: 1,000 iterations (target 12,900; 73,728,000 learner
  steps). No budget increase to compensate for the small dataset.
* IL `runs/s4-il`, `configs/experiments/il-exp.toml`: identical plus `[imitation] coef = 0.02, matches = [0, 1],
  batch_size = 2048` (resumed with `--allow-config-change`, which permits exactly that block).
* Imitation term (`src/botcore/imitation.py`): loss = mean over a demonstration minibatch of
  -log sum_{a in class} pi(a|s), with `class` the table rows equivalent to the human label in that state. Added via a
  wrapper on the actor optimizer's step: after the library has accumulated and clipped the PPO gradients, the wrapper
  adds 0.02 x grad(imitation loss) on one minibatch of 2,048 blocks and re-clips the actor gradient to max_grad_norm
  0.5 (2 updates per iteration). Logged every iteration: imitation loss, its weighted value, PPO and imitation actor
  gradient norms, their cosine, combined norm before clipping.

## Preflight (20 iterations in `runs/il-preflight`, discarded; neither arm starts from it)
Pass requires all of: no NaN/inf in the imitation loss or gradients (enforced in code); imitation loss on a fixed
4,096-block sample of the TRAINING matches lower after 20 iterations than before; mean imitation gradient norm >= 10%
of the PPO actor gradient norm (measurable contribution); PPO approx_kl < 0.01 and clip fraction < 0.1 on every
iteration. On failure: stop and report; no setting is tuned.

## Evaluation (fresh seed offsets from 17,000; every policy samples its actions, as in all earlier decisions)
1. **Held-out probability of human jump/dodge actions** (m03 primary, pilot0 secondary) for 11900, CTRL-final and
   IL-final: mean -log P(class) over all usable blocks; argmax-in-class accuracy; mean P(class) at human first jumps
   and at human dodge presses; P(any jump row) at human dodge presses.
2. **Complete dodges in simulated play**: CTRL-final and IL-final each vs S3C-iter11900, 20 colour-swapped pairs
   (offsets 17,400-17,419, 40 matches of 300 s): dodges (has_flipped rising edges) and jumps per car-minute.
3. **Usefulness of those dodges** (supporting, never an objective): per dodge, speed change 0.5 s after the press,
   change in velocity towards the ball, share with a ball touch within 1.0 s, share back on four wheels within 1.0 s;
   own goals per match. Recoveries (landing after >= 0.5 s airborne) and the share reaching 1,000 uu/s within 1 s:
   descriptive only.
4. **Match strength**: IL vs CTRL head-to-head, 100 pairs (offsets 17,000-17,099); IL and CTRL each vs S3C-iter11900,
   100 pairs (offsets 17,200-17,299, same seeds for both); regression checks vs R0 and M2REF, 40 pairs (seeds 0-39).

## Decision rules (fixed now)
* Head-to-head: IL **worse** if the 95% CI upper bound < 0.5; **better** if the lower bound > 0.5; otherwise
  **inconclusive** (stated as such, with the interval).
* Imitation effect on held-out human actions: reported with the change from 11900 and from CTRL; no threshold.
* Dodges "emerge" only if IL-final executes complete dodges in simulated matches at a rate clearly above CTRL
  (CTRL and 11900 are at ~0); their usefulness is described, not scored.
* Promotion (standing rule): a final may replace S3C-iter11900 as best_verified only with 95% CI lower bound > 0.5
  vs S3C-iter11900 (100 fresh pairs) and >= 0.45 in both regression checks. If both qualify: IL if better
  head-to-head, otherwise CTRL. A promoted checkpoint is installed in bot/model only after the fixed 4-match All-Star
  series shows no runner faults. Otherwise S3C-iter11900 stays best_verified and deployed.
* No extension, no further demonstrations and no second coefficient if the result is inconclusive.

---

## Results
Run 2026-11-17. Preflight `runs/il-eval/preflight.json`; arms `runs/s4-ctrl`, `runs/s4-il`; evaluations
`runs/il-eval/{heldout,dodges,decision}.json` and the per-comparison reports beside them. Every policy sampled its
actions.

**Preflight (passed; run discarded).** Imitation loss on the fixed training sample 8.68 -> 3.24 in 20 iterations;
imitation/PPO actor gradient ratio 0.224; max approx_kl 0.0053, max clip fraction 0.056; everything finite; cosine
between the two gradients -0.136. Entropy rose 3.04 -> 3.26 (noted, not a criterion).

**Training (both 1,000 iterations from the same S3C-iter11900 checkpoint, no faults, no NaN, logp recompute checks
clean).**
| | approx_kl max | clip max | entropy first -> last-100 mean | imitation loss (train minibatch) first -> last | grad norm il / ppo, first -> last |
|---|---|---|---|---|---|
| CTRL | 0.0031 | 0.029 | 3.05 -> 3.07 | - | - |
| IL | 0.0051 | 0.054 | 3.08 -> 2.76 | 8.25 -> 0.86 | 0.170 / 0.340 -> 0.016 / 0.294 |
The imitation share of the actor gradient was about 50% at the start and about 5% by the end, because the loss on
the two training matches fell to 0.86 (largely fitted). Cosine with the PPO gradient: -0.22 at the start, then
around 0 to +0.1.

### 1. Held-out probability of human jump/dodge actions
| Policy | m03 -log P(class) | m03 argmax in class | m03 P(class) at first jumps (52) | m03 P(class) at dodge presses (44) | m03 P(any jump row) at dodge presses | pilot0 -log P / P at dodge presses | training matches -log P / P at dodge presses |
|---|---|---|---|---|---|---|---|
| S3C-11900 | 8.88 | 8.9% | 0.000 | 0.000 | 0.000 | 8.56 / 0.000 | 8.70 / 0.000 |
| CTRL-12900 | 9.54 | 9.4% | 0.000 | 0.000 | 0.000 | 9.30 / 0.000 | 9.50 / 0.000 |
| IL-12900 | 3.43 | 28.0% | 0.033 | 0.021 | 0.070 | 3.59 / 0.091 | 0.86 / 0.535 |
On held-out play the imitation arm moved from essentially zero to small but non-zero probability at the human's
jump and dodge moments. The gap between training (0.86) and held-out (3.43) loss shows heavy fitting to the two
training matches; the held-out gain is real but much smaller than the training numbers.

### 2. Complete dodges in simulated play (each policy vs S3C-11900, 20 colour-swapped pairs, 40 x 300 s)
| Policy | dodges | per car-minute [95% CI] | matches with any dodge | jumps per car-minute |
|---|---|---|---|---|
| S3C-11900 (vs itself) | 5 | 0.025 [0.005, 0.045] | 5 / 40 | 7.36 |
| CTRL-12900 | 1 | 0.005 [0.000, 0.015] | 1 / 40 | 6.27 |
| IL-12900 | 494 | 2.47 [2.21, 2.73] | 40 / 40 | 8.19 |
Dodges emerged under the pre-registered definition: about 100x the control's rate, in every match. For scale only,
the human performed about 33 dodge sequences in each ~5-minute demonstration match.

### 3. Usefulness of those dodges (descriptive)
| Policy | speed change 0.5 s after press (uu/s) | change in velocity towards ball | touch within 1 s | four wheels within 1 s |
|---|---|---|---|---|
| S3C-11900 (5 dodges) | -412 | -857 | 0% | 20% |
| CTRL-12900 (1 dodge) | +460 | +114 | 0% | 100% |
| IL-12900 (494 dodges) | +293 | +23 | 22.1% | 7.5% |
IL's dodges add speed on average (+293 uu/s) and about a fifth are followed by a ball touch within 1 s, but on
average they do not increase speed towards the ball, and only 7.5% are back on four wheels within 1 s. The
landing/recovery side of the dodge is weak. Own goals per match: IL 0.525 vs 11900, CTRL 0.605 vs 11900.
Recoveries (descriptive only): landings after >= 0.5 s airborne reaching 1,000 uu/s within 1 s: 11900 57.5%, CTRL 59.5%,
IL 68.2%.

### 4. Match strength (sampled actions)
| Comparison | pairs | W-D-L | score [95% CI] | goals | verdict |
|---|---|---|---|---|---|
| IL vs CTRL | 100 | 159-0-41 | 0.795 [0.735, 0.850] | 1,065-641 | IL **better** |
| IL vs S3C-11900 | 100 | 148-0-52 | 0.740 [0.680, 0.800] | 1,054-690 | above 0.5 |
| CTRL vs S3C-11900 | 100 | 82-1-117 | 0.412 [0.347, 0.477] | 924-1,017 | below 0.5 |
| IL / CTRL vs R0 | 40 | 80-0-0 / 80-0-0 | 1.000 / 1.000 | | regression check passed |
| IL / CTRL vs M2REF | 40 | 80-0-0 / 80-0-0 | 1.000 / 1.000 | | regression check passed |
The unchanged control continuation scored **below** its own starting point against frozen 11900 (upper bound
0.477). Part of IL's margin over CTRL is therefore CTRL's regression; IL's margin over 11900 itself is direct.

**Caveats.** One seed per arm (one training trajectory each), one demonstrator, two training matches. All
opponents are from the same lineage and almost never dodge, so IL's strength gain may partly come from play these
opponents have not seen; this says nothing about strength against humans or other bots. RocketSim only.

**Promotion.** IL-iter12900 meets the standing rule: lower bound 0.680 > 0.5 vs S3C-iter11900 on 100 fresh pairs;
regression checks 1.000 / 1.000 (40 pairs each, as pre-registered and as in the VTB promotion); `check_promotion`
on the report vs best_verified returned no problems. CTRL does not qualify. **IL-iter12900 is `best_verified`**
(`runs/s4-il` pointers; weights `b1dbba3f4585`; read-only copies `runs/reference/S4I-iter12900[-checkpoint]`,
hash-verified).

**All-Star series (IL-iter12900, same fixed setup; diagnostic only, not a rank estimate):** 4-0-0, goals 38-1
(10-0, 8-1, 9-0, 11-0); BotCore own goals 0; shots 56 vs 5. Runner: fallbacks 0, policy faults 0, missed kickoff
resets 0, p99 <= 1.7 ms, no tracebacks (5 rlgym-rlbot missed-step recoveries; Rocket League's "Error connecting"
lines after RLBotServer shut down, as in earlier series). Details: `runs/allstar/il-12900/`. For comparison
CTRL-iter11900 went 4-0-0, 37-7: four matches cannot separate them. Dodge counts in the real game were not measured.

**Deployed:** the runner showed no faults, so IL-iter12900 (`b1dbba3f4585`) is installed in `bot/model`;
S3C-iter11900 (`ae97d768444d`) is in `bot/model.previous` and `runs/reference/S3C-iter11900`. The deployed bot still
samples its actions (argmax remains opt-in via `BOTCORE_DETERMINISTIC=1`).

**Verdict.** In this one-seed pilot, a small imitation term on two human matches made the policy perform complete
dodges in simulated play (2.47 per car-minute vs ~0), raised held-out probability of the human's jump/dodge actions
from ~0 to a few percent, and gave a stronger policy against frozen 11900 and the control, which itself regressed.
The dodges add speed but rarely land quickly or aim at the ball. Not tested against humans; not extended.
