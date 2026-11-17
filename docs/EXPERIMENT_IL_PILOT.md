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
