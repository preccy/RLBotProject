# Imitation pilot: plan (DRAFT; finalised and committed before any training, after the demonstrations exist)

## Question
Does adding a modest imitation loss on recorded human play to the unchanged continuation of iteration 11900 make
useful dodges and better movement emerge without reducing match strength? Kickoffs are excluded. One demonstrator,
one seed per arm and a short budget can show feasibility and absence of harm; they cannot establish improvement.

## Data
* Session `s1`: 3 human-vs-BotCore matches recorded with `scripts/record_demo.ps1` (bot = deployed 11900, sampled).
  A small feasibility pilot: with two training matches the conclusions are correspondingly weaker.
* Before training, `scripts/il_feasibility/count_usable.py` reports per match the usable movement blocks,
  recoveries and complete jump-release-dodge sequences; a specific shortage is reported before asking for more
  matches. Reference from the earlier argmax-opponent match: 4,130 usable blocks, 2,126 grounded movement blocks,
  17 recoveries, 25 jump-release-dodge sequences (6 if the first-jump label must also be usable; 2 with approximate
  direction); 23 dodge presses do not start from a ground jump within 1.3 s.
* `scripts/il_feasibility/build_demos.py`: causal labels (see docs/IL_FEASIBILITY.md), kickoff + first 1 s,
  non-Active phases, packet gaps and demolitions excluded; `press_inflight` and `dodge_lost` blocks dropped from the
  loss; `approx_dir` dodge labels kept but reported separately.
* Split by whole matches: m01-m02 train, m03 held-out validation; the earlier argmax-opponent match
  (`runs/demos/pilot0-argmax-match.npz`) is a second held-out set. No frame-level random split.

## Recorded data (2026-11-17, `runs/demos/s1.npz`, built with build_demos.py)
| Match | Score (human-bot) | Usable blocks | Ground movement blocks | Jump-release-dodge sequences | Recoveries | Approx-direction dodge blocks |
|---|---|---|---|---|---|---|
| m01 (train) | 13-1 | 4,223 | 2,063 | 30 | 11 | 30 |
| m02 (train) | 13-2 | 4,187 | 2,197 | 39 | 8 | 24 |
| m03 (held out) | 20-1 | 4,252 | 2,079 | 33 | 11 | 12 |

All three runs fault-free (0 fallbacks, 0 policy faults, 0 missed kickoff resets). Every in-grid dodge kept its
press-release-press labels except one in m03 (3 blocks flagged `dodge_lost`). Shortage: recoveries (19 train / 11
held out) are too few to assess; recovery behaviour is reported descriptively only and is not a pilot question.

## Arms (both resume from runs/reference/S3C-iter11900-checkpoint, same seed, rewards/architecture/action table unchanged)
* CTRL: `vtb-control` configuration, 1,000 more iterations (73.7M learner steps, ~55 min). The budget is not
  increased to compensate for the smaller dataset.
* IL: identical plus an imitation term on the actor: L = L_PPO + lambda * mean(-log sum_{a in class(s, label)} pi(a|s))
  over a demonstration minibatch each update; `class` = state-equivalent table rows (scripts/il_feasibility/equiv.py),
  so the loss does not fight the policy's choice among rows that do the same thing. lambda is fixed in advance
  (proposed 0.02: the initial imitation term ~0.16 is the same order as the PPO objective, not dominant), no sweep.
* Needed before running (not built yet): ~60 lines in the trainer (load the npz, sample a demo minibatch, add the
  term, log train / held-out imitation metrics every 50 iterations), an `[imitation]` config block, tests.
* Preflight (not counted): 20 iterations of IL; abort if PPO approx_kl or clip fraction leave their usual range
  (kl ~0.002, clip ~0.02) or held-out imitation loss does not fall.

## Evaluation (fresh seed offsets from 17,000; every policy sampled, as in all earlier decisions)
1. IL vs CTRL head-to-head, 100 pairs: IL is "worse" if the 95% CI upper bound < 0.5. No-harm requires not worse.
2. IL and CTRL each vs S3C-iter11900, 100 pairs; regression checks vs R0 / M2REF (40 pairs).
3. Held-out human matches: equivalence-class accuracy, mean log P(class), probability of a jump row at the human's
   dodge presses (baseline 11900: 9.8%, -7.90, ~0).
4. Dodge use in the evaluation matches (supporting, not an objective): dodges per car-minute (now 0) and the share
   followed within 0.5 s by higher speed towards the ball or a touch; own goals per match.
5. If IL is not worse: one recorded human match against it for qualitative feedback.

## Decisions
* Promotion only by the standing rule vs best_verified (11900): >= 100 fresh pairs, CI lower bound > 0.5,
  regression checks >= 0.45. Otherwise 11900 stays best_verified and deployed.
* Dodges emerging is reported, never optimised or used alone as a success criterion.
