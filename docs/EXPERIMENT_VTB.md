# VTB comparison: removing velocity_to_ball (pre-registered)

Written and committed before either branch starts training. Results are appended below "Results" afterwards;
nothing above it is edited once training begins.

## Question

Does removing the `velocity_to_ball` shaping term at this stage improve full-match results compared with an
otherwise identical, unchanged continuation?

Why this term: in a 960-car-episode rollout of iteration 9,900 in the training scenario mix, it was the largest
weighted term per episode (winner 4.45, loser 4.29, vs goals +/-3), it barely differed between winners and losers, a
scripted pure ball-chaser that lost 94% of episodes earned more of it per second than 9,900 (0.178 vs 0.155), and its
formula charges for any movement away from the ball (rotating back, pad detours, shadowing). That shows the incentive
exists; it does **not** show the term causes the weaknesses seen in the human match. This experiment tests that.

## Branches

* Both start from the full checkpoint of **S2N-iter9900** (`runs/reference/S2N-iter9900-checkpoint`, read-only,
  weights `d4ee480068c2`): weights, both Adam states, return normaliser, counters and RNG. Iteration 9,900 itself is
  untouched and stays `best_verified` and deployed unless the rule below promotes something.
* **CTRL** (`runs/s3-ctrl`, `configs/experiments/vtb-control.toml`): unchanged continuation.
* **EXP** (`runs/s3-novtb`, `configs/experiments/vtb-exp.toml`): `rewards.weights.velocity_to_ball` 0.5 -> 0.0.
  Nothing else differs (the component is still computed and logged with weight 0). Resumed with
  `--allow-config-change`, which permits exactly this difference.
* Budget: **2,000 iterations per branch** (target iteration 11,900; 147,456,000 learner steps each). Run one after the
  other (CTRL first) with Rocket League closed; checkpoints every 50 iterations, all kept. Same seed and RNG state, so
  the branches differ only by the reward (and the chaotic divergence that follows from it). One seed per branch.
* No extension, no other setting changed, whatever the result.

## Evaluations (existing harness: colour-swapped seed pairs, 300 s + up to 120 s overtime)

Action selection: **sampled (stochastic) for every policy in every evaluation**, identical across branches (the
harness default). Sampled vs argmax is not part of this experiment.

Fresh seed offsets (never used before):

1. **Head-to-head (primary): EXP-final vs CTRL-final, 150 pairs, offsets 10,000-10,149.**
2. Fixed opponents, the same for both branches:
   * vs S2N-iter9900 (start, best_verified): 150 pairs, offsets 11,000-11,149;
   * vs S1L-iter2263: 150 pairs, offsets 12,000-12,149;
   * regression checks vs R0 and M2REF: 40 pairs, diagnostic seeds 0-39.
3. **Supporting diagnostics (explanatory only; never a decision input or an objective):**
   * Defence scenarios: 200 starts (seeds 13,000-13,199; `save` and `low_boost_defence` 50/50, the training
     generators) defended by each of S2N-iter9900, CTRL-final and EXP-final against a frozen S2N-iter9900 attacker;
     outcome = goal conceded within 15 s of game time. Same seeds for all three defenders.
   * Goal-side share in full matches: each final vs S2N-iter9900, 20 pairs (seeds 14,000-14,019): fraction of time
     the car is between the ball and its own goal along the field axis. Plus touches/min, airborne, boost, speed and
     own goals from the decision matches.
   * Visible play: HTML top-down replays (`botcore replay-html`) of EXP-final vs CTRL-final on fixed seeds.

## Decision rules (fixed now)

* **EXP better** if the head-to-head 95% CI lower bound > 0.5; **EXP worse** if the upper bound < 0.5; otherwise
  **inconclusive**, reported as such.
* **Promotion** (standing rule, unchanged): a branch final may replace S2N-iter9900 as `best_verified` only with 95% CI
  lower bound > 0.5 vs S2N-iter9900 (150 fresh pairs) and score >= 0.45 in both regression checks.
  * If only one branch qualifies, it is the candidate.
  * If both qualify: EXP if it is better head-to-head, CTRL if EXP is worse, and CTRL (the unchanged configuration) if
    the head-to-head is inconclusive.
* **Deployment:** a promoted checkpoint is installed in `bot/model` only after the established 4-match All-Star series
  (`scripts/allstar_series.py`) shows no runner faults; otherwise S2N-iter9900 stays deployed. If nothing is
  promoted, S2N-iter9900 stays `best_verified` and deployed.

## What this can and cannot establish

It can show whether removing the term changes full-match strength over ~147M steps relative to an identical
unchanged run, for this seed. It cannot separate the effect from seed-to-seed training variance (one run per arm),
cannot say what happens with longer training, measures strength only against this lineage and two weak references,
and does not test aerials, flips, dribbles or play against humans. The critic must re-fit after the reward change, so
any early cost of that is included in EXP's result.

---

## Results

**Training (2026-11-17).** CTRL 14:30-16:12, EXP 16:12-17:52; 2,000 iterations (147,456,000 learner steps) each,
both ending at iteration 11,900 (877,363,200 total steps); ~24k steps/s. Both resumed exactly from S2N-iter9900; EXP's
log shows the single intended config difference (`velocity_to_ball` 0.5 -> 0.0). No NaNs, faults or tracebacks;
log-prob guard max < 1e-5; entropy 3.08 -> 3.04 (CTRL) and 3.08 -> 3.02 (EXP). In EXP the (unweighted, still logged)
`velocity_to_ball` rate fell from 0.021 to 0.013 per step: it drove towards the ball less.

**Evaluations** (every policy sampled stochastically, identical for both branches):

| Comparison | Pairs | W-D-L | Score [95% CI] | Goals | Own goals / match (A) |
|---|---|---|---|---|---|
| **EXP vs CTRL (primary)** | 150 | 121-2-177 | **0.407 [0.353, 0.458]** | 1,227-1,457 | 0.44 |
| CTRL vs S2N-iter9900 | 150 | 199-1-100 | 0.665 [0.613, 0.717] | 1,555-1,180 | 0.31 (9,900: 0.63) |
| EXP vs S2N-iter9900 | 150 | 148-2-150 | 0.497 [0.445, 0.548] | 1,329-1,311 | 0.43 (9,900: 0.59) |
| CTRL vs S1L-iter2263 | 150 | 300-0-0 | 1.000 [0.976, 1.000] | 2,924-501 | - |
| EXP vs S1L-iter2263 | 150 | 290-0-10 | 0.967 [0.943, 0.987] | 2,389-620 | - |
| CTRL / EXP vs R0 | 40 | 80-0-0 / 80-0-0 | 1.000 / 1.000 | 1,613-7 / 1,753-3 | - |
| CTRL / EXP vs M2REF | 40 | 80-0-0 / 80-0-0 | 1.000 / 1.000 | 1,339-7 / 1,356-11 | - |

**Verdict: EXP worse** (head-to-head upper bound 0.458 < 0.5). Removing `velocity_to_ball` did not help over this
budget: EXP did not improve on its own start (0.497), while the unchanged CTRL did (0.665). Own-goal difference in the
head-to-head, EXP - CTRL, per match: +0.03 [-0.07, 0.14] (no difference).

**Supporting diagnostics (not decision inputs):**

* Defence scenarios (200 starts, frozen 9,900 attacker, goal conceded within 15 s): S2N-iter9900 0.38 [0.32, 0.45],
  CTRL 0.32 [0.25, 0.38], EXP 0.34 [0.27, 0.40]. Overlapping intervals; no demonstrated defensive change.
* Goal-side share vs S2N-iter9900 (40 matches each): CTRL 0.70 (opponent 0.68), EXP 0.69 (opponent 0.67). No change.
* Behaviour vs S2N-iter9900 in the decision matches: EXP touched the ball less (27.7 vs 36.6 touches/min for its
  opponent), stayed further from it (mean 1,161 vs 1,039 uu), was slightly more airborne (0.18 vs 0.16) and similar on
  speed and boost; CTRL was close to 9,900 on every metric (38.1 vs 34.2 touches/min, 999 vs 1,024 uu).
  Interpretation (not proven): without the term the policy approached the ball less, without a measurable gain in
  positioning or defence, and lost more challenges for the ball.
* HTML replays EXP (blue) vs CTRL (orange): `runs/viz/vtb/EXP-blue-vs-CTRL-orange-seed{0,1,2}.html`.

**Promotion.** CTRL-iter11900 meets the standing rule (lower bound 0.613 > 0.5 vs S2N-iter9900 on 150 fresh pairs;
regression checks 1.000 / 1.000; `check_promotion`: no problems). EXP does not (0.445). By the pre-registered rule
**CTRL-iter11900 is `best_verified`** (`runs/s3-ctrl` pointers; weights `ae97d768444d`; read-only copies
`runs/reference/S3C-iter11900[-checkpoint]`, sha256-verified).

**What this does not show:** one seed per branch, so this is one training trajectory per arm; a smaller reduction,
a gradual decay or a longer budget were not tested; the result says nothing about which mechanism the term provides
beyond these metrics, and nothing about play against humans.

**All-Star series (CTRL-iter11900, same fixed setup; diagnostic only, not a rank estimate):** 4-0-0, goals 37-7
(4-2, 15-2, 8-2, 10-1); BotCore own goals 0; shots 55 vs 13. Runner: fallbacks 0, missed kickoff resets 0, p99
<= 1.6 ms, no tracebacks (12 rlgym-rlbot missed-step recoveries, as in earlier series). Details:
`runs/allstar/vtb-ctrl-11900/`. For comparison S2N-iter9900 went 4-0-0, 40-4: four matches cannot separate them.

**Deployed:** the runner showed no faults, so CTRL-iter11900 (`ae97d768444d`) is installed in `bot/model`;
S2N-iter9900 (`d4ee480068c2`) is in `bot/model.previous` and `runs/reference/S2N-iter9900`.
