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
