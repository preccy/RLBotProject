# Design: actions, timing, rewards, curriculum, self-play, evaluation, deployment

Companion to [DECISIONS.md](DECISIONS.md) (why this stack) and [OBSERVATION_SPEC.md](OBSERVATION_SPEC.md)
(the input contract). Numbers measured on this PC are in [../STATUS.md](../STATUS.md).

## 1. Timing

* Physics: 120 ticks/s in RocketSim and in Rocket League (RLBot's tick rate is min(FPS, 120)).
* **Action repeat 8**: one decision per 8 ticks = 15 decisions/s (66.7 ms). Configurable (`[env] action_repeat`),
  recorded in every checkpoint and export manifest, and the RLBot runner always uses the exported value.
* **Input delay**: `RocketSimEngine(rlbot_delay=True)` steps physics before applying new controls, so each
  decision takes effect one tick late (and the first tick after a reset is a no-op), as in RLBot. The runner uses
  rlgym-rlbot's matching `StepOffset(-1, ACTION_END)`: the next observation is built from the state one tick before
  the current action ends, giving the bot a tick to compute and submit it.
* **Kickoff**: rlgym-rlbot builds the first observation from the 440th countdown tick and repeats the first
  controller state until the kickoff starts. Caveat from its README: a policy that wants to jump on the first
  kickoff tick will appear to hold jump instead.
* **Changing action repeat is a different learning problem**, not a free upgrade: gamma is per decision (0.99 at
  15 Hz is a ~6.7 s horizon; at 30 Hz the same gamma halves it), rollout lengths change meaning, PPO's credit
  assignment changes, and a policy's jump timing is learned at its own granularity. `per_second` rewards are scaled
  by elapsed time so their totals stay comparable; event rewards are per event. Treat a repeat change as a new run
  (or a fine-tune with re-evaluation), never as a checkpoint-compatible config edit (resume refuses it without
  `--allow-config-change`).

## 2. Action space

RLGym's 90-entry lookup table (pinned by SHA-256 `86baa15c...`, id `rlgym-lookup-90-v1`). Controls:
throttle, steer, pitch, yaw, roll, jump, boost, handbrake.

* 24 ground actions: throttle {-1, 0, 1} x steer {-1, 0, 1} x handbrake, plus boost (only with throttle 1).
* 66 air/jump actions: pitch, yaw, roll in {-1, 0, 1}, jump, boost; steer = yaw; throttle = boost; jump only with
  yaw 0; handbrake on jump+direction rows (wavedash/flip-cancel setups).

What it can express (tested where marked): driving, powerslide, boosting, single jump, **double jump** (press,
release, press: tested), **dodges** in 8 directions (front flip tested), half-flip / flip-cancel sequences, air
roll in both directions with pitch/yaw, aerial boosting, wavedash inputs.

Known omissions (from `audit_table`, tested): only full-deflection analog values (no feathered steering/throttle
or partial air control); no jump with yaw/steer (no yaw-direction dodges, no turning on the jump tick); boost is
always paired with full throttle on the ground and air throttle always equals boost; no braking while boosting.
Inputs are held for 8 ticks, so precise tick-level mechanics (fast-aerial double-jump timing, flip-reset
timing, ceiling shots needing sub-66 ms inputs) are coarser than a human's. A finite table cannot express every
mechanic; adding rows or a factored action space is a new action contract (new id, retraining).

Jump semantics: the jump button is held for the whole decision. The previous action is part of the observation,
so the policy knows whether jump is currently held (a second jump needs a release decision in between). Holding
jump for up to 3 decisions (0.2 s) gives the higher jump, as in the game.

**No action masking.** Every row is legal in every state (some are merely ineffective, e.g. jump while airborne
without a flip). Masking ineffective actions would need flip/boost state that is reconstructed at deployment and
would remove legitimate options (e.g. stalls of air-roll inputs). If masks are added, `ActorCriticPolicy` must
derive them from the observation so sampling, log-probability and entropy all use the same mask.

**Review (Milestone 2): keep the contract.** Evidence: (1) deployment: in live test 1, 100% of 46,033 in-play
ticks of applied inputs were exact table rows, with zero fallbacks, so the table creates no deployment problem;
(2) learning: 18 of 90 rows press jump, so a near-random policy jumps on ~20% of decisions and is airborne
~75-80% of the time (measured identically in sim and live). That makes early exploration air-heavy, and is the
main visible cost of this table; the M2 evaluation diagnostics (airborne fraction vs training) show whether
training moves away from it (see EXPERIMENT_M2.md results). This is the standard RLGym table used for strong
community bots, so changing it (new id, retraining) needs concrete evidence that it blocks learning, which the
current data does not provide. The kickoff jump caveat (section 8) is the only measured sim/game difference
caused by action timing.

## 3. Rewards

All components return raw values; `LoggedReward` applies weights and logs raw and weighted sums per component
(`reward_per_agent_step` in `metrics.jsonl`). Kinds: `event` (per occurrence), `per_second` (rate x elapsed
seconds, so totals do not change with action repeat), `potential` (difference of a state potential).

| Component | Kind | Credit | Default weight | Intended behaviour | Failure mode / guard |
|---|---|---|---|---|---|
| `goal_scored` | event | team | 3 (stage 1) | The actual objective: +1 to every car on the team that scores. | Sparse; alone it gives no gradient early on. Counted once per episode even if the goal state persists (tested). |
| `goal_conceded` | event | team | 3 (stage 1) | Defence matters as much as scoring: -1 to every car on the conceding team. | If weighted far above goal_scored the policy can become passive. Keep equal unless evaluation shows a defensive weakness. |
| `ball_touch` | event | individual | 0.5 | Bootstrap contact; at most one event per car per decision, scaled by the ball's velocity change (0..1). | Touch farming: light taps earn almost nothing (tested), but hard back-and-forth hits can still be farmed. Decay once contact is reliable; it under-rewards soft dribble touches. |
| `velocity_to_ball` | per second | individual | 0.5 | Early shaping: speed towards the ball / 2300. | Ball chasing; penalises rotating back and boost collection. Remove after stage 1. |
| `ball_to_goal` | per second | team | 1.5 | Ball speed towards the opponent goal / 6000; antisymmetric between teams (tested). | Rewards booming the ball away (giving up possession). Reduce relative to goals later. |
| `boost_economy` | potential | individual | 0.1 | sqrt(boost/100) potential: pickups matter most when low. | A potential difference cannot be farmed by burn/collect cycles (tested); a large weight causes hoarding. Skipped across demolitions. |

Typical magnitudes per decision at the stage-1 defaults: velocity_to_ball up to 0.033, ball_to_goal up to ~0.05
for a fast shot, a solid touch ~0.25-0.5, a full boost pickup from empty 0.1, a goal 3. No single flashy mechanic
(air time, speed, boost held, flip resets) is rewarded on its own.

Why stage 1 uses goals = 3 (measured, STATUS.md): with the RLGym-guide-like weights (goals 10, dense terms ~0.01
per step) the first 5M steps changed nothing measurable, because early goals are essentially random and their
+-10 returns dominate the normalised advantage; a velocity-to-ball-only run learned within 2M steps, and the
stage-1 weights learned contact (touch signal x7, speed towards the ball from 0 to ~0.1 of max) within 4.4M steps.
Stage 2 restores goals to 10 and shrinks the shaping so that winning, not chasing, dominates.

**Team modes**: `team_spirit` blends each individual term with the team mean (team total preserved, tested), so
a supporting player is never punished because a teammate touched the ball (tested). Goal and ball-to-goal terms
are already team-level. Opponent outcomes: only goals and ball-to-goal are zero-sum by construction; touch, speed
and boost terms are not mirrored, both sides can earn them.

## 4. Scenarios and curriculum

`ScenarioMutator` draws one scenario per episode from a seeded mixture; the protagonist is blue or orange with
equal probability (tested), and all states pass a physical-sanity validator (tested for every scenario and team
size, including a short simulation from the spawn): kickoff (standard 5 spawns), open play, shot, save,
low-boost defence, recovery (airborne, tumbling, flip still available as after leaving a wall; never mid-flip),
aerial/wall opportunity. Default mixture: kickoff 50%, open play 20%, shot 8%, save 8%, low-boost defence 4%,
recovery 5%, aerial 5%. Episodes end on a goal (terminal) or after 30 s without a touch / 300 s (truncated,
bootstrapped).

Stages (progression is a **manual** decision based on evaluation and watching, never on a timestep count; select a
stage by editing a config and resuming with `--allow-config-change`; the stage label is stored in checkpoints):

| Stage | Change from the previous stage | Gate to move on (all must hold) |
|---|---|---|
| s1-contact (default.toml) | as shipped | Beats its own iteration-0 snapshot with score CI lower bound > 0.9 (50 pairs); goal fraction of episodes clearly rising in `metrics.jsonl`; watching: drives to the ball, recovers to wheels, no obvious farming loop. |
| s2-shots-saves | goals 10/10, `velocity_to_ball` 0.05, `ball_touch` 0.2, `boost_economy` 0.25; shot and save scenario weights 0.15 each | Beats best s1 with CI lower bound > 0.5 over 100 pairs; no regression vs older snapshots (promotion rule); visibly shoots on target and saves easy shots. |
| s3-recovery-boost | recovery and low-boost defence 0.1 each; `velocity_to_ball` 0 | Same promotion rule vs best s2; fewer turtled/stuck situations when watched; real-game check vs Psyonix bots. |
| s4-mechanics | needs new scenario generators and reward terms (dribble, flick, wall, aerial follow-up), possibly action repeat 4 as a separate experiment | Same rule; mechanic success rates measured per scenario (to be implemented). |
| s5-team | `team_size = 2`, `team_spirit` 0.3, slot shuffling or the entity encoder | 2v2 head-to-head gains while 1v1 strength is not lost (evaluate both). |

**Accepting more cars is not team play.** obs-v1 has slots for 3v3, but a 1v1-trained policy has never seen
teammates: it will double-commit and ignore rotation. Team play needs 2v2/3v3 self-play, team-level rewards
(`team_spirit`), evaluation in team formats, and either slot-order augmentation or the order-invariant entity
encoder.

## 5. PPO, self-play and opponents

* PPO maths: rlgym-learn-algos `PPOLearner` (clipped surrogate, entropy bonus, MSE value loss, per-batch
  advantage standardisation, grad-norm clip 0.5) and its Rust GAE processor (terminal: next value 0; truncated:
  value of the trajectory's own final observation; tested against a reference). Rewards are divided by a running
  return standard deviation and clipped to +-10.
* On-policy: every iteration collects fresh data with one policy version (`policy_version` = iteration in the
  metrics), the buffer is cleared before submission, no lag and no reuse. Pipelined collection overlaps
  simulation and inference without changing the data (tested equal to lockstep).
* Episode boundaries: workers reset finished environments immediately; the final observation of the old episode is
  kept for bootstrapping and the new episode starts with a cleared previous action (tested). Rollouts cut mid
  episode are truncations, bootstrapped from the next observation.
* **Self-play**: by default both cars in every environment are the current policy and both produce experience.
* **Frozen opponents** (`[self_play] frozen_opponent_fraction`): in that fraction of environments one whole team
  (alternating orange/blue) is driven by a frozen snapshot, sampled uniformly from the run's opponent pool each
  iteration. Its actions come from its own bundle (same contract check as deployment) and its transitions are
  never given to the learner (tested). Snapshots are added every `snapshot_every_iterations`.
  Suggested start once the policy plays coherent games: 0.2 frozen, i.e. 80% current self-play. Uniform sampling,
  0.2 and the snapshot interval are hypotheses to tune against forgetting (evaluate vs old snapshots) and
  over-exploitation (evaluate vs held-out snapshots).
* Opponent registry (`runs/<run>/opponents/registry.json`) marks each opponent `sim`, `rlbot-only` or
  `not-integrated`. Only `sim` entries can be used in training/headless evaluation. The RLBot bot pack bots
  (Necto, Nexto etc. as executables) and Psyonix bots are `rlbot-only`: usable for real-game checks only.
  Not yet built: a league with per-opponent win tracking (prioritised sampling).

## 6. Evaluation and promotion

`python -m botcore evaluate` plays colour-swapped seed pairs (seeds from 1,000,000 upwards, disjoint from training
seeds) under fixed rules: 300 s regulation, overtime to the next goal capped at 120 s, then a draw; kickoff
after each goal; no replays. Reports record W/D/L, goals for/against, score, a 95% bootstrap interval with the
**seed pair** as the sampling unit, per-colour scores, seeds, bundle hashes and sources, inference mode per
side (deterministic vs stochastic is chosen per side and labelled), dependency versions via the manifests.

Pointers: `latest` (moved by training), `candidate` (set by `export --mark-candidate`), `best_verified` (only set
by `promote`). Promotion rule (predeclared in `evaluate.py`): >= 100 seed pairs vs the current best with CI lower
bound > 0.5, and no score below 0.45 vs any other benchmark opponent supplied. A reward increase never promotes
anything. A short evaluation (like the smoke test's) validates the harness only.

Internal Elo is not implemented; if added it is relative to its own opponent pool and is not Rocket League MMR.

Judging progress in practice: every few hours of training, evaluate `latest` vs the previous candidate and vs an
early snapshot (100 pairs ~ 3-6 min on this CPU), watch a few minutes in RLViser, and do a real-game check
(vs Psyonix All-Star, then vs you). Simulator wins alone do not prove deployment works.

## 7. Models

* **MLP baseline** (default, 512x3 actor and critic, separate trunks): fast, simple, sees car slots positionally.
* **Entity encoder** (`configs/entity.toml`): the controlled car is a query attending over ball, car and 34 pad
  tokens with a key-padding mask for absent cars (2 layers, 4 heads, d=128). Order-invariant over cars and
  independent of player count. Implemented, exported and loaded through the same paths and unit-tested; its
  training value is **not yet measured** (experiment 3 below). Benchmark numbers in STATUS.md.
* Architecture changes never preserve weights: resuming with a different architecture is refused. Moving to a new
  architecture means a new run, or distillation from the old policy (follow-up).
* No recurrence yet. Partial observability is limited (full state is visible; the previous action covers held
  inputs). If a recurrent or history model is added, its hidden state must be reset wherever the action parser
  is reset (environment reset in training; kickoff and on-the-fly resets in rlgym-rlbot), and the critic may use
  privileged inputs only through a separate critic observation, never the actor's.

## 8. Deployment limits (simulator vs game)

* `on_ground` during the first ticks of a jump is inferred by rlgym-compat's 6-tick heuristic (`AirState.Jumping`);
  it can differ from RocketSim's wheel contact for a few ticks (the only exemption in the parity test).
* Boost pad timers are reconstructed from pickups; a pad already empty when the bot connects uses the packet
  timer, whose documented meaning changed in Oct 2026, until it next becomes active.
* Flip availability after the 1.25 s window: fixed from `dodge_timeout`; untested against the real game.
* Demolished cars: kinematics zeroed (sim and game report different placeholder poses).
* Car slot order follows the packet's player order; with one opponent (1v1) this is irrelevant.
* RocketSim is not a perfect replica of Rocket League; small physics differences and a different input path mean
  that sim win rates do not transfer exactly.
* rlbot 2.0.0b52 (pinned by rlgym-rlbot) is older than the installed RLBotServer build (11d5d86); the schema changes
  in between are additive, and the combination **ran without errors in live test 1** (docs/LIVE_TEST_1.md).
* **Overtime kickoffs** (found live): rlgym-rlbot 0.3.0 only runs its kickoff reset when the countdown follows
  GoalScored/Replay, so the overtime kickoff (Active -> Countdown) skipped the reset, kept the previous action and
  re-timed the last action through the countdown. BotCoreBot now routes such countdowns through the normal reset
  path; verified by replaying the live recording through the real runner offline (1 -> 2 resets) and by a test.
* **Kickoff jump caveat** (seen once in 4 live kickoffs): if the first kickoff decision contains jump, rlgym-rlbot
  sends it during the countdown, so the game sees a held jump, not a press, and the car does not jump. Training
  treats that decision as a real jump. Left as is: a trained policy that relies on a first-tick kickoff jump would
  need a hard-coded kickoff or a countdown-release tweak; measure how often the trained policy does it first.
* Live diagnostics: `BOTCORE_LIVE_DIR=<dir>` makes each bot write a log, a packet/decision/timing recording and a
  summary; `python -m botcore live-report --dir <dir>` analyses them; `python -m botcore replay-html --recording
  <file.bcp>` draws the match; `botcore.deploy.offline` drives the real runner with recorded packets.
* Faults: if the bundle is missing/incompatible the bot logs CRITICAL and sends neutral inputs; inference
  exceptions fall back to the all-zero action with logged counts; a tick with no action produced gets a zero
  ControllerState (rlgym-rlbot would repeat the last action).
* Timing: rlgym-rlbot warns above 6 ms per tick; BotCore logs p50/p99/max and the count over 6 ms every 30 s.
  Use a capped 120 or 240 FPS.

## 9. Next experiments

1. **First long stage-1 run plus a real-game check.** `train --config configs/default.toml` for ~12-24 h
   (~0.8-1.6B learner steps at ~19k steps/s), evaluating every few hours vs the iteration-0 and earlier
   snapshots, then `export --to-bot` and play vs Psyonix bots and yourself with `BOTCORE_RECORD` on.
   Success: each later candidate beats the previous one (CI lower bound > 0.5, 100 pairs); in game: zero fallback
   ticks, <1% of ticks over 6 ms, coherent play; a concrete milestone is winning >= 8 of 10 matches vs All-Star.
2. **Live parity and timing verification.** Record packets in a real match (`BOTCORE_RECORD=...`), run
   `python -m botcore replay`, and compare against the synthetic parity tests; repeat at 120 and 240 FPS.
   Success: no non-finite or out-of-scale observations, deadline misses below 1%, no on-the-fly resets other
   than kickoffs in the log.
3. **MLP vs entity encoder at equal wall-clock** (e.g. 8 h each, same seeds and config otherwise), then 2v2.
   Success: entity encoder >= MLP head-to-head (CI over 100 pairs) at equal time while p99 deployment latency
   stays under 6 ms; then start s5 with the better model.

Further candidates: frozen-opponent fraction 0.2 vs 0 (forgetting/exploitation), action repeat 4 vs 8 (as a new
run), replay pretraining / behaviour cloning from high-level replays, distillation into larger nets, per-scenario
success metrics for curriculum gates.
