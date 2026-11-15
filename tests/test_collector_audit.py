"""Semantic audit of the real multi-process collector (RocketSim workers, not synthetic StepResults).

Each test targets one consequential failure mode and checks *meaning*, not just shapes:
* terminal vs truncated labels and where the goal reward lands;
* the final observation is captured before the automatic reset, in each agent's own team frame;
* no trajectory mixes two agents or two episodes;
* stored log-probabilities equal a recomputation from the stored (obs, action) with the same network, which
  fails if observations, actions or log-probs are misaligned or come from more than one policy version;
* frozen-opponent transitions never reach the learner.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from botcore import constants as C
from botcore.collect import FROZEN, Collector, role_map, run_rollout
from botcore.config import Config
from botcore.env import EnvSpec
from botcore.models import arch_from_config, build_actor, sample_actions
from botcore.obs import LAYOUT, OBS_SIZE
from botcore.obs import spec as S

pytestmark = pytest.mark.slow

BALL_Y = LAYOUT.ball_slice.start + S.BALL_IDX["pos"].start + 1
SELF_POS = slice(LAYOUT.self_slice.start + S.SELF_IDX["pos"].start, LAYOUT.self_slice.start + S.SELF_IDX["pos"].start + 3)
GOAL_LINE = C.GOAL_THRESHOLD_Y / S.POS_SCALE
IDLE = 8


def _spec(scenarios, rewards, no_touch=30.0):
    cfg = Config()
    cfg.env.scenario_weights = scenarios
    cfg.rewards.weights = rewards
    cfg.env.no_touch_timeout_seconds = no_touch
    return EnvSpec.from_config(cfg)


def _actor(seed=0):
    cfg = Config()
    cfg.model.actor_layers = (64, 64)
    torch.manual_seed(seed)
    return build_actor(arch_from_config(cfg.model), OBS_SIZE, 90).eval()


def _sampler(actor, generator_seed=0):
    gen = torch.Generator().manual_seed(generator_seed)

    def act(obs):
        with torch.no_grad():
            a, lp = sample_actions(actor(torch.as_tensor(obs)), False, gen)
        return a.numpy(), lp.numpy()

    return act


def test_goal_rewards_terminals_and_final_observations():
    # Defenders stay idle while a fast ball heads for their goal: many short terminal episodes; the
    # 3 s no-touch timeout adds truncations.
    spec = _spec({"kickoff": 1e-9, "save": 1.0}, {"goal_scored": 1.0, "goal_conceded": 1.0}, no_touch=3.0)
    collector = Collector(spec, n_workers=2, envs_per_worker=3, seed=11)
    try:
        obs = collector.reset()
        roles = role_map(6, collector.agents, 0)
        idle = lambda o: (np.full(len(o), IDLE), np.zeros(len(o), dtype=np.float32))
        trajs, _, _ = run_rollout(collector, obs, roles, 150, idle)
        _, episodes = collector.pop_stats()
    finally:
        collector.close()

    terminal = [t for t in trajs if not t.truncated]
    truncated = [t for t in trajs if t.truncated]
    assert len(terminal) >= 10 and len(truncated) >= 4, (len(terminal), len(truncated))
    goals = [e for e in episodes if e["terminated"]]
    assert len(terminal) == 2 * len(goals), "each goal must close exactly one trajectory per agent"

    for t in terminal:
        reward = t.rewards[-1]
        assert reward in (1.0, -1.0), "the goal reward must be on the terminal step"
        assert all(r == 0.0 for r in t.rewards[:-1]), "goal reward must not be repeated or shifted"
        ball_y = t.final_obs[BALL_Y]
        # Final observation is pre-reset (ball in a net) and in this agent's team frame:
        # the scorer sees the ball in the attacked (+y) net, the conceder in its own (-y) net.
        assert abs(ball_y) > GOAL_LINE, f"final obs ball y {ball_y * S.POS_SCALE:.0f} is not in a goal (post-reset?)"
        assert np.sign(ball_y) == np.sign(reward), "reward sign disagrees with the side of the goal in the obs"
    for t in truncated:
        assert all(r == 0.0 for r in t.rewards), "a goal reward appeared in a trajectory not ending in a goal"

    max_step = 0.0
    print(f"\n[audit] goals={len(goals)} terminal trajs={len(terminal)} truncated trajs={len(truncated)} "
          f"terminal final-obs |ball y| min={min(abs(t.final_obs[BALL_Y]) for t in terminal) * S.POS_SCALE:.0f} uu "
          f"(goal line {C.GOAL_THRESHOLD_Y:.0f}); rewards on terminal steps: "
          f"+1 x{sum(t.rewards[-1] == 1.0 for t in terminal)}, -1 x{sum(t.rewards[-1] == -1.0 for t in terminal)}")
    for t in trajs:
        # one trajectory = one agent, one episode: the car cannot teleport between consecutive decisions
        pos = np.asarray(t.obs)[:, SELF_POS] * S.POS_SCALE
        if len(pos) > 1:
            step = np.linalg.norm(np.diff(pos, axis=0), axis=1).max()
            assert step < C.CAR_MAX_SPEED * 8 / 120 * 1.5 + 50, f"{t.agent_id}: jump of {step:.0f} uu inside a trajectory"
            max_step = max(max_step, step)
        # the first observation after a goal is a fresh state, never the ball still in a net
        assert abs(t.obs[0][BALL_Y]) < GOAL_LINE
    print(f"[audit] largest own-car displacement between consecutive decisions: {max_step:.0f} uu "
          f"(limit {C.CAR_MAX_SPEED * 8 / 120 * 1.5 + 50:.0f})")


def test_stored_log_probs_match_recomputation_with_frozen_opponents():
    """Alignment + single policy version + frozen exclusion, in pipelined mode with real workers."""
    spec = _spec({"kickoff": 0.5, "open_play": 0.5}, {"ball_touch": 1.0, "velocity_to_ball": 1.0}, no_touch=4.0)
    learner, frozen = _actor(0), _actor(1)
    with torch.no_grad():
        frozen.head.weight.mul_(300.0)  # a confident, very different policy, so leaked frozen data would be obvious
    frozen_gen = torch.Generator().manual_seed(9)

    def act_frozen(o):
        with torch.no_grad():
            return sample_actions(frozen(torch.as_tensor(o)), False, frozen_gen)[0].numpy()

    collector = Collector(spec, n_workers=3, envs_per_worker=2, seed=21)
    try:
        obs = collector.reset()
        roles = role_map(6, collector.agents, 2)
        trajs, obs, timing = run_rollout(collector, obs, roles, 120, _sampler(learner), act_frozen, pipeline=True)
        trajs2, _, _ = run_rollout(collector, obs, roles, 30, _sampler(learner, 1), act_frozen, pipeline=True)
    finally:
        collector.close()
    trajs = trajs + trajs2

    learner_slots = int((roles != FROZEN).sum())
    assert sum(len(t.obs) for t in trajs) == learner_slots * 150
    for t in trajs:
        j = collector.agents.index(t.agent_id)
        assert roles[t.env_index, j] != FROZEN, f"frozen slot env {t.env_index} {t.agent_id} reached the learner"

    all_obs = torch.as_tensor(np.concatenate([np.asarray(t.obs) for t in trajs]))
    all_act = torch.as_tensor(np.concatenate([np.asarray(t.actions) for t in trajs])).long()
    stored = np.concatenate([t.log_probs for t in trajs])
    with torch.no_grad():
        recomputed = torch.log_softmax(learner(all_obs), -1).gather(-1, all_act).squeeze(-1).numpy()
        under_frozen = torch.log_softmax(frozen(all_obs), -1).gather(-1, all_act).squeeze(-1).numpy()
    err = np.abs(recomputed - stored)
    print(f"\n[audit] {len(stored)} learner transitions ({len(trajs)} trajectories, 2 frozen envs, pipelined, 2 "
          f"rollouts): max |stored - recomputed log-prob| = {err.max():.2e}; mean |stored - frozen-net log-prob| = "
          f"{np.abs(under_frozen - stored).mean():.2f}")
    assert err.max() < 1e-5, f"log-prob mismatch {err.max():.2e} at {int(err.argmax())}"
    assert np.abs(under_frozen - stored).mean() > 0.5, "the check must be able to tell the two policies apart"
    # rewards were recorded for the acting agent: velocity_to_ball is per-agent and differs between the two cars
    assert any(len(set(t.rewards)) > 1 for t in trajs)


def test_rewards_are_assigned_to_the_agent_that_earned_them():
    """Blue drives at the ball, orange reverses away: only blue may earn velocity_to_ball reward."""
    spec = _spec({"kickoff": 1.0}, {"velocity_to_ball": 1.0}, no_touch=30.0)
    from botcore.actions import get_action_table

    table = get_action_table().controls
    forward = int(np.nonzero((table == np.array([1, 0, 0, 0, 0, 0, 0, 0])).all(axis=1))[0][0])
    reverse = int(np.nonzero((table == np.array([-1, 0, 0, 0, 0, 0, 0, 0])).all(axis=1))[0][0])
    collector = Collector(spec, n_workers=2, envs_per_worker=2, seed=5)
    try:
        obs = collector.reset()
        roles = role_map(4, collector.agents, 0)
        blue = collector.agents.index("blue-0")

        def policy(o):
            # rows of `o` are learner slots in env-major order: blue-0, orange-0, blue-0, ...
            n = len(o)
            acts = np.array([forward if k % 2 == blue else reverse for k in range(n)])
            return acts, np.zeros(n, dtype=np.float32)

        trajs, _, _ = run_rollout(collector, obs, roles, 20, policy, pipeline=True)
    finally:
        collector.close()
    blue_total = sum(sum(t.rewards[3:]) for t in trajs if t.agent_id == "blue-0")
    orange_total = sum(sum(t.rewards[3:]) for t in trajs if t.agent_id == "orange-0")
    print(f"\n[audit] velocity_to_ball reward: blue (driving at ball) {blue_total:+.3f}, orange (reversing) {orange_total:+.3f}")
    assert blue_total > 0.05 and orange_total < 0.0, (blue_total, orange_total)
    for t in trajs:  # the recorded action is the one this agent took
        expected = forward if t.agent_id == "blue-0" else reverse
        assert all(int(a[0]) == expected for a in t.actions)
