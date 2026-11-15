"""PPO data path: GAE terminal vs truncation bootstrapping, rollout boundaries, frozen-opponent isolation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rlgym_learn_algos.ppo import GAETrajectoryProcessor, GAETrajectoryProcessorConfigModel, Trajectory
from rlgym_learn_algos.ppo.trajectory_processor import DerivedTrajectoryProcessorConfig

from botcore.collect import FROZEN, LEARNER, Collector, RolloutBuffer, StepResult, role_map, run_rollout
from botcore.config import Config
from botcore.env import EnvSpec
from botcore.models import ActorCriticPolicy, arch_from_config, build_actor, build_critic
from botcore.obs import LAYOUT, OBS_SIZE
from botcore.obs import spec as S

GAMMA, LAMBDA = 0.9, 0.8


def _processor(standardize=False):
    tp = GAETrajectoryProcessor()
    tp.load(DerivedTrajectoryProcessorConfig(
        trajectory_processor_config=GAETrajectoryProcessorConfigModel(gamma=GAMMA, lmbda=LAMBDA,
                                                                      standardize_rewards=standardize,
                                                                      reward_clip=None),
        agent_controller_name="test", dtype=torch.float32, device=torch.device("cpu"), seed=0))
    return tp


def _reference_gae(rewards, values, final_value, truncated):
    adv = np.zeros(len(rewards))
    next_value = final_value if truncated else 0.0
    running = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_value - values[t]
        running = delta + GAMMA * LAMBDA * running
        adv[t] = running
        next_value = values[t]
    return adv


def _trajectory(agent, ids, rewards, values, final_value, truncated):
    obs = [np.array([float(i)], dtype=np.float32) for i in ids]
    return Trajectory(agent, obs, [np.array([0]) for _ in ids], torch.zeros(len(ids)), list(rewards),
                      torch.tensor(values, dtype=torch.float32), np.array([-1.0], dtype=np.float32),
                      torch.tensor(final_value, dtype=torch.float32), truncated)


def test_gae_bootstraps_truncations_but_not_terminals_and_never_crosses_trajectories():
    rewards_a, values_a = [1.0, 0.0, 2.0], [0.5, 0.4, 0.3]
    rewards_b, values_b = [0.0, 0.0, 10.0], [1.0, 2.0, 3.0]
    trajectories = [
        _trajectory("a", [0, 1, 2], rewards_a, values_a, 5.0, truncated=True),
        _trajectory("b", [10, 11, 12], rewards_b, values_b, 100.0, truncated=False),  # final value must be ignored
    ]
    (agent_ids, obs, _, _, value_out, adv), _ = _processor().process_trajectories(trajectories)
    got = {int(o[0]): float(a) for o, a in zip(obs, np.asarray(adv))}
    expected_a = _reference_gae(rewards_a, values_a, 5.0, truncated=True)
    expected_b = _reference_gae(rewards_b, values_b, 100.0, truncated=False)
    for i, e in zip([0, 1, 2], expected_a):
        assert got[i] == pytest.approx(e, rel=1e-5)
    for i, e in zip([10, 11, 12], expected_b):
        assert got[i] == pytest.approx(e, rel=1e-5)
    # the bootstrapped truncation and the terminal really differ by the final value term
    terminal_a = _reference_gae(rewards_a, values_a, 5.0, truncated=False)
    assert got[2] - terminal_a[2] == pytest.approx(GAMMA * 5.0, rel=1e-5)


def _result(n_envs, n_agents, rewards=0.0, terminated=(), truncated=(), final_obs=None, obs_value=0.0):
    term = np.zeros(n_envs, dtype=bool)
    trunc = np.zeros(n_envs, dtype=bool)
    term[list(terminated)] = True
    trunc[list(truncated)] = True
    return StepResult(obs=np.full((n_envs, n_agents, 3), obs_value, dtype=np.float32),
                      rewards=np.full((n_envs, n_agents), rewards, dtype=np.float32),
                      terminated=term, truncated=trunc, final_obs=final_obs or {})


def test_rollout_buffer_splits_episodes_at_terminal_and_truncation_boundaries():
    agents = ["blue-0", "orange-0"]
    roles = role_map(2, agents, 0)
    buf = RolloutBuffer(roles, agents)
    actions = np.zeros((2, 2), dtype=np.int64)
    logp = np.zeros((2, 2), dtype=np.float32)
    obs0 = np.zeros((2, 2, 3), dtype=np.float32)
    # step 1: env 0 terminates (goal), env 1 continues
    end_obs = {0: np.full((2, 3), 7.0, dtype=np.float32)}
    r1 = _result(2, 2, rewards=1.0, terminated=[0], final_obs=end_obs, obs_value=1.0)
    buf.record(obs0, actions, logp, r1)
    # step 2: env 1 is truncated (timeout)
    r2 = _result(2, 2, rewards=2.0, truncated=[1], final_obs={1: np.full((2, 3), 9.0, dtype=np.float32)}, obs_value=2.0)
    buf.record(r1.obs, actions, logp, r2)
    trajs = buf.finish(r2.obs)
    assert buf.steps == 8
    terminal = [t for t in trajs if not t.truncated]
    assert len(terminal) == 2 and all(len(t.obs) == 1 and t.final_obs[0] == 7.0 for t in terminal)
    timeouts = [t for t in trajs if t.truncated and t.final_obs[0] == 9.0]
    assert len(timeouts) == 2 and all(len(t.obs) == 2 and t.rewards == [1.0, 2.0] for t in timeouts)
    # the trajectory that started after env 0's reset begins with the post-reset observation, not the old one
    restarted = [t for t in trajs if t.truncated and t.final_obs[0] == 2.0 and len(t.obs) == 1]
    assert len(restarted) == 2 and all(t.obs[0][0] == 1.0 for t in restarted)


def test_frozen_agents_never_enter_learner_experience():
    agents = ["blue-0", "orange-0"]
    roles = role_map(4, agents, 2)
    assert (roles == FROZEN).sum() == 2
    assert {tuple(r) for r in roles.tolist()} == {(LEARNER, LEARNER), (LEARNER, FROZEN), (FROZEN, LEARNER)}
    buf = RolloutBuffer(roles, agents)
    assert len(buf.learner_slots) == 6
    obs = np.zeros((4, 2, 3), dtype=np.float32)
    buf.record(obs, np.zeros((4, 2), dtype=np.int64), np.zeros((4, 2), dtype=np.float32), _result(4, 2))
    trajs = buf.finish(obs)
    learner_pairs = {(i, j) for i in range(4) for j in range(2) if roles[i, j] == LEARNER}
    assert len(trajs) == len(learner_pairs) == 6


def test_backprop_log_probs_match_sampling_distribution():
    cfg = Config()
    arch = arch_from_config(cfg.model)
    torch.manual_seed(0)
    policy = ActorCriticPolicy(build_actor(arch, OBS_SIZE, 90), build_critic(arch, OBS_SIZE), "cpu")
    obs = torch.randn(64, OBS_SIZE)
    actions, log_probs = policy.act(obs)
    lp, entropy, values = policy.get_backprop_data(None, obs, actions.unsqueeze(-1))
    torch.testing.assert_close(lp, log_probs)
    assert values.shape == (64,) and torch.isfinite(entropy)
    assert entropy.item() == pytest.approx(np.log(90), rel=0.02)  # near-uniform at initialisation


@pytest.mark.slow
def test_collector_resets_cleanly_and_shuts_down():
    cfg = Config()
    cfg.env.no_touch_timeout_seconds = 2.0  # force truncations quickly
    spec = EnvSpec.from_config(cfg)
    collector = Collector(spec, n_workers=2, envs_per_worker=2, seed=5)
    try:
        obs = collector.reset()
        assert obs.shape == (4, 2, OBS_SIZE) and collector.agents == ["blue-0", "orange-0"]
        prev_idx = LAYOUT.self_slice.start + S.SELF_IDX["prev_action"].start
        rng = np.random.default_rng(0)
        ended = 0
        for _ in range(60):
            result = collector.step(rng.integers(0, 90, size=(4, 2)))
            assert np.isfinite(result.obs).all() and np.isfinite(result.rewards).all()
            for env_idx, final in result.final_obs.items():
                ended += 1
                assert final.shape == (2, OBS_SIZE)
                # the returned observation is the first one of the NEW episode: previous action cleared
                assert not result.obs[env_idx, :, prev_idx:prev_idx + 8].any()
            assert not (result.terminated & result.truncated).any()
        assert ended > 0
        stats, episodes = collector.pop_stats()
        assert stats["agent_steps"] == 60 * 4 * 2 and len(episodes) == ended
    finally:
        collector.close()
    assert collector.alive_workers() == 0


@pytest.mark.slow
def test_run_rollout_requires_a_frozen_policy_when_roles_need_one():
    cfg = Config()
    collector = Collector(EnvSpec.from_config(cfg), n_workers=1, envs_per_worker=2, seed=1)
    try:
        obs = collector.reset()
        roles = role_map(2, collector.agents, 1)
        with pytest.raises(ValueError, match="frozen policy"):
            run_rollout(collector, obs, roles, 2, lambda o: (np.zeros(len(o), dtype=np.int64), np.zeros(len(o))))
        calls = []

        def frozen(o):
            calls.append(len(o))
            return np.full(len(o), 8)

        trajs, _, timing = run_rollout(collector, obs, roles, 5,
                                       lambda o: (np.zeros(len(o), dtype=np.int64), np.zeros(len(o))), frozen)
        assert calls == [1] * 5 and timing["learner_steps"] == 3 * 5
        assert sum(len(t.obs) for t in trajs) == 15
    finally:
        collector.close()


@pytest.mark.slow
def test_pipelined_rollout_produces_exactly_the_same_data_as_lockstep():
    cfg = Config()
    cfg.env.no_touch_timeout_seconds = 3.0  # make episode boundaries happen inside the rollout
    spec = EnvSpec.from_config(cfg)

    def policy(o):  # deterministic function of the observation
        idx = (np.abs(o[:, :40]).sum(axis=1) * 1000).astype(np.int64) % 90
        return idx, -np.ones(len(o), dtype=np.float32)

    results = []
    for pipeline in (False, True):
        collector = Collector(spec, n_workers=2, envs_per_worker=2, seed=123)
        try:
            obs = collector.reset()
            roles = role_map(4, collector.agents, 0)
            trajs, next_obs, _ = run_rollout(collector, obs, roles, 80, policy, pipeline=pipeline)
            trajs2, _, _ = run_rollout(collector, next_obs, roles, 20, policy, pipeline=pipeline)
        finally:
            collector.close()
        results.append((trajs + trajs2, next_obs))
    (a, obs_a), (b, obs_b) = results
    np.testing.assert_array_equal(obs_a, obs_b)
    key = lambda t: (t.agent_id, float(t.obs[0].sum()), len(t.obs))
    a, b = sorted(a, key=key), sorted(b, key=key)
    assert len(a) == len(b) and any(not t.truncated for t in a) or any(t.truncated for t in a)
    for ta, tb in zip(a, b):
        assert ta.agent_id == tb.agent_id and ta.truncated == tb.truncated
        np.testing.assert_array_equal(np.asarray(ta.obs), np.asarray(tb.obs))
        np.testing.assert_array_equal(np.asarray(ta.actions), np.asarray(tb.actions))
        assert ta.rewards == tb.rewards
        np.testing.assert_array_equal(ta.final_obs, tb.final_obs)
