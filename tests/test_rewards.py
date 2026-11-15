"""Reward components: single counting of events, time scaling, logging, team handling."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from botcore import constants as C
from botcore.rewards import COMPONENTS, LoggedReward, apply_team_spirit
from tests.helpers import make_env, stepped_state

AGENTS = ["blue-0", "orange-0"]
NOT_DONE = {a: False for a in AGENTS}


def _advance(state, ticks=8):
    nxt = copy.deepcopy(state)
    nxt.tick_count = state.tick_count + ticks
    for car in nxt.cars.values():
        car.ball_touches = 0
    return nxt


def _reward(weights, state, team_spirit=0.0):
    fn = LoggedReward(weights, team_spirit)
    fn.reset(list(state.cars), state, {})
    return fn


def test_unknown_component_is_a_config_error():
    with pytest.raises(ValueError, match="unknown reward component"):
        LoggedReward({"style_points": 1.0})


def test_goal_is_rewarded_exactly_once_even_if_the_goal_state_persists():
    _, state = stepped_state(1, seed=0)
    fn = _reward({"goal_scored": 10.0, "goal_conceded": 10.0}, state)
    scored = _advance(state)
    scored.goal_scored = True
    scored.ball.position = np.array([0.0, C.GOAL_THRESHOLD_Y + 50.0, 100.0], dtype=np.float32)  # in orange's net
    first = fn.get_rewards(AGENTS, scored, NOT_DONE, NOT_DONE, {})
    assert first == {"blue-0": 10.0, "orange-0": -10.0}
    still_in_goal = _advance(scored)
    still_in_goal.goal_scored = True
    still_in_goal.ball.position = scored.ball.position.copy()
    second = fn.get_rewards(AGENTS, still_in_goal, NOT_DONE, NOT_DONE, {})
    assert second == {"blue-0": 0.0, "orange-0": 0.0}
    stats = fn.pop_stats()
    assert stats["raw"]["goal_scored"] == 1.0 and stats["raw"]["goal_conceded"] == -1.0
    # a new episode can be rewarded again
    fn.reset(AGENTS, state, {})
    again = fn.get_rewards(AGENTS, scored, NOT_DONE, NOT_DONE, {})
    assert again["blue-0"] == 10.0


def test_one_contact_is_one_touch_event_scaled_by_ball_velocity_change():
    _, state = stepped_state(1, seed=0)
    state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    fn = _reward({"ball_touch": 1.0}, state)
    hit = _advance(state)
    hit.cars["blue-0"].ball_touches = 5  # several contact ticks inside one decision
    hit.ball.linear_velocity = np.array([1000.0, 0.0, 0.0], dtype=np.float32)
    out = fn.get_rewards(AGENTS, hit, NOT_DONE, NOT_DONE, {})
    assert out["blue-0"] == pytest.approx(0.5) and out["orange-0"] == 0.0
    # a hard hit saturates at 1 event, never more
    hard = _advance(hit)
    hard.cars["blue-0"].ball_touches = 1
    hard.ball.linear_velocity = np.array([6000.0, 0.0, 0.0], dtype=np.float32)
    assert fn.get_rewards(AGENTS, hard, NOT_DONE, NOT_DONE, {})["blue-0"] == pytest.approx(1.0)
    # no touch flag -> no reward even though the ball changed velocity
    free = _advance(hard)
    free.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    assert fn.get_rewards(AGENTS, free, NOT_DONE, NOT_DONE, {})["blue-0"] == 0.0


def test_light_taps_earn_almost_nothing():
    _, state = stepped_state(1, seed=0)
    state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    fn = _reward({"ball_touch": 1.0}, state)
    total = 0.0
    cur = state
    for i in range(20):  # twenty consecutive 30 uu/s nudges
        cur = _advance(cur)
        cur.cars["blue-0"].ball_touches = 1
        cur.ball.linear_velocity = np.array([30.0 * (i + 1), 0.0, 0.0], dtype=np.float32)
        total += fn.get_rewards(AGENTS, cur, NOT_DONE, NOT_DONE, {})["blue-0"]
    assert total < 0.5, "twenty light taps must be worth less than half of one solid hit"


def test_per_second_components_scale_with_elapsed_time_not_step_count():
    _, state = stepped_state(1, seed=1)
    car = state.cars["blue-0"]
    car.physics.position = np.array([0.0, -1000.0, 17.0], dtype=np.float32)
    car.physics.linear_velocity = np.array([0.0, 1150.0, 0.0], dtype=np.float32)  # half max speed at the ball
    state.ball.position = np.array([0.0, 0.0, 17.0], dtype=np.float32)
    results = {}
    for ticks in (4, 8, 16):
        fn = _reward({"velocity_to_ball": 1.0}, state)
        results[ticks] = fn.get_rewards(AGENTS, _advance(state, ticks), NOT_DONE, NOT_DONE, {})["blue-0"]
    assert results[8] == pytest.approx(0.5 * 8 / 120)
    assert results[16] == pytest.approx(2 * results[8]) and results[4] == pytest.approx(results[8] / 2)


def test_ball_to_goal_is_antisymmetric_between_teams():
    _, state = stepped_state(1, seed=1)
    state.ball.position = np.array([0.0, 0.0, 100.0], dtype=np.float32)
    state.ball.linear_velocity = np.array([0.0, 3000.0, 0.0], dtype=np.float32)
    fn = _reward({"ball_to_goal": 1.0}, state)
    nxt = _advance(state, 120)
    out = fn.get_rewards(AGENTS, nxt, NOT_DONE, NOT_DONE, {})
    assert out["blue-0"] > 0 and out["blue-0"] == pytest.approx(-out["orange-0"])
    assert out["blue-0"] == pytest.approx(0.5, abs=0.01)  # 3000/6000 for one second


def test_boost_economy_cannot_be_farmed_and_skips_demolitions():
    _, state = stepped_state(1, seed=2)
    state.cars["blue-0"].boost_amount = 0.0
    fn = _reward({"boost_economy": 1.0}, state)
    total = 0.0
    cur = state
    for amount in (100.0, 0.0, 100.0, 0.0):  # collect, burn, collect, burn
        cur = _advance(cur)
        cur.cars["blue-0"].boost_amount = amount
        total += fn.get_rewards(AGENTS, cur, NOT_DONE, NOT_DONE, {})["blue-0"]
    assert total == pytest.approx(0.0, abs=1e-9)
    demoed = _advance(cur)
    demoed.cars["blue-0"].demo_respawn_timer = 3.0
    demoed.cars["blue-0"].boost_amount = 33.0
    assert fn.get_rewards(AGENTS, demoed, NOT_DONE, NOT_DONE, {})["blue-0"] == 0.0


def test_coincident_positions_do_not_produce_nan():
    _, state = stepped_state(1, seed=3)
    state.ball.position = np.array(state.cars["blue-0"].physics.position, dtype=np.float32)
    fn = _reward({name: 1.0 for name in COMPONENTS}, state)
    out = fn.get_rewards(AGENTS, _advance(state), NOT_DONE, NOT_DONE, {})
    assert all(np.isfinite(v) for v in out.values())


def test_stats_log_raw_and_weighted_contributions_separately():
    _, state = stepped_state(1, seed=4)
    weights = {"velocity_to_ball": 0.15, "ball_to_goal": 1.5, "ball_touch": 0.25}
    fn = _reward(weights, state)
    totals = {a: 0.0 for a in AGENTS}
    cur = state
    for _ in range(5):
        cur = _advance(cur)
        for agent, value in fn.get_rewards(AGENTS, cur, NOT_DONE, NOT_DONE, {}).items():
            totals[agent] += value
    stats = fn.pop_stats()
    assert stats["agent_steps"] == 10 and stats["agent_seconds"] == pytest.approx(10 * 8 / 120)
    for name, weight in weights.items():
        assert stats["weighted"][name] == pytest.approx(stats["raw"][name] * weight)
    assert sum(stats["weighted"].values()) == pytest.approx(sum(totals.values()))
    assert fn.pop_stats()["agent_steps"] == 0, "pop_stats must reset the accumulators"


def test_team_spirit_only_redistributes_within_a_team():
    values = np.array([4.0, 0.0, -2.0, 0.0])
    teams = [0, 0, 1, 1]
    np.testing.assert_allclose(apply_team_spirit(values, teams, 0.0), values)
    blended = apply_team_spirit(values, teams, 0.5)
    np.testing.assert_allclose(blended, [3.0, 1.0, -1.5, -0.5])
    assert blended[:2].sum() == values[:2].sum() and blended[2:].sum() == values[2:].sum()
    np.testing.assert_allclose(apply_team_spirit(values, teams, 1.0), [2.0, 2.0, -1.0, -1.0])


def test_supporting_teammate_is_not_punished_when_a_teammate_touches():
    _, state = stepped_state(2, seed=5)
    agents = list(state.cars)
    state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    fn = LoggedReward({"ball_touch": 1.0}, team_spirit=0.3)
    fn.reset(agents, state, {})
    hit = _advance(state)
    hit.cars["blue-0"].ball_touches = 1
    hit.ball.linear_velocity = np.array([2000.0, 0.0, 0.0], dtype=np.float32)
    out = fn.get_rewards(agents, hit, {a: False for a in agents}, {a: False for a in agents}, {})
    assert out["blue-0"] == pytest.approx(0.85) and out["blue-1"] == pytest.approx(0.15)
    assert out["orange-0"] == 0.0 and out["orange-1"] == 0.0


def test_rewards_are_finite_in_a_real_rollout():
    env = make_env(1, seed=11)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(400):
        _, rewards, term, trunc = env.step({a: np.array([rng.integers(90)]) for a in env.agents})
        assert all(np.isfinite(r) for r in rewards.values())
        if any(term.values()) or any(trunc.values()):
            env.reset()
