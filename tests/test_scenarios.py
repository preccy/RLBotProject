"""Reset-state generation: validity, reproducibility, colour balance, simulator acceptance."""

from __future__ import annotations

import numpy as np
import pytest
from rlgym.rocket_league.sim import RocketSimEngine

from botcore import constants as C
from botcore.scenarios import SCENARIO_KEY, SCENARIOS, ScenarioMutator, validate_state
from tests.helpers import make_env


def _generate(name: str, team_size: int, seed: int, n: int):
    engine = RocketSimEngine()
    mutator = ScenarioMutator(team_size, {name: 1.0}, seed)
    states = []
    for _ in range(n):
        state = engine.create_base_state()
        mutator.apply(state, {})
        states.append(state)
    return states


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("team_size", [1, 2, 3])
def test_generated_states_are_physically_sensible(name, team_size):
    for state in _generate(name, team_size, seed=123, n=150):
        assert len(state.cars) == 2 * team_size
        assert validate_state(state) == []


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_same_seed_gives_identical_states(name):
    a = _generate(name, 2, seed=77, n=10)
    b = _generate(name, 2, seed=77, n=10)
    c = _generate(name, 2, seed=78, n=10)
    different = False
    for sa, sb, sc in zip(a, b, c):
        np.testing.assert_array_equal(sa.ball.position, sb.ball.position)
        for agent in sa.cars:
            np.testing.assert_array_equal(sa.cars[agent].physics.position, sb.cars[agent].physics.position)
            np.testing.assert_array_equal(sa.cars[agent].physics.linear_velocity, sb.cars[agent].physics.linear_velocity)
            assert sa.cars[agent].boost_amount == sb.cars[agent].boost_amount
            different = different or not np.array_equal(
                sa.cars[agent].physics.position, sc.cars[agent].physics.position
            )
    assert different, "a different seed should produce different states"


def test_kickoff_matches_the_standard_spawns():
    for state in _generate("kickoff", 1, seed=5, n=40):
        np.testing.assert_allclose(state.ball.position, [0, 0, C.BALL_RESTING_HEIGHT])
        blue, orange = state.cars["blue-0"], state.cars["orange-0"]
        np.testing.assert_allclose(orange.physics.position[:2], -blue.physics.position[:2])  # mirrored kickoff
        assert blue.physics.position[1] < 0 < orange.physics.position[1]
        assert blue.boost_amount == pytest.approx(33.3) and orange.boost_amount == pytest.approx(33.3)
        # both cars face the ball
        for car in (blue, orange):
            to_ball = -np.asarray(car.physics.position[:2], dtype=np.float64)
            assert np.dot(car.physics.forward[:2], to_ball / np.linalg.norm(to_ball)) > 0.7


def test_both_colours_play_each_role():
    """In the save scenario the defending side should be blue about half the time."""
    states = _generate("save", 1, seed=9, n=400)
    blue_defends = sum(1 for s in states if s.ball.linear_velocity[1] < 0)
    assert 150 < blue_defends < 250


def test_mixture_respects_weights_and_records_the_scenario():
    mutator = ScenarioMutator(1, {"kickoff": 0.75, "shot": 0.25, "save": 0.0}, seed=3)
    engine = RocketSimEngine()
    counts = {}
    for _ in range(2000):
        shared = {}
        mutator.apply(engine.create_base_state(), shared)
        counts[shared[SCENARIO_KEY]] = counts.get(shared[SCENARIO_KEY], 0) + 1
    assert set(counts) == {"kickoff", "shot"}
    assert 0.70 < counts["kickoff"] / 2000 < 0.80


def test_invalid_mixtures_are_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        ScenarioMutator(1, {"moonball": 1.0}, seed=0)
    with pytest.raises(ValueError, match="positive weight"):
        ScenarioMutator(1, {"kickoff": 0.0}, seed=0)
    with pytest.raises(ValueError, match="team_size"):
        ScenarioMutator(4, {"kickoff": 1.0}, seed=0)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_simulator_accepts_the_state_and_stays_stable(name):
    """After the engine loads the state and runs with idle controls nothing explodes or tunnels."""
    env = make_env(2, seed=31, scenario_weights={"kickoff": 1e-9, name: 1.0})
    idle = {}
    for _ in range(25):
        obs = env.reset()
        grounded = {a for a, car in env.state.cars.items() if car.on_ground}
        idle = {a: np.array([8]) for a in env.agents}  # index 8 = all-zero controls
        for _ in range(4):
            obs, rewards, _, _ = env.step(idle)
        state = env.state
        assert all(np.isfinite(o).all() for o in obs.values())
        assert np.isfinite(state.ball.position).all() and abs(state.ball.position[2]) < C.CEILING_Z
        for agent, car in state.cars.items():
            pos = car.physics.position
            assert np.isfinite(pos).all() and abs(pos[0]) < C.SIDE_WALL_X and abs(pos[1]) < C.BACK_NET_Y
            assert -5 < pos[2] < C.CEILING_Z
            if agent in grounded and not car.is_demoed:
                assert pos[2] < 60, f"{agent} spawned on the ground but is at z={pos[2]:.1f} after 32 ticks"
