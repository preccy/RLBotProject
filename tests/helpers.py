"""Shared test helpers."""

from __future__ import annotations

import copy

import numpy as np

from botcore.config import Config
from botcore.env import EnvSpec, build_env

OPEN_PLAY_ONLY = {"kickoff": 1e-9, "open_play": 1.0}


def make_spec(team_size: int = 1, scenario_weights=None, **overrides) -> EnvSpec:
    cfg = Config()
    cfg.env.team_size = team_size
    if scenario_weights is not None:
        cfg.env.scenario_weights = dict(scenario_weights)
    spec = EnvSpec.from_config(cfg)
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def make_env(team_size: int = 1, seed: int = 0, scenario_weights=None, **overrides):
    return build_env(make_spec(team_size, scenario_weights, **overrides), seed)


def stepped_state(team_size: int = 1, seed: int = 0, steps: int = 12, scenario_weights=OPEN_PLAY_ONLY):
    """A non-trivial state: reset from a random scenario then apply random actions for a few decisions."""
    env = make_env(team_size, seed, scenario_weights)
    env.reset()
    rng = np.random.default_rng(seed)
    n_actions = env.action_parser.table.n_actions
    for _ in range(steps):
        env.step({a: np.array([rng.integers(n_actions)]) for a in env.agents})
    return env, env.state


def counterpart(agent_id: str) -> str:
    team, idx = agent_id.split("-")
    return f"{'orange' if team == 'blue' else 'blue'}-{idx}"


def rotated_swapped(state):
    """The same situation seen from the other end of the pitch: rotate 180 deg about Z and swap team colours."""
    out = copy.deepcopy(state)
    out.ball = state.ball.inverted()
    out._inverted_ball = None
    out.boost_pad_timers = np.ascontiguousarray(state.boost_pad_timers[::-1])
    out._inverted_boost_pad_timers = None
    cars = {}
    for agent_id, car in state.cars.items():
        new = copy.deepcopy(car)
        new.team_num = 1 - car.team_num
        new.physics = car.physics.inverted()
        new._inverted_physics = None
        cars[counterpart(agent_id)] = new
    out.cars = cars
    return out
