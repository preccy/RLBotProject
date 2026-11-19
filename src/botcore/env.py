"""Construct the headless RLGym v2 training environment from config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine

from botcore.actions import LookupActionParser
from botcore.config import Config
from botcore.obs import SpecObsBuilder
from botcore.rewards import LoggedReward
from botcore.scenarios import ScenarioMutator


@dataclass
class EnvSpec:
    """Plain, picklable description of one environment (sent to worker processes)."""

    team_size: int
    action_repeat: int
    rlbot_delay: bool
    no_touch_timeout_seconds: float
    episode_timeout_seconds: float
    scenario_weights: Dict[str, float]
    reward_weights: Dict[str, float]
    team_spirit: float = 0.0

    @classmethod
    def from_config(cls, cfg: Config) -> "EnvSpec":
        return cls(
            team_size=cfg.env.team_size,
            action_repeat=cfg.env.action_repeat,
            rlbot_delay=cfg.env.rlbot_delay,
            no_touch_timeout_seconds=cfg.env.no_touch_timeout_seconds,
            episode_timeout_seconds=cfg.env.episode_timeout_seconds,
            scenario_weights=dict(cfg.env.scenario_weights),
            reward_weights=dict(cfg.rewards.weights),
            team_spirit=cfg.rewards.team_spirit,
        )


def build_env(spec: EnvSpec, seed: int, renderer=None, raw_controls: bool = False) -> RLGym:
    """Build one 1v1..3v3 RocketSim environment.

    Termination (true terminal state): a goal. Truncation (artificial cut, value is bootstrapped):
    no touch for `no_touch_timeout_seconds`, or `episode_timeout_seconds` of game time.
    `raw_controls` lets one car send 8 raw controls (external frozen opponent, botcore.necto).
    """
    return RLGym(
        state_mutator=ScenarioMutator(spec.team_size, spec.scenario_weights, seed),
        obs_builder=SpecObsBuilder(),
        action_parser=LookupActionParser(spec.action_repeat, allow_raw_controls=raw_controls),
        reward_fn=LoggedReward(spec.reward_weights, spec.team_spirit),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=spec.no_touch_timeout_seconds),
            TimeoutCondition(timeout_seconds=spec.episode_timeout_seconds),
        ),
        transition_engine=RocketSimEngine(rlbot_delay=spec.rlbot_delay),
        renderer=renderer,
    )
