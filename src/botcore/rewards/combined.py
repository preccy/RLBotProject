"""Weighted sum of reward components with per-component logging."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from rlgym.api import RewardFunction

from botcore import constants as C
from botcore.rewards.components import RewardComponent, StepContext, make_components


def apply_team_spirit(values: np.ndarray, teams: Sequence[int], team_spirit: float) -> np.ndarray:
    """Blend each car's individual reward with its team's mean: r_i' = (1-ts) r_i + ts mean(team).

    The team's total is preserved, so this only changes credit assignment between teammates. With one car
    per team (1v1) it is the identity. Opponent rewards are NOT subtracted: zero-sum treatment is a property
    of individual components (goals, ball_to_goal), not something forced onto every shaping term.
    """
    if team_spirit <= 0.0:
        return values
    out = values.copy()
    teams_arr = np.asarray(teams)
    for team in np.unique(teams_arr):
        mask = teams_arr == team
        out[mask] = (1.0 - team_spirit) * values[mask] + team_spirit * values[mask].mean()
    return out


class LoggedReward(RewardFunction[Any, Any, float]):
    """RLGym v2 RewardFunction: sum of weighted components, recording raw and weighted contributions.

    ``stats`` accumulates, per component, the sum of raw and weighted contributions over all agent-steps
    since the last `pop_stats()`; the collector ships it to the trainer every rollout.
    """

    def __init__(self, weights: Dict[str, float], team_spirit: float = 0.0):
        if not 0.0 <= team_spirit <= 1.0:
            raise ValueError("team_spirit must be in [0, 1]")
        self.components: List[tuple[RewardComponent, float]] = make_components(weights)
        self.team_spirit = float(team_spirit)
        self._prev_state = None
        self._goal_rewarded = False
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._raw = {c.name: 0.0 for c, _ in self.components}
        self._weighted = {c.name: 0.0 for c, _ in self.components}
        self._agent_steps = 0
        self._seconds = 0.0

    def pop_stats(self) -> dict:
        out = {
            "raw": dict(self._raw),
            "weighted": dict(self._weighted),
            "agent_steps": self._agent_steps,
            "agent_seconds": self._seconds,
        }
        self._reset_stats()
        return out

    def reset(self, agents: List[Any], initial_state, shared_info: Dict[str, Any]) -> None:
        self._prev_state = initial_state
        self._goal_rewarded = False
        for component, _ in self.components:
            component.reset(agents, initial_state)

    def get_rewards(self, agents: List[Any], state, is_terminated: Dict[Any, bool],
                    is_truncated: Dict[Any, bool], shared_info: Dict[str, Any]) -> Dict[Any, float]:
        prev_state = self._prev_state if self._prev_state is not None else state
        dt = max(0.0, (state.tick_count - prev_state.tick_count) / C.TICKS_PER_SECOND)
        goal_event = bool(state.goal_scored) and not self._goal_rewarded
        if goal_event:
            self._goal_rewarded = True
        ctx = StepContext(dt=dt, goal_event=goal_event, scoring_team=state.scoring_team if goal_event else None)
        teams = [state.cars[a].team_num for a in agents]

        total = np.zeros(len(agents), dtype=np.float64)
        self.last_breakdown = {}  # component -> weighted contribution per agent (diagnostics)
        for component, weight in self.components:
            raw = np.asarray(component.compute(agents, state, prev_state, ctx), dtype=np.float64)
            if component.kind == "per_second":
                raw = raw * dt
            if component.individual:
                raw = apply_team_spirit(raw, teams, self.team_spirit)
            weighted = raw * weight
            self.last_breakdown[component.name] = weighted.copy()
            self._raw[component.name] += float(raw.sum())
            self._weighted[component.name] += float(weighted.sum())
            total += weighted

        self._agent_steps += len(agents)
        self._seconds += dt * len(agents)
        self._prev_state = state
        return {agent: float(total[i]) for i, agent in enumerate(agents)}
