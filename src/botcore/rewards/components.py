"""Individually logged reward components.

Three kinds, so magnitudes stay interpretable if `action_repeat` changes:

* ``event``      - paid once when something happens (value per event).
* ``per_second`` - a rate; the combined reward multiplies it by the seconds the step covered, so the
                   same weight means "reward per second of play" at any decision frequency.
* ``potential``  - difference of a state potential between consecutive decisions (no time scaling).

Every component returns its RAW value; weights are applied by `LoggedReward`, which also records raw
and weighted sums for each component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from botcore import constants as C


@dataclass
class StepContext:
    dt: float  # seconds of game time covered by this step
    goal_event: bool  # True only on the first step of an episode in which a goal is registered
    scoring_team: Optional[int]


class RewardComponent:
    name: str = ""
    kind: str = "event"  # "event" | "per_second" | "potential"
    individual: bool = True  # True: credited to one car (subject to team_spirit); False: a team-level outcome
    intent: str = ""
    failure_mode: str = ""

    def reset(self, agents: Sequence, state) -> None:
        pass

    def compute(self, agents: Sequence, state, prev_state, ctx: StepContext) -> np.ndarray:
        raise NotImplementedError


class GoalScored(RewardComponent):
    name = "goal_scored"
    kind = "event"
    individual = False
    intent = "The actual objective: +1 to every car on the team that scores."
    failure_mode = ("Sparse; alone it gives no gradient early on. Double counting if the goal state is seen on "
                    "several steps - guarded by StepContext.goal_event (one event per episode).")

    def compute(self, agents, state, prev_state, ctx):
        out = np.zeros(len(agents), dtype=np.float64)
        if ctx.goal_event:
            for i, agent in enumerate(agents):
                if state.cars[agent].team_num == ctx.scoring_team:
                    out[i] = 1.0
        return out


class GoalConceded(RewardComponent):
    name = "goal_conceded"
    kind = "event"
    individual = False
    intent = "Defence matters as much as scoring: -1 to every car on the team that concedes."
    failure_mode = ("If weighted far above goal_scored the policy can become passive (never risk a challenge). "
                    "Keep it equal to goal_scored unless evaluation shows a defensive weakness.")

    def compute(self, agents, state, prev_state, ctx):
        out = np.zeros(len(agents), dtype=np.float64)
        if ctx.goal_event:
            for i, agent in enumerate(agents):
                if state.cars[agent].team_num != ctx.scoring_team:
                    out[i] = -1.0
        return out


class BallTouch(RewardComponent):
    name = "ball_touch"
    kind = "event"
    individual = True
    reference_speed = 2000.0  # uu/s change in ball velocity that earns the full event value
    intent = ("Bootstrap contact and reward purposeful hits: at most one event per car per decision, scaled by "
              "how much the ball's velocity changed (0..1).")
    failure_mode = ("Touch farming. Scaling by ball velocity change makes endless light taps nearly worthless, "
                    "but hard back-and-forth hits can still be farmed; decay this weight once the policy "
                    "contacts the ball reliably. It also under-rewards soft controlled touches (dribbles), so it "
                    "must not stay dominant when dribbling is trained.")

    def compute(self, agents, state, prev_state, ctx):
        out = np.zeros(len(agents), dtype=np.float64)
        touched = [i for i, agent in enumerate(agents) if state.cars[agent].ball_touches > 0]
        if touched:
            dv = np.asarray(state.ball.linear_velocity, dtype=np.float64) - np.asarray(
                prev_state.ball.linear_velocity, dtype=np.float64
            )
            value = min(1.0, float(np.linalg.norm(dv)) / self.reference_speed)
            for i in touched:
                out[i] = value
        return out


class VelocityToBall(RewardComponent):
    name = "velocity_to_ball"
    kind = "per_second"
    individual = True
    intent = "Early shaping: car speed towards the ball as a fraction of max car speed (-1..1 per second)."
    failure_mode = ("Ball chasing: rewards driving at the ball regardless of position, and penalises rotating "
                    "back or collecting boost. Keep it small and remove it once contact is reliable.")

    def compute(self, agents, state, prev_state, ctx):
        out = np.zeros(len(agents), dtype=np.float64)
        ball_pos = np.asarray(state.ball.position, dtype=np.float64)
        for i, agent in enumerate(agents):
            car = state.cars[agent]
            if car.is_demoed:
                continue
            to_ball = ball_pos - np.asarray(car.physics.position, dtype=np.float64)
            dist = float(np.linalg.norm(to_ball))
            if dist < 1e-6:
                continue  # coincident positions: direction undefined, contribute nothing rather than NaN
            vel = np.asarray(car.physics.linear_velocity, dtype=np.float64)
            out[i] = float(np.dot(vel, to_ball / dist)) / C.CAR_MAX_SPEED
        return out


class BallToGoal(RewardComponent):
    name = "ball_to_goal"
    kind = "per_second"
    individual = False
    intent = ("Ball speed towards the opponent goal as a fraction of max ball speed (-1..1 per second). "
              "Positive for the attacking team and negative for the defending team by construction.")
    failure_mode = ("Booming the ball downfield (giving away possession) scores well; it can discourage "
                    "controlled play and passing back. Reduce relative to goals as play matures.")

    def compute(self, agents, state, prev_state, ctx):
        out = np.zeros(len(agents), dtype=np.float64)
        ball_pos = np.asarray(state.ball.position, dtype=np.float64)
        ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float64)
        for i, agent in enumerate(agents):
            sign = 1.0 if state.cars[agent].team_num == C.BLUE_TEAM else -1.0
            goal = np.array([0.0, sign * C.BACK_NET_Y, C.GOAL_HEIGHT / 2.0])
            to_goal = goal - ball_pos
            dist = float(np.linalg.norm(to_goal))
            if dist < 1e-6:
                continue
            out[i] = float(np.dot(ball_vel, to_goal / dist)) / C.BALL_MAX_SPEED
        return out


class BoostEconomy(RewardComponent):
    name = "boost_economy"
    kind = "potential"
    individual = True
    intent = ("Potential sqrt(boost/100): collecting boost is rewarded most when the tank is low and spending "
              "the last of it costs the most, encouraging pickups without a separate pickup bonus.")
    failure_mode = ("Because it is a potential difference, burning boost and re-collecting nets zero, so it "
                    "cannot be farmed; but a large weight makes the policy hoard boost instead of using it. "
                    "Skipped across demolitions (respawn resets boost).")

    def compute(self, agents, state, prev_state, ctx):
        out = np.zeros(len(agents), dtype=np.float64)
        for i, agent in enumerate(agents):
            car = state.cars[agent]
            prev = prev_state.cars.get(agent)
            if prev is None or car.is_demoed or prev.is_demoed:
                continue
            now = max(0.0, float(car.boost_amount)) / 100.0
            before = max(0.0, float(prev.boost_amount)) / 100.0
            out[i] = np.sqrt(now) - np.sqrt(before)
        return out


COMPONENTS: Dict[str, type] = {
    cls.name: cls
    for cls in (GoalScored, GoalConceded, BallTouch, VelocityToBall, BallToGoal, BoostEconomy)
}


def make_components(weights: Dict[str, float]) -> List[tuple]:
    """Instantiate the configured components. Unknown names are a configuration error."""
    unknown = sorted(set(weights) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"unknown reward component(s) {unknown}; available: {sorted(COMPONENTS)}")
    return [(COMPONENTS[name](), float(weight)) for name, weight in weights.items()]
