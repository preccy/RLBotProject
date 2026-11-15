"""Observation builder implementing the obs-v1 contract.

`build_obs` is a pure function of (state, agent, previous action, optional pad timers) and only
touches attributes that exist on BOTH `rlgym.rocket_league.api.GameState` (simulator) and
`rlgym_compat.GameState` (RLBot runner), so the exact same code produces training and
deployment observations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rlgym.api import ObsBuilder

from botcore import constants as C
from botcore.obs import spec as S

PREV_ACTIONS_KEY = "prev_actions"  # shared_info: {agent_id: np.ndarray(8)} written by the action parser
PAD_TIMERS_KEY = "pad_timers_override"  # shared_info: np.ndarray(34) in BOOST_LOCATIONS order (deployment only)
FLIP_EXPIRED_KEY = "flip_expired"  # shared_info: set of agent ids whose flip window has expired (deployment only)

_ZERO3 = np.zeros(3, dtype=np.float32)


class UnsupportedPlayerCountError(ValueError):
    """Raised when a state has more teammates/opponents than the observation contract has slots for."""


def _flip_window(car, expired: bool = False) -> float:
    """Fraction of the flip/double-jump window remaining, 0 if no flip is available.

    `expired` comes from the RLBot packet (dodge_timeout back to -1 in the air after a jump). rlgym-compat keeps
    the last air_time_since_jump in that case, which would otherwise report a tiny but non-zero window forever.
    """
    if expired or car.has_double_jumped or car.has_flipped:
        return 0.0
    remaining = 1.0 - float(car.air_time_since_jump) / C.DOUBLEJUMP_MAX_DELAY
    return float(min(1.0, max(0.0, remaining)))


def _physics(car, inverted: bool):
    return car.inverted_physics if inverted else car.physics


def build_obs(
    state,
    agent_id,
    prev_action: Optional[np.ndarray] = None,
    pad_timers: Optional[np.ndarray] = None,
    fill: float = 0.0,
    flip_expired: Optional[set] = None,
) -> np.ndarray:
    """Build the obs-v1 vector for one agent.

    :param state: rlgym or rlgym-compat GameState.
    :param agent_id: key of the controlled car in ``state.cars``.
    :param prev_action: the 8 controls this agent applied on its previous decision (zeros if None).
    :param pad_timers: optional replacement for ``state.boost_pad_timers`` (seconds until available,
        BOOST_LOCATIONS order). Used by the RLBot runner, which reconstructs timers itself.
    :param fill: initial value of the buffer. Tests pass NaN to prove every index is written.
    :param flip_expired: agent ids whose flip window the game reports as expired (deployment only).
    """
    flip_expired = flip_expired or set()
    layout = S.LAYOUT
    obs = np.full(layout.size, fill, dtype=np.float32)

    me = state.cars[agent_id]
    inverted = me.team_num == C.ORANGE_TEAM
    ball = state.inverted_ball if inverted else state.ball

    # ---------------------------------------------------------------- self
    sv = obs[layout.self_slice]
    si = S.SELF_IDX
    me_alive = not me.is_demoed
    if me_alive:
        p = _physics(me, inverted)
        my_pos = np.asarray(p.position, dtype=np.float32)
        my_rot = np.asarray(p.rotation_mtx, dtype=np.float32)
        my_vel = np.asarray(p.linear_velocity, dtype=np.float32)
        my_ang = np.asarray(p.angular_velocity, dtype=np.float32)
        sv[si["pos"]] = my_pos / S.POS_SCALE
        sv[si["forward"]] = my_rot[:, 0]
        sv[si["up"]] = my_rot[:, 2]
        sv[si["lin_vel"]] = my_vel / S.LIN_VEL_SCALE
        sv[si["ang_vel"]] = my_ang / S.CAR_ANG_VEL_SCALE
        sv[si["lin_vel_local"]] = (my_rot.T @ my_vel) / S.LIN_VEL_SCALE
        sv[si["ang_vel_local"]] = (my_rot.T @ my_ang) / S.CAR_ANG_VEL_SCALE
    else:
        my_pos = my_vel = _ZERO3
        my_rot = None
        for key in ("pos", "forward", "up", "lin_vel", "ang_vel", "lin_vel_local", "ang_vel_local"):
            sv[si[key]] = 0.0

    window = _flip_window(me, agent_id in flip_expired)
    sv[si["boost"]] = float(me.boost_amount) / S.BOOST_SCALE
    sv[si["on_ground"]] = float(me.on_ground)
    sv[si["has_jumped"]] = float(me.has_jumped)
    sv[si["has_double_jumped"]] = float(me.has_double_jumped)
    sv[si["has_flipped"]] = float(me.has_flipped)
    sv[si["flip_available"]] = float(window > 0.0)
    sv[si["flip_window"]] = window
    sv[si["is_flipping"]] = float(me.is_flipping)
    sv[si["is_demoed"]] = float(me.is_demoed)
    sv[si["demo_timer"]] = min(1.0, max(0.0, float(me.demo_respawn_timer) / S.DEMO_TIMER_SCALE))
    sv[si["is_supersonic"]] = float(me.is_supersonic)
    if prev_action is None:
        sv[si["prev_action"]] = 0.0
    else:
        prev = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        if prev.shape[0] != C.NUM_CONTROLS:
            raise ValueError(f"prev_action must have {C.NUM_CONTROLS} values, got shape {prev.shape}")
        sv[si["prev_action"]] = prev

    # ---------------------------------------------------------------- ball
    bv = obs[layout.ball_slice]
    bi = S.BALL_IDX
    ball_pos = np.asarray(ball.position, dtype=np.float32)
    ball_vel = np.asarray(ball.linear_velocity, dtype=np.float32)
    bv[bi["pos"]] = ball_pos / S.POS_SCALE
    bv[bi["lin_vel"]] = ball_vel / S.LIN_VEL_SCALE
    bv[bi["ang_vel"]] = np.asarray(ball.angular_velocity, dtype=np.float32) / S.BALL_ANG_VEL_SCALE
    if me_alive:
        rel_pos = ball_pos - my_pos
        rel_vel = ball_vel - my_vel
        bv[bi["rel_pos"]] = rel_pos / S.POS_SCALE
        bv[bi["rel_vel"]] = rel_vel / S.LIN_VEL_SCALE
        bv[bi["rel_pos_local"]] = (my_rot.T @ rel_pos) / S.POS_SCALE
        bv[bi["rel_vel_local"]] = (my_rot.T @ rel_vel) / S.LIN_VEL_SCALE
        bv[bi["distance"]] = float(np.sqrt(np.dot(rel_pos, rel_pos))) / S.POS_SCALE
    else:
        for key in ("rel_pos", "rel_vel", "rel_pos_local", "rel_vel_local", "distance"):
            bv[bi[key]] = 0.0

    # ---------------------------------------------------------------- other cars
    cars_view = obs[layout.cars_slice]
    cars_view[:] = 0.0  # absent slots are all zeros, including the presence flag
    ci = S.CAR_IDX
    n_mates = 0
    n_opps = 0
    for other_id, other in state.cars.items():
        if other_id == agent_id:
            continue
        teammate = other.team_num == me.team_num
        if teammate:
            if n_mates >= S.N_TEAMMATE_SLOTS:
                raise UnsupportedPlayerCountError(
                    f"{S.OBS_SPEC_VERSION} supports at most {S.N_TEAMMATE_SLOTS} teammates ({C.MAX_TEAM_SIZE}v{C.MAX_TEAM_SIZE})"
                )
            slot = n_mates
            n_mates += 1
        else:
            if n_opps >= S.N_OPPONENT_SLOTS:
                raise UnsupportedPlayerCountError(
                    f"{S.OBS_SPEC_VERSION} supports at most {S.N_OPPONENT_SLOTS} opponents ({C.MAX_TEAM_SIZE}v{C.MAX_TEAM_SIZE})"
                )
            slot = S.N_TEAMMATE_SLOTS + n_opps
            n_opps += 1

        cv = cars_view[slot * S.CAR_SIZE:(slot + 1) * S.CAR_SIZE]
        cv[ci["present"]] = 1.0
        cv[ci["is_teammate"]] = float(teammate)
        cv[ci["boost"]] = float(other.boost_amount) / S.BOOST_SCALE
        cv[ci["on_ground"]] = float(other.on_ground)
        cv[ci["flip_available"]] = float(_flip_window(other, other_id in flip_expired) > 0.0)
        cv[ci["is_demoed"]] = float(other.is_demoed)
        cv[ci["is_supersonic"]] = float(other.is_supersonic)
        if other.is_demoed:
            continue  # kinematic and relative features stay zero
        op = _physics(other, inverted)
        o_pos = np.asarray(op.position, dtype=np.float32)
        o_rot = np.asarray(op.rotation_mtx, dtype=np.float32)
        o_vel = np.asarray(op.linear_velocity, dtype=np.float32)
        cv[ci["pos"]] = o_pos / S.POS_SCALE
        cv[ci["forward"]] = o_rot[:, 0]
        cv[ci["up"]] = o_rot[:, 2]
        cv[ci["lin_vel"]] = o_vel / S.LIN_VEL_SCALE
        cv[ci["ang_vel"]] = np.asarray(op.angular_velocity, dtype=np.float32) / S.CAR_ANG_VEL_SCALE
        to_ball = ball_pos - o_pos
        cv[ci["ball_rel_pos"]] = to_ball / S.POS_SCALE
        cv[ci["ball_distance"]] = float(np.sqrt(np.dot(to_ball, to_ball))) / S.POS_SCALE
        if me_alive:
            rel = o_pos - my_pos
            cv[ci["rel_pos"]] = rel / S.POS_SCALE
            cv[ci["rel_vel"]] = (o_vel - my_vel) / S.LIN_VEL_SCALE
            cv[ci["rel_pos_local"]] = (my_rot.T @ rel) / S.POS_SCALE
            cv[ci["distance"]] = float(np.sqrt(np.dot(rel, rel))) / S.POS_SCALE

    # ---------------------------------------------------------------- boost pads
    timers = state.boost_pad_timers if pad_timers is None else pad_timers
    timers = np.asarray(timers, dtype=np.float32)
    if timers.shape[0] != C.NUM_BOOST_PADS:
        raise ValueError(
            f"expected {C.NUM_BOOST_PADS} boost pads (standard soccar map), got {timers.shape[0]}"
        )
    if inverted:
        timers = timers[::-1]
    obs[layout.pads_slice] = np.clip(timers / S.PAD_TIMER_SCALE, 0.0, 1.0)
    return obs


class SpecObsBuilder(ObsBuilder[Any, np.ndarray, Any, Tuple[str, int]]):
    """RLGym v2 ObsBuilder for the obs-v1 contract (works in the simulator and under rlgym-rlbot)."""

    def get_obs_space(self, agent) -> Tuple[str, int]:
        return "real", S.OBS_SIZE

    def reset(self, agents: List[Any], initial_state, shared_info: Dict[str, Any]) -> None:
        # Stateless. The previous action lives in shared_info and is cleared by the action parser's reset.
        pass

    def build_obs(self, agents: List[Any], state, shared_info: Dict[str, Any]) -> Dict[Any, np.ndarray]:
        prev_actions = shared_info.get(PREV_ACTIONS_KEY) or {}
        pad_timers = shared_info.get(PAD_TIMERS_KEY)
        flip_expired = shared_info.get(FLIP_EXPIRED_KEY)
        return {
            agent: build_obs(state, agent, prev_actions.get(agent), pad_timers, flip_expired=flip_expired)
            for agent in agents
        }
