"""Build synthetic RLBot v5 GamePackets from RocketSim/RLGym states.

Used for observation-parity tests and for latency benchmarks of the deployment pipeline without a running
game. The mapping follows the RLBot schema documentation (rlbot_flatbuffers 0.18.x) and rlgym-compat's
inverse conversions. It is an approximation of what Rocket League sends: values RocketSim does not track
in the same way (e.g. AirState transitions, dodge_timeout after a flip) are filled with the documented
semantics, so this cannot prove that the real game reports identical numbers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from rlbot import flat
from rlgym.rocket_league.common_values import BOOST_LOCATIONS

from botcore import constants as C

OCTANE_HITBOX = flat.BoxShape(length=118.00738, width=84.19941, height=36.159073)
OCTANE_OFFSET = flat.Vector3(x=13.87566, y=0.0, z=20.754988)


def _v3(a) -> flat.Vector3:
    return flat.Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2]))


def _physics(phys) -> flat.Physics:
    pitch, yaw, roll = (float(v) for v in phys.euler_angles)
    return flat.Physics(location=_v3(phys.position), rotation=flat.Rotator(pitch=pitch, yaw=yaw, roll=roll),
                        velocity=_v3(phys.linear_velocity), angular_velocity=_v3(phys.angular_velocity))


def field_info(pad_order: Optional[Sequence[int]] = None) -> flat.FieldInfo:
    """FieldInfo with the standard pads, optionally in a different order (RLBot's order differs from RLGym's)."""
    order = list(range(C.NUM_BOOST_PADS)) if pad_order is None else list(pad_order)
    pads = [flat.BoostPad(location=_v3(BOOST_LOCATIONS[i]), is_full_boost=i in C.BIG_PAD_INDICES) for i in order]
    return flat.FieldInfo(boost_pads=pads, goals=[])


def _air_state(car) -> flat.AirState:
    if car.on_ground and not car.is_jumping:
        return flat.AirState.OnGround
    if car.is_jumping:
        return flat.AirState.Jumping
    if car.has_flipped and car.flip_time < C.FLIP_TORQUE_TIME:
        return flat.AirState.Dodging
    return flat.AirState.InAir


def _dodge_timeout(car) -> float:
    if car.on_ground and not car.is_jumping:
        return -1.0
    if not car.has_jumped:
        return -1.0  # fell off a surface: flip available indefinitely, timer not running
    if car.has_flipped or car.has_double_jumped:
        return -1.0
    remaining = C.DOUBLEJUMP_MAX_DELAY - float(car.air_time_since_jump)
    return remaining if remaining > 0 else -1.0


def player_info(car, player_id: int) -> flat.PlayerInfo:
    torque = np.asarray(car.flip_torque, dtype=np.float64)
    return flat.PlayerInfo(
        physics=_physics(car.physics),
        score_info=flat.ScoreInfo(),
        hitbox=OCTANE_HITBOX,
        hitbox_offset=OCTANE_OFFSET,
        latest_touch=None,
        air_state=_air_state(car),
        dodge_timeout=_dodge_timeout(car),
        demolished_timeout=float(car.demo_respawn_timer) if car.is_demoed else -1.0,
        is_supersonic=bool(car.is_supersonic),
        is_bot=True,
        name=f"player{player_id}",
        team=int(car.team_num),
        boost=float(car.boost_amount),
        player_id=int(player_id),
        accolades=[],
        last_input=flat.ControllerState(jump=bool(car.is_holding_jump), boost=bool(car.is_boosting),
                                        handbrake=bool(car.handbrake > 0)),
        has_jumped=bool(car.has_jumped),
        has_double_jumped=bool(car.has_double_jumped),
        has_dodged=bool(car.has_flipped),
        dodge_elapsed=float(car.flip_time),
        dodge_dir=flat.Vector2(x=float(torque[1]), y=float(-torque[0])),
    )


def state_to_packet(state, frame: int, player_ids: Dict, pad_order: Optional[Sequence[int]] = None,
                    phase=flat.MatchPhase.Active) -> flat.GamePacket:
    """`player_ids` maps RLGym agent id -> RLBot player id; players are emitted in that dict's order."""
    order = list(range(C.NUM_BOOST_PADS)) if pad_order is None else list(pad_order)
    timers = np.asarray(state.boost_pad_timers, dtype=np.float64)
    pads = [flat.BoostPadState(is_active=bool(timers[i] <= 0.0), timer=float(timers[i])) for i in order]
    players: List[flat.PlayerInfo] = [player_info(state.cars[a], pid) for a, pid in player_ids.items()]
    ball = flat.BallInfo(physics=_physics(state.ball))
    return flat.GamePacket(
        players=players,
        boost_pads=pads,
        balls=[ball],
        match_info=flat.MatchInfo(seconds_elapsed=frame / C.TICKS_PER_SECOND, frame_num=int(frame),
                                  match_phase=phase, world_gravity_z=-650.0, game_speed=1.0),
        teams=[flat.TeamInfo(team_index=0), flat.TeamInfo(team_index=1)],
    )


class CompatObsPipeline:
    """The deployment observation path without sockets: rlgym-compat GameState + botcore reconstruction.

    BotCoreBot feeds the same objects (rlgym-compat state, PadTimerTracker, flip expiry) into the same
    SpecObsBuilder; this class exists so that path can be exercised with recorded or synthetic packets.
    """

    def __init__(self, field: flat.FieldInfo):
        from rlgym_compat import GameState

        from botcore.deploy.state_fixes import PadTimerTracker

        self.state = GameState.create_compat_game_state(field_info=field)
        self.pads = PadTimerTracker(field.boost_pads)
        self.expired: set = set()

    def update(self, packet: flat.GamePacket) -> None:
        from botcore.deploy.state_fixes import flip_expired_ids

        self.state.update(packet)
        self.pads.update(packet)
        self.expired = flip_expired_ids(packet)

    def obs(self, agent_id, prev_action=None) -> np.ndarray:
        from botcore.obs import build_obs

        return build_obs(self.state, agent_id, prev_action, self.pads.timers(), flip_expired=self.expired)
