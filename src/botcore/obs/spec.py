"""Versioned observation contract (obs-v1).

This module is the single source of truth for what the actor network sees. The same layout
is used by the simulator ObsBuilder and by the RLBot runner, and its hash is written into
every export manifest so a network can never be fed a different layout by accident.

Design rules
------------
* Actor inputs contain only quantities the RLBot v5 GamePacket provides or that the bot can
  reconstruct itself (its own previous action). No simulator-only state, no opponent inputs,
  no future trajectories.
* Team-relative convention: observations for orange players are rotated 180 degrees about
  the world Z axis (x -> -x, y -> -y). This is a proper rotation, not a reflection, so
  ordinary vectors and angular velocities transform identically and left/right controls keep
  their meaning. After the transform the agent always defends -y and attacks +y.
* No left/right mirroring is applied anywhere. A mirror is a reflection: it would need a sign
  flip on angular velocity (a pseudovector), a remap of steer/yaw/roll in the action table and
  a boost pad permutation. That is deliberately out of scope for obs-v1.
* All scaling is by fixed constants defined here. There is no learned/running normaliser, so
  nothing besides the network weights has to be exported.
* Absent teammates/opponents are explicit: each car slot starts with a presence flag and an
  absent slot is all zeros.

Changing anything in this file (feature order, sizes, scales, slot counts) changes OBS_SPEC_HASH
and makes existing checkpoints incompatible. Bump OBS_SPEC_VERSION when you do so on purpose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from botcore import constants as C

OBS_SPEC_VERSION = "obs-v1"

# Fixed scales (physical value is DIVIDED by these)
POS_SCALE = 4096.0  # isotropic so directions of relative vectors are preserved
LIN_VEL_SCALE = C.CAR_MAX_SPEED  # 2300 uu/s, used for cars, ball and relative velocities
CAR_ANG_VEL_SCALE = C.CAR_MAX_ANG_VEL  # 5.5 rad/s
BALL_ANG_VEL_SCALE = C.BALL_MAX_ANG_VEL  # 6 rad/s
BOOST_SCALE = 100.0
PAD_TIMER_SCALE = C.BIG_PAD_RECHARGE_SECONDS  # 10 s
DEMO_TIMER_SCALE = C.DEMO_RESPAWN_SECONDS  # 3 s

N_TEAMMATE_SLOTS = C.MAX_TEAMMATES  # 2
N_OPPONENT_SLOTS = C.MAX_OPPONENTS  # 3
N_CAR_SLOTS = N_TEAMMATE_SLOTS + N_OPPONENT_SLOTS  # teammates first, then opponents

_WORLD = "team-relative world frame (orange rotated 180 deg about Z)"
_LOCAL = "controlled car's local frame (components along its forward, lateral and up axes)"
_NA = "n/a (scalar)"
_PKT = "direct from GamePacket via rlgym-compat"
_RESET_NONE = "stateless: recomputed from the current state every decision"


@dataclass(frozen=True)
class Feature:
    name: str
    size: int
    units: str
    scale: str
    frame: str
    deploy: str  # availability at deployment and reconstruction assumptions
    missing: str  # missing-value handling
    reset: str  # reset behaviour


def _f(name, size, units, scale, frame, deploy=_PKT, missing="always present", reset=_RESET_NONE) -> Feature:
    return Feature(name, size, units, scale, frame, deploy, missing, reset)


_DEMO_ZERO = "zeroed while the car is demolished (sim and game report different placeholder poses)"

SELF_FEATURES: tuple[Feature, ...] = (
    _f("pos", 3, "uu", f"/{POS_SCALE:g}", _WORLD, missing=_DEMO_ZERO),
    _f("forward", 3, "unit vector", "none", _WORLD,
       deploy="from packet rotator via rlgym-compat euler_to_rotation", missing=_DEMO_ZERO),
    _f("up", 3, "unit vector", "none", _WORLD,
       deploy="from packet rotator via rlgym-compat euler_to_rotation", missing=_DEMO_ZERO),
    _f("lin_vel", 3, "uu/s", f"/{LIN_VEL_SCALE:g}", _WORLD, missing=_DEMO_ZERO),
    _f("ang_vel", 3, "rad/s", f"/{CAR_ANG_VEL_SCALE:g}", _WORLD, missing=_DEMO_ZERO),
    _f("lin_vel_local", 3, "uu/s", f"/{LIN_VEL_SCALE:g}", _LOCAL, missing=_DEMO_ZERO),
    _f("ang_vel_local", 3, "rad/s", f"/{CAR_ANG_VEL_SCALE:g}", _LOCAL, missing=_DEMO_ZERO),
    _f("boost", 1, "boost 0-100", f"/{BOOST_SCALE:g}", _NA),
    _f("on_ground", 1, "bool", "0/1", _NA,
       deploy="sim: >=3 wheels in contact. Game: AirState.OnGround, or Jumping within 6 ticks of the jump "
              "(rlgym-compat heuristic). Can disagree for a few ticks around take-off/landing and when a "
              "car rests on fewer than 3 wheels."),
    _f("has_jumped", 1, "bool", "0/1", _NA),
    _f("has_double_jumped", 1, "bool", "0/1", _NA),
    _f("has_flipped", 1, "bool", "0/1", _NA, deploy="packet has_dodged"),
    _f("flip_available", 1, "bool", "0/1", _NA,
       deploy="derived: not double-jumped, not flipped and the 1.25 s window has not expired. In the game the "
              "window comes from dodge_timeout; botcore patches rlgym-compat's stale value once dodge_timeout "
              "returns to -1 in the air (see deploy/state_fixes.py)."),
    _f("flip_window", 1, "seconds remaining", f"/{C.DOUBLEJUMP_MAX_DELAY:g}, clipped to [0,1]", _NA,
       deploy="1 - air_time_since_jump/1.25; 1 on the ground or after a flip reset, 0 when no flip is available"),
    _f("is_flipping", 1, "bool", "0/1", _NA,
       deploy="has_dodged and dodge_elapsed < 0.65 s"),
    _f("is_demoed", 1, "bool", "0/1", _NA, deploy="demolished_timeout != -1"),
    _f("demo_timer", 1, "seconds until respawn", f"/{DEMO_TIMER_SCALE:g}, clipped to [0,1]", _NA,
       deploy="demolished_timeout", missing="0 when alive"),
    _f("is_supersonic", 1, "bool", "0/1", _NA),
    _f("prev_action", C.NUM_CONTROLS, "controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)",
       "none (already in [-1,1] / {0,1})", _NA,
       deploy="the bot's own last chosen action, tracked by the action parser (never read from the packet)",
       missing="zeros before the first action",
       reset="zeroed on every environment reset, and on kickoff/goal resets and on-the-fly resets in the RLBot runner"),
)

BALL_FEATURES: tuple[Feature, ...] = (
    _f("pos", 3, "uu", f"/{POS_SCALE:g}", _WORLD),
    _f("lin_vel", 3, "uu/s", f"/{LIN_VEL_SCALE:g}", _WORLD),
    _f("ang_vel", 3, "rad/s", f"/{BALL_ANG_VEL_SCALE:g}", _WORLD),
    _f("rel_pos", 3, "uu (ball - self)", f"/{POS_SCALE:g}", _WORLD, missing=_DEMO_ZERO),
    _f("rel_vel", 3, "uu/s (ball - self)", f"/{LIN_VEL_SCALE:g}", _WORLD, missing=_DEMO_ZERO),
    _f("rel_pos_local", 3, "uu (ball - self)", f"/{POS_SCALE:g}", _LOCAL, missing=_DEMO_ZERO),
    _f("rel_vel_local", 3, "uu/s (ball - self)", f"/{LIN_VEL_SCALE:g}", _LOCAL, missing=_DEMO_ZERO),
    _f("distance", 1, "uu", f"/{POS_SCALE:g}", _NA,
       missing="0 while self is demolished; exactly 0 (never NaN) for coincident positions"),
)

_ABSENT = "all zeros when the slot is empty (present = 0)"
_OTHER_DEMO = "zeroed while that car is demolished"

CAR_FEATURES: tuple[Feature, ...] = (
    _f("present", 1, "bool", "0/1", _NA, deploy="one slot per car in packet.players other than self",
       missing="0 marks an empty slot; every other value in the slot is then 0"),
    _f("is_teammate", 1, "bool", "0/1", _NA, deploy="packet team vs own team", missing=_ABSENT),
    _f("pos", 3, "uu", f"/{POS_SCALE:g}", _WORLD, missing=f"{_ABSENT}; {_OTHER_DEMO}"),
    _f("forward", 3, "unit vector", "none", _WORLD, missing=f"{_ABSENT}; {_OTHER_DEMO}"),
    _f("up", 3, "unit vector", "none", _WORLD, missing=f"{_ABSENT}; {_OTHER_DEMO}"),
    _f("lin_vel", 3, "uu/s", f"/{LIN_VEL_SCALE:g}", _WORLD, missing=f"{_ABSENT}; {_OTHER_DEMO}"),
    _f("ang_vel", 3, "rad/s", f"/{CAR_ANG_VEL_SCALE:g}", _WORLD, missing=f"{_ABSENT}; {_OTHER_DEMO}"),
    _f("boost", 1, "boost 0-100", f"/{BOOST_SCALE:g}", _NA,
       deploy="packet reports every player's boost (true information in offline matches)", missing=_ABSENT),
    _f("on_ground", 1, "bool", "0/1", _NA, deploy="same heuristic as self.on_ground", missing=_ABSENT),
    _f("flip_available", 1, "bool", "0/1", _NA, deploy="same derivation as self.flip_available", missing=_ABSENT),
    _f("is_demoed", 1, "bool", "0/1", _NA, missing=_ABSENT),
    _f("is_supersonic", 1, "bool", "0/1", _NA, missing=_ABSENT),
    _f("rel_pos", 3, "uu (car - self)", f"/{POS_SCALE:g}", _WORLD, missing=f"{_ABSENT}; zero if either car is demolished"),
    _f("rel_vel", 3, "uu/s (car - self)", f"/{LIN_VEL_SCALE:g}", _WORLD, missing=f"{_ABSENT}; zero if either car is demolished"),
    _f("rel_pos_local", 3, "uu (car - self)", f"/{POS_SCALE:g}", _LOCAL, missing=f"{_ABSENT}; zero if either car is demolished"),
    _f("distance", 1, "uu", f"/{POS_SCALE:g}", _NA, missing=f"{_ABSENT}; zero if either car is demolished"),
    _f("ball_rel_pos", 3, "uu (ball - car)", f"/{POS_SCALE:g}", _WORLD, missing=f"{_ABSENT}; {_OTHER_DEMO}"),
    _f("ball_distance", 1, "uu", f"/{POS_SCALE:g}", _NA, missing=f"{_ABSENT}; {_OTHER_DEMO}"),
)

PAD_FEATURES: tuple[Feature, ...] = (
    _f("pad_timers", C.NUM_BOOST_PADS, "seconds until the pad is available (0 = available)",
       f"/{PAD_TIMER_SCALE:g}", "pad order = BOOST_LOCATIONS for blue, reversed for orange (the reversed order is "
       "exactly the 180 deg rotated layout; verified against RocketSim 2.2.1)",
       deploy="reconstructed by the runner from is_active transitions and the known 4 s / 10 s cooldowns "
              "(deploy/state_fixes.PadTimerTracker), so it does not depend on which meaning BoostPadState.timer "
              "has in a given RLBot build. A pad already inactive when the bot first sees it uses the packet "
              "timer clipped to the cooldown.",
       missing="0 (available) if a pad is missing from the packet"),
)


def _block_size(features: tuple[Feature, ...]) -> int:
    return sum(f.size for f in features)


def _block_slices(features: tuple[Feature, ...]) -> dict[str, slice]:
    out: dict[str, slice] = {}
    offset = 0
    for f in features:
        if f.name in out:
            raise ValueError(f"duplicate feature name {f.name!r}")
        out[f.name] = slice(offset, offset + f.size)
        offset += f.size
    return out


SELF_SIZE = _block_size(SELF_FEATURES)
BALL_SIZE = _block_size(BALL_FEATURES)
CAR_SIZE = _block_size(CAR_FEATURES)
PAD_SIZE = _block_size(PAD_FEATURES)

SELF_IDX = _block_slices(SELF_FEATURES)
BALL_IDX = _block_slices(BALL_FEATURES)
CAR_IDX = _block_slices(CAR_FEATURES)


@dataclass(frozen=True)
class ObsLayout:
    """Offsets of each block in the flat float32 observation vector."""

    version: str
    self_size: int
    ball_size: int
    car_size: int
    n_teammate_slots: int
    n_opponent_slots: int
    pad_size: int

    @property
    def n_car_slots(self) -> int:
        return self.n_teammate_slots + self.n_opponent_slots

    @property
    def self_slice(self) -> slice:
        return slice(0, self.self_size)

    @property
    def ball_slice(self) -> slice:
        return slice(self.self_size, self.self_size + self.ball_size)

    @property
    def cars_slice(self) -> slice:
        start = self.self_size + self.ball_size
        return slice(start, start + self.n_car_slots * self.car_size)

    @property
    def pads_slice(self) -> slice:
        start = self.cars_slice.stop
        return slice(start, start + self.pad_size)

    @property
    def size(self) -> int:
        return self.pads_slice.stop

    def car_slot_slice(self, slot: int) -> slice:
        if not 0 <= slot < self.n_car_slots:
            raise IndexError(f"car slot {slot} out of range (0..{self.n_car_slots - 1})")
        start = self.cars_slice.start + slot * self.car_size
        return slice(start, start + self.car_size)


LAYOUT = ObsLayout(
    version=OBS_SPEC_VERSION,
    self_size=SELF_SIZE,
    ball_size=BALL_SIZE,
    car_size=CAR_SIZE,
    n_teammate_slots=N_TEAMMATE_SLOTS,
    n_opponent_slots=N_OPPONENT_SLOTS,
    pad_size=PAD_SIZE,
)

OBS_SIZE = LAYOUT.size


def spec_dict() -> dict:
    """Canonical description of the contract; hashed for the manifest."""
    return {
        "version": OBS_SPEC_VERSION,
        "scales": {
            "pos": POS_SCALE,
            "lin_vel": LIN_VEL_SCALE,
            "car_ang_vel": CAR_ANG_VEL_SCALE,
            "ball_ang_vel": BALL_ANG_VEL_SCALE,
            "boost": BOOST_SCALE,
            "pad_timer": PAD_TIMER_SCALE,
            "demo_timer": DEMO_TIMER_SCALE,
        },
        "slots": {"teammates": N_TEAMMATE_SLOTS, "opponents": N_OPPONENT_SLOTS},
        "blocks": {
            "self": [[f.name, f.size] for f in SELF_FEATURES],
            "ball": [[f.name, f.size] for f in BALL_FEATURES],
            "car": [[f.name, f.size] for f in CAR_FEATURES],
            "pads": [[f.name, f.size] for f in PAD_FEATURES],
        },
        "size": OBS_SIZE,
    }


def spec_hash() -> str:
    payload = json.dumps(spec_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


OBS_SPEC_HASH = spec_hash()


def render_markdown() -> str:
    """Feature-by-feature table used to generate docs/OBSERVATION_SPEC.md."""
    lines = [
        f"# Observation specification `{OBS_SPEC_VERSION}`",
        "",
        "Generated from `src/botcore/obs/spec.py` by `python -m botcore obs-doc`. Do not edit by hand.",
        "",
        f"- Flat `float32` vector of length **{OBS_SIZE}**; spec hash `{OBS_SPEC_HASH}`.",
        f"- Layout: self `[0:{LAYOUT.self_slice.stop}]`, ball `[{LAYOUT.ball_slice.start}:{LAYOUT.ball_slice.stop}]`, "
        f"cars `[{LAYOUT.cars_slice.start}:{LAYOUT.cars_slice.stop}]` ({LAYOUT.n_car_slots} slots x {CAR_SIZE}; "
        f"slots 0-{N_TEAMMATE_SLOTS - 1} teammates, {N_TEAMMATE_SLOTS}-{N_CAR_SLOTS - 1} opponents), "
        f"pads `[{LAYOUT.pads_slice.start}:{LAYOUT.pads_slice.stop}]`.",
        "- Team-relative: orange observations are rotated 180 degrees about Z (a rotation, never a reflection).",
        "- Within a team, cars fill slots in the stable order of the state's car dictionary "
        "(RLGym agent order in the simulator, packet player order in RLBot). The order is arbitrary but fixed "
        "for a match; a slot-order-sensitive model (the MLP) therefore needs slot shuffling or the entity "
        "encoder before 2v2/3v3 training.",
        "",
    ]
    for title, feats, base in (
        ("Self block", SELF_FEATURES, LAYOUT.self_slice.start),
        ("Ball block", BALL_FEATURES, LAYOUT.ball_slice.start),
        (f"Car slot block (repeated {N_CAR_SLOTS} times; offsets shown for slot 0)", CAR_FEATURES, LAYOUT.cars_slice.start),
        ("Boost pad block", PAD_FEATURES, LAYOUT.pads_slice.start),
    ):
        lines += [f"## {title}", "",
                  "| Index | Feature | Units | Scaling | Frame | Deployment availability / reconstruction | Missing values | Reset behaviour |",
                  "|---|---|---|---|---|---|---|---|"]
        offset = base
        for f in feats:
            idx = f"{offset}" if f.size == 1 else f"{offset}-{offset + f.size - 1}"
            row = asdict(f)
            lines.append(
                f"| {idx} | `{row['name']}` | {row['units']} | {row['scale']} | {row['frame']} | "
                f"{row['deploy']} | {row['missing']} | {row['reset']} |"
            )
            offset += f.size
        lines.append("")
    return "\n".join(lines)
