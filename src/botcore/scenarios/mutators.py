"""Seeded, reproducible reset-state generation.

One `ScenarioMutator` creates the cars and then applies one scenario drawn from a weighted mixture.
All randomness comes from a single `numpy.random.Generator` owned by the mutator, so a given
(seed, reset index) always yields the same state regardless of any global RNG use.

Scenarios are written in "protagonist-relative" coordinates (the protagonist defends -y and attacks
+y) and the protagonist is blue or orange with equal probability; for orange the whole layout is
rotated 180 degrees about Z. Both colours therefore see every role equally often.

Only physically reachable states are generated:
* grounded cars are upright at the octane resting height with `on_ground=True`, velocity along heading;
* airborne cars have `on_ground=False` and `has_jumped=False` - the state of a car that drove off a
  wall or was bumped, which legitimately still has its flip. No scenario sets has_flipped,
  has_double_jumped, is_jumping or mid-flip torque, because RocketSim cannot faithfully start mid-flip.

Kickoff spawn positions/yaws are the standard five (values as in rlgym.rocket_league
state_mutators.KickoffMutator, rlgym-rocket-league 2.0.1, Apache-2.0).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import Car, GameState, PhysicsObject

from botcore import constants as C

SCENARIO_KEY = "scenario"  # shared_info: name of the scenario used for the current episode

_KICKOFF_POS = np.array(
    [[-2048, -2560, 17], [2048, -2560, 17], [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]], dtype=np.float32
)
_KICKOFF_YAW = np.array([0.25 * np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi])

CAR_X_LIMIT = 3500.0
CAR_Y_LIMIT = 4500.0
MIN_CAR_BALL_DIST = 300.0
MIN_CAR_CAR_DIST = 350.0


def _new_car(team: int) -> Car:
    car = Car()
    car.team_num = team
    car.hitbox_type = 0  # octane
    car.ball_touches = 0
    car.bump_victim_id = None
    car.physics = PhysicsObject()
    car.physics.position = np.zeros(3, dtype=np.float32)
    car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
    car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
    car.physics.euler_angles = np.zeros(3, dtype=np.float32)
    car.demo_respawn_timer = 0.0
    car.on_ground = True
    car.supersonic_time = 0.0
    car.boost_amount = 0.0
    car.boost_active_time = 0.0
    car.handbrake = 0.0
    car.has_jumped = False
    car.is_holding_jump = False
    car.is_jumping = False
    car.jump_time = 0.0
    car.has_flipped = False
    car.has_double_jumped = False
    car.air_time_since_jump = 0.0
    car.flip_time = 0.0
    car.flip_torque = np.zeros(3, dtype=np.float32)
    car.is_autoflipping = False
    car.autoflip_timer = 0.0
    car.autoflip_direction = 0.0
    return car


class Frame:
    """Maps protagonist-relative coordinates to world coordinates (identity for blue, 180 deg yaw for orange)."""

    def __init__(self, protagonist_team: int):
        self.team = protagonist_team
        self.sign = 1.0 if protagonist_team == C.BLUE_TEAM else -1.0

    def vec(self, x: float, y: float, z: float) -> np.ndarray:
        return np.array([self.sign * x, self.sign * y, z], dtype=np.float32)

    def yaw(self, yaw: float) -> float:
        out = yaw if self.sign > 0 else yaw + np.pi
        return float((out + np.pi) % (2 * np.pi) - np.pi)


def _set_ground_car(car: Car, frame: Frame, x: float, y: float, yaw: float, speed: float, boost: float) -> None:
    world_yaw = frame.yaw(yaw)
    car.physics.position = frame.vec(x, y, C.CAR_RESTING_HEIGHT)
    car.physics.euler_angles = np.array([0.0, world_yaw, 0.0], dtype=np.float32)
    car.physics.linear_velocity = np.array(
        [speed * np.cos(world_yaw), speed * np.sin(world_yaw), 0.0], dtype=np.float32
    )
    car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
    car.on_ground = True
    car.boost_amount = float(boost)


def _set_ball(state: GameState, frame: Frame, pos: Tuple[float, float, float], vel: Tuple[float, float, float]) -> None:
    state.ball.position = frame.vec(*pos)
    state.ball.linear_velocity = frame.vec(*vel)
    state.ball.angular_velocity = np.zeros(3, dtype=np.float32)


def _random_ground_pose(rng: np.random.Generator, y_lo: float = -CAR_Y_LIMIT, y_hi: float = CAR_Y_LIMIT):
    return (
        float(rng.uniform(-CAR_X_LIMIT, CAR_X_LIMIT)),
        float(rng.uniform(y_lo, y_hi)),
        float(rng.uniform(-np.pi, np.pi)),
    )


def _teams(state: GameState, frame: Frame) -> Tuple[List[Car], List[Car]]:
    mine = [c for c in state.cars.values() if c.team_num == frame.team]
    theirs = [c for c in state.cars.values() if c.team_num != frame.team]
    return mine, theirs


def _separate(state: GameState, rng: np.random.Generator, fixed: List[Car]) -> None:
    """Re-place grounded cars that overlap the ball or another car. `fixed` cars are never moved."""
    ball_xy = np.asarray(state.ball.position[:2], dtype=np.float64)
    ball_low = float(state.ball.position[2]) < 400.0
    placed: List[Car] = list(fixed)
    for car in state.cars.values():
        if any(car is f for f in fixed):
            continue
        for _ in range(50):
            xy = np.asarray(car.physics.position[:2], dtype=np.float64)
            clash = ball_low and np.linalg.norm(xy - ball_xy) < MIN_CAR_BALL_DIST
            clash = clash or any(
                np.linalg.norm(xy - np.asarray(o.physics.position[:2], dtype=np.float64)) < MIN_CAR_CAR_DIST
                for o in placed
            )
            if not clash:
                break
            car.physics.position = np.array(
                [rng.uniform(-CAR_X_LIMIT, CAR_X_LIMIT), rng.uniform(-CAR_Y_LIMIT, CAR_Y_LIMIT), C.CAR_RESTING_HEIGHT],
                dtype=np.float32,
            )
        placed.append(car)


def _fill_remaining(cars: List[Car], frame: Frame, rng: np.random.Generator, own_half: bool) -> None:
    """Place supporting cars (2v2/3v3) on the ground in their own half with ordinary speed and boost."""
    for car in cars:
        if own_half:
            x, y, yaw = _random_ground_pose(rng, -CAR_Y_LIMIT, 0.0)
        else:
            x, y, yaw = _random_ground_pose(rng, 0.0, CAR_Y_LIMIT)
        _set_ground_car(car, frame, x, y, yaw, float(rng.uniform(0, 1200)), float(rng.uniform(0, 100)))


# ------------------------------------------------------------------------------------------ scenarios
def kickoff(state: GameState, frame: Frame, rng: np.random.Generator) -> None:
    _set_ball(state, frame, (0.0, 0.0, C.BALL_RESTING_HEIGHT), (0.0, 0.0, 0.0))
    order = rng.permutation(len(_KICKOFF_POS))
    mine, theirs = _teams(state, frame)
    # Both teams use the same spawn indices, giving the usual mirrored kickoff.
    for team_cars, flip in ((mine, 1.0), (theirs, -1.0)):
        for i, car in enumerate(team_cars):
            pos = _KICKOFF_POS[order[i]]
            yaw = _KICKOFF_YAW[order[i]]
            if flip < 0:
                x, y, yaw = -float(pos[0]), -float(pos[1]), yaw + np.pi
            else:
                x, y = float(pos[0]), float(pos[1])
            _set_ground_car(car, frame, x, y, float(yaw), 0.0, 33.3)


def open_play(state: GameState, frame: Frame, rng: np.random.Generator) -> None:
    airborne = rng.random() < 0.3
    z = float(rng.uniform(250, 900)) if airborne else C.BALL_RESTING_HEIGHT
    speed = float(rng.uniform(0, 1500))
    heading = float(rng.uniform(-np.pi, np.pi))
    vz = float(rng.uniform(-300, 500)) if airborne else 0.0
    _set_ball(
        state, frame,
        (float(rng.uniform(-3200, 3200)), float(rng.uniform(-4000, 4000)), z),
        (speed * np.cos(heading), speed * np.sin(heading), vz),
    )
    for car in state.cars.values():
        x, y, yaw = _random_ground_pose(rng)
        _set_ground_car(car, frame, x, y, yaw, float(rng.uniform(0, 1400)), float(rng.uniform(0, 100)))
    _separate(state, rng, fixed=[])


def shot(state: GameState, frame: Frame, rng: np.random.Generator) -> None:
    mine, theirs = _teams(state, frame)
    ax, ay = float(rng.uniform(-2500, 2500)), float(rng.uniform(0, 2500))
    ayaw = float(np.pi / 2 + rng.uniform(-0.5, 0.5))
    _set_ground_car(mine[0], frame, ax, ay, ayaw, float(rng.uniform(500, 1400)), float(rng.uniform(20, 100)))
    lead = float(rng.uniform(500, 1200))
    bx = float(np.clip(ax + lead * np.cos(ayaw) + rng.uniform(-200, 200), -3500, 3500))
    by = float(np.clip(ay + lead * np.sin(ayaw), -4500, 3800))  # stays >= 400 uu clear of the defender
    _set_ball(state, frame, (bx, by, C.BALL_RESTING_HEIGHT), (float(rng.uniform(-200, 200)), float(rng.uniform(0, 600)), 0.0))
    fixed = [mine[0]]
    if theirs:
        _set_ground_car(
            theirs[0], frame, float(rng.uniform(-800, 800)), float(rng.uniform(4200, 4900)),
            float(-np.pi / 2 + rng.uniform(-0.6, 0.6)), 0.0, float(rng.uniform(0, 100)),
        )
        fixed.append(theirs[0])
    _fill_remaining(mine[1:], frame, rng, own_half=True)
    _fill_remaining(theirs[1:], frame, rng, own_half=False)
    _separate(state, rng, fixed=fixed)


def _save_layout(state: GameState, frame: Frame, rng: np.random.Generator, defender_boost: Tuple[float, float]) -> None:
    mine, theirs = _teams(state, frame)
    bx, by = float(rng.uniform(-2000, 2000)), float(rng.uniform(-3000, -1000))
    bz = float(rng.uniform(C.BALL_RESTING_HEIGHT, 600))
    target = np.array([rng.uniform(-700, 700), -C.BACK_WALL_Y, rng.uniform(100, 500)])
    direction = target - np.array([bx, by, bz])
    direction /= np.linalg.norm(direction)
    speed = float(rng.uniform(1200, 2600))
    vel = direction * speed
    vel[2] = float(rng.uniform(0, 400)) if bz > 150 else 0.0
    _set_ball(state, frame, (bx, by, bz), (float(vel[0]), float(vel[1]), float(vel[2])))
    _set_ground_car(
        mine[0], frame, float(rng.uniform(-900, 900)), float(rng.uniform(-4800, -4000)),
        float(np.pi / 2 + rng.uniform(-1.0, 1.0)), float(rng.uniform(0, 500)), float(rng.uniform(*defender_boost)),
    )
    fixed = [mine[0]]
    if theirs:
        # Attacker trails the ball it has just struck.
        back = float(rng.uniform(600, 1500))
        ox = float(np.clip(bx - direction[0] * back, -CAR_X_LIMIT, CAR_X_LIMIT))
        oy = float(np.clip(by - direction[1] * back, -CAR_Y_LIMIT, CAR_Y_LIMIT))
        yaw = float(np.arctan2(direction[1], direction[0]))
        _set_ground_car(theirs[0], frame, ox, oy, yaw, float(rng.uniform(600, 1600)), float(rng.uniform(0, 100)))
        fixed.append(theirs[0])
    _fill_remaining(mine[1:], frame, rng, own_half=True)
    _fill_remaining(theirs[1:], frame, rng, own_half=False)
    _separate(state, rng, fixed=fixed)


def save(state: GameState, frame: Frame, rng: np.random.Generator) -> None:
    _save_layout(state, frame, rng, defender_boost=(20.0, 100.0))


def low_boost_defence(state: GameState, frame: Frame, rng: np.random.Generator) -> None:
    _save_layout(state, frame, rng, defender_boost=(0.0, 12.0))


def recovery(state: GameState, frame: Frame, rng: np.random.Generator) -> None:
    open_play(state, frame, rng)
    mine, _ = _teams(state, frame)
    car = mine[0]
    ball_pos = np.asarray(state.ball.position, dtype=np.float64)
    for _ in range(50):
        car.physics.position = frame.vec(
            float(rng.uniform(-3000, 3000)), float(rng.uniform(-4000, 4000)), float(rng.uniform(400, 1200))
        )
        if np.linalg.norm(np.asarray(car.physics.position, dtype=np.float64) - ball_pos) > 400.0:
            break
    car.physics.euler_angles = np.array(
        [rng.uniform(-np.pi / 2, np.pi / 2), rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)], dtype=np.float32
    )
    vel = rng.normal(size=3)
    vel = vel / np.linalg.norm(vel) * rng.uniform(0, 1200)
    car.physics.linear_velocity = vel.astype(np.float32)
    ang = rng.normal(size=3)
    ang = ang / np.linalg.norm(ang) * rng.uniform(0, 3.0)
    car.physics.angular_velocity = ang.astype(np.float32)
    car.on_ground = False
    car.boost_amount = float(rng.uniform(0, 60))


def aerial(state: GameState, frame: Frame, rng: np.random.Generator) -> None:
    mine, theirs = _teams(state, frame)
    near_wall = rng.random() < 0.5
    side = 1.0 if rng.random() < 0.5 else -1.0
    bx = side * float(rng.uniform(3000, 3700)) if near_wall else float(rng.uniform(-2500, 2500))
    by = float(rng.uniform(-3000, 3500))
    bz = float(rng.uniform(600, 1500))
    vx = -side * float(rng.uniform(0, 400)) if near_wall else float(rng.uniform(-300, 300))
    _set_ball(state, frame, (bx, by, bz), (vx, float(rng.uniform(-300, 300)), float(rng.uniform(0, 400))))
    angle = float(rng.uniform(-np.pi, np.pi))
    dist = float(rng.uniform(1500, 2500))
    cx = float(np.clip(bx + dist * np.cos(angle), -CAR_X_LIMIT, CAR_X_LIMIT))
    cy = float(np.clip(by + dist * np.sin(angle), -CAR_Y_LIMIT, CAR_Y_LIMIT))
    yaw = float(np.arctan2(by - cy, bx - cx))
    _set_ground_car(mine[0], frame, cx, cy, yaw, float(rng.uniform(300, 1200)), float(rng.uniform(60, 100)))
    _fill_remaining(mine[1:], frame, rng, own_half=True)
    _fill_remaining(theirs, frame, rng, own_half=False)
    _separate(state, rng, fixed=[mine[0]])


SCENARIOS: Dict[str, Callable[[GameState, Frame, np.random.Generator], None]] = {
    "kickoff": kickoff,
    "open_play": open_play,
    "shot": shot,
    "save": save,
    "low_boost_defence": low_boost_defence,
    "recovery": recovery,
    "aerial": aerial,
}


class ScenarioMutator(StateMutator[GameState]):
    """Creates `team_size` cars per team and applies one scenario from a weighted, seeded mixture."""

    def __init__(self, team_size: int, weights: Dict[str, float], seed: int):
        if not 1 <= team_size <= C.MAX_TEAM_SIZE:
            raise ValueError(f"team_size must be 1..{C.MAX_TEAM_SIZE}, got {team_size}")
        unknown = sorted(set(weights) - set(SCENARIOS))
        if unknown:
            raise ValueError(f"unknown scenario(s) {unknown}; available: {sorted(SCENARIOS)}")
        names = [n for n, w in weights.items() if w > 0]
        if not names:
            raise ValueError("at least one scenario needs a positive weight")
        probs = np.array([weights[n] for n in names], dtype=np.float64)
        self.team_size = team_size
        self.names = names
        self.probs = probs / probs.sum()
        self.rng = np.random.default_rng(seed)
        self.last_scenario: str | None = None

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if len(state.cars) != 0:
            raise RuntimeError("ScenarioMutator must be the mutator that creates the cars")
        for i in range(self.team_size):
            state.cars[f"blue-{i}"] = _new_car(C.BLUE_TEAM)
        for i in range(self.team_size):
            state.cars[f"orange-{i}"] = _new_car(C.ORANGE_TEAM)
        name = self.names[int(self.rng.choice(len(self.names), p=self.probs))]
        protagonist = C.BLUE_TEAM if self.rng.random() < 0.5 else C.ORANGE_TEAM
        SCENARIOS[name](state, Frame(protagonist), self.rng)
        self.last_scenario = name
        shared_info[SCENARIO_KEY] = name


def validate_state(state: GameState) -> List[str]:
    """Return a list of physical-sanity violations (empty when the state is acceptable)."""
    problems: List[str] = []
    ball_pos = np.asarray(state.ball.position, dtype=np.float64)
    ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float64)
    if not (np.isfinite(ball_pos).all() and np.isfinite(ball_vel).all()):
        problems.append("ball state not finite")
    if abs(ball_pos[0]) > C.SIDE_WALL_X - C.BALL_RADIUS or abs(ball_pos[1]) > C.BACK_WALL_Y - C.BALL_RADIUS:
        problems.append(f"ball outside the field: {ball_pos}")
    if ball_pos[2] < C.BALL_RADIUS or ball_pos[2] > C.CEILING_Z - C.BALL_RADIUS:
        problems.append(f"ball height invalid: {ball_pos[2]}")
    if np.linalg.norm(ball_vel) > C.BALL_MAX_SPEED:
        problems.append("ball faster than the speed limit")
    cars = list(state.cars.items())
    for agent, car in cars:
        pos = np.asarray(car.physics.position, dtype=np.float64)
        vel = np.asarray(car.physics.linear_velocity, dtype=np.float64)
        ang = np.asarray(car.physics.angular_velocity, dtype=np.float64)
        rot = np.asarray(car.physics.rotation_mtx, dtype=np.float64)
        if not (np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(ang).all() and np.isfinite(rot).all()):
            problems.append(f"{agent}: state not finite")
            continue
        if abs(pos[0]) > 4000 or abs(pos[1]) > 5000 or pos[2] < C.CAR_RESTING_HEIGHT - 0.5 or pos[2] > 1900:
            problems.append(f"{agent}: position outside safe bounds {pos}")
        if np.linalg.norm(vel) > C.CAR_MAX_SPEED or np.linalg.norm(ang) > C.CAR_MAX_ANG_VEL:
            problems.append(f"{agent}: velocity over the limit")
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4):
            problems.append(f"{agent}: rotation matrix not orthonormal")
        if not 0.0 <= car.boost_amount <= 100.0:
            problems.append(f"{agent}: boost {car.boost_amount} outside 0..100")
        if car.on_ground and abs(pos[2] - C.CAR_RESTING_HEIGHT) > 1.0:
            problems.append(f"{agent}: flagged on_ground but z={pos[2]}")
        if car.has_flipped or car.has_double_jumped or car.is_jumping or car.has_jumped:
            problems.append(f"{agent}: spawned with jump/flip state, which RocketSim cannot start faithfully")
        if np.linalg.norm(pos - ball_pos) < 200.0:
            problems.append(f"{agent}: overlapping the ball")
    for i in range(len(cars)):
        for j in range(i + 1, len(cars)):
            a = np.asarray(cars[i][1].physics.position, dtype=np.float64)
            b = np.asarray(cars[j][1].physics.position, dtype=np.float64)
            if np.linalg.norm(a - b) < 250.0:
                problems.append(f"{cars[i][0]} and {cars[j][0]} overlap")
    return problems
