"""Physical constants and normalisation scales shared by simulator and deployment code.

Units are Unreal Units (uu, 1 uu = 1 cm), uu/s, rad/s and seconds unless stated otherwise.
Field values come from rlgym.rocket_league.common_values (rlgym-rocket-league 2.0.1) and are
duplicated here only where a value is part of a versioned contract (so an upstream edit can
never silently change what a trained network sees).
"""

from __future__ import annotations

TICKS_PER_SECOND = 120
TICK_TIME = 1.0 / TICKS_PER_SECOND

BLUE_TEAM = 0
ORANGE_TEAM = 1

# Field geometry
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
BACK_NET_Y = 6000.0
CEILING_Z = 2044.0
GOAL_HEIGHT = 642.775
GOAL_CENTER_TO_POST = 892.755
GOAL_THRESHOLD_Y = 5215.5  # |ball y| beyond this counts as a goal in RocketSimEngine
BALL_RADIUS = 91.25
BALL_RESTING_HEIGHT = 93.15
CAR_RESTING_HEIGHT = 17.0  # octane resting z used by RLGym's KickoffMutator

# Speed limits
CAR_MAX_SPEED = 2300.0
BALL_MAX_SPEED = 6000.0
CAR_MAX_ANG_VEL = 5.5
BALL_MAX_ANG_VEL = 6.0
SUPERSONIC_THRESHOLD = 2200.0

# Car mechanics timing
DOUBLEJUMP_MAX_DELAY = 1.25  # seconds after the first jump ends in which a flip/double jump is allowed
FLIP_TORQUE_TIME = 0.65
DEMO_RESPAWN_SECONDS = 3.0

# Boost pads
NUM_BOOST_PADS = 34
BIG_PAD_INDICES = (3, 4, 15, 18, 29, 30)  # verified against RocketSim 2.2.1 arena.get_boost_pads()
BIG_PAD_RECHARGE_SECONDS = 10.0
SMALL_PAD_RECHARGE_SECONDS = 4.0

# Controls layout used by RLGym / RocketSimEngine
NUM_CONTROLS = 8
THROTTLE, STEER, PITCH, YAW, ROLL, JUMP, BOOST, HANDBRAKE = range(NUM_CONTROLS)
CONTROL_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")

# Player-count ceiling of the observation contract (3v3)
MAX_TEAM_SIZE = 3
MAX_TEAMMATES = MAX_TEAM_SIZE - 1
MAX_OPPONENTS = MAX_TEAM_SIZE
