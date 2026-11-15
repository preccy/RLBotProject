# Observation specification `obs-v1`

Generated from `src/botcore/obs/spec.py` by `python -m botcore obs-doc`. Do not edit by hand.

- Flat `float32` vector of length **276**; spec hash `48516c0485f94db39b64da90b0c7a78facbac84c75ffe1de7c29b4e1c8cb9de6`.
- Layout: self `[0:40]`, ball `[40:62]`, cars `[62:242]` (5 slots x 36; slots 0-1 teammates, 2-4 opponents), pads `[242:276]`.
- Team-relative: orange observations are rotated 180 degrees about Z (a rotation, never a reflection).
- Within a team, cars fill slots in the stable order of the state's car dictionary (RLGym agent order in the simulator, packet player order in RLBot). The order is arbitrary but fixed for a match; a slot-order-sensitive model (the MLP) therefore needs slot shuffling or the entity encoder before 2v2/3v3 training.

## Self block

| Index | Feature | Units | Scaling | Frame | Deployment availability / reconstruction | Missing values | Reset behaviour |
|---|---|---|---|---|---|---|---|
| 0-2 | `pos` | uu | /4096 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 3-5 | `forward` | unit vector | none | team-relative world frame (orange rotated 180 deg about Z) | from packet rotator via rlgym-compat euler_to_rotation | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 6-8 | `up` | unit vector | none | team-relative world frame (orange rotated 180 deg about Z) | from packet rotator via rlgym-compat euler_to_rotation | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 9-11 | `lin_vel` | uu/s | /2300 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 12-14 | `ang_vel` | rad/s | /5.5 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 15-17 | `lin_vel_local` | uu/s | /2300 | controlled car's local frame (components along its forward, lateral and up axes) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 18-20 | `ang_vel_local` | rad/s | /5.5 | controlled car's local frame (components along its forward, lateral and up axes) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 21 | `boost` | boost 0-100 | /100 | n/a (scalar) | direct from GamePacket via rlgym-compat | always present | stateless: recomputed from the current state every decision |
| 22 | `on_ground` | bool | 0/1 | n/a (scalar) | sim: >=3 wheels in contact. Game: AirState.OnGround, or Jumping within 6 ticks of the jump (rlgym-compat heuristic). Can disagree for a few ticks around take-off/landing and when a car rests on fewer than 3 wheels. | always present | stateless: recomputed from the current state every decision |
| 23 | `has_jumped` | bool | 0/1 | n/a (scalar) | direct from GamePacket via rlgym-compat | always present | stateless: recomputed from the current state every decision |
| 24 | `has_double_jumped` | bool | 0/1 | n/a (scalar) | direct from GamePacket via rlgym-compat | always present | stateless: recomputed from the current state every decision |
| 25 | `has_flipped` | bool | 0/1 | n/a (scalar) | packet has_dodged | always present | stateless: recomputed from the current state every decision |
| 26 | `flip_available` | bool | 0/1 | n/a (scalar) | derived: not double-jumped, not flipped and the 1.25 s window has not expired. In the game the window comes from dodge_timeout; botcore patches rlgym-compat's stale value once dodge_timeout returns to -1 in the air (see deploy/state_fixes.py). | always present | stateless: recomputed from the current state every decision |
| 27 | `flip_window` | seconds remaining | /1.25, clipped to [0,1] | n/a (scalar) | 1 - air_time_since_jump/1.25; 1 on the ground or after a flip reset, 0 when no flip is available | always present | stateless: recomputed from the current state every decision |
| 28 | `is_flipping` | bool | 0/1 | n/a (scalar) | has_dodged and dodge_elapsed < 0.65 s | always present | stateless: recomputed from the current state every decision |
| 29 | `is_demoed` | bool | 0/1 | n/a (scalar) | demolished_timeout != -1 | always present | stateless: recomputed from the current state every decision |
| 30 | `demo_timer` | seconds until respawn | /3, clipped to [0,1] | n/a (scalar) | demolished_timeout | 0 when alive | stateless: recomputed from the current state every decision |
| 31 | `is_supersonic` | bool | 0/1 | n/a (scalar) | direct from GamePacket via rlgym-compat | always present | stateless: recomputed from the current state every decision |
| 32-39 | `prev_action` | controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake) | none (already in [-1,1] / {0,1}) | n/a (scalar) | the bot's own last chosen action, tracked by the action parser (never read from the packet) | zeros before the first action | zeroed on every environment reset, and on kickoff/goal resets and on-the-fly resets in the RLBot runner |

## Ball block

| Index | Feature | Units | Scaling | Frame | Deployment availability / reconstruction | Missing values | Reset behaviour |
|---|---|---|---|---|---|---|---|
| 40-42 | `pos` | uu | /4096 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | always present | stateless: recomputed from the current state every decision |
| 43-45 | `lin_vel` | uu/s | /2300 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | always present | stateless: recomputed from the current state every decision |
| 46-48 | `ang_vel` | rad/s | /6 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | always present | stateless: recomputed from the current state every decision |
| 49-51 | `rel_pos` | uu (ball - self) | /4096 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 52-54 | `rel_vel` | uu/s (ball - self) | /2300 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 55-57 | `rel_pos_local` | uu (ball - self) | /4096 | controlled car's local frame (components along its forward, lateral and up axes) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 58-60 | `rel_vel_local` | uu/s (ball - self) | /2300 | controlled car's local frame (components along its forward, lateral and up axes) | direct from GamePacket via rlgym-compat | zeroed while the car is demolished (sim and game report different placeholder poses) | stateless: recomputed from the current state every decision |
| 61 | `distance` | uu | /4096 | n/a (scalar) | direct from GamePacket via rlgym-compat | 0 while self is demolished; exactly 0 (never NaN) for coincident positions | stateless: recomputed from the current state every decision |

## Car slot block (repeated 5 times; offsets shown for slot 0)

| Index | Feature | Units | Scaling | Frame | Deployment availability / reconstruction | Missing values | Reset behaviour |
|---|---|---|---|---|---|---|---|
| 62 | `present` | bool | 0/1 | n/a (scalar) | one slot per car in packet.players other than self | 0 marks an empty slot; every other value in the slot is then 0 | stateless: recomputed from the current state every decision |
| 63 | `is_teammate` | bool | 0/1 | n/a (scalar) | packet team vs own team | all zeros when the slot is empty (present = 0) | stateless: recomputed from the current state every decision |
| 64-66 | `pos` | uu | /4096 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zeroed while that car is demolished | stateless: recomputed from the current state every decision |
| 67-69 | `forward` | unit vector | none | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zeroed while that car is demolished | stateless: recomputed from the current state every decision |
| 70-72 | `up` | unit vector | none | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zeroed while that car is demolished | stateless: recomputed from the current state every decision |
| 73-75 | `lin_vel` | uu/s | /2300 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zeroed while that car is demolished | stateless: recomputed from the current state every decision |
| 76-78 | `ang_vel` | rad/s | /5.5 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zeroed while that car is demolished | stateless: recomputed from the current state every decision |
| 79 | `boost` | boost 0-100 | /100 | n/a (scalar) | packet reports every player's boost (true information in offline matches) | all zeros when the slot is empty (present = 0) | stateless: recomputed from the current state every decision |
| 80 | `on_ground` | bool | 0/1 | n/a (scalar) | same heuristic as self.on_ground | all zeros when the slot is empty (present = 0) | stateless: recomputed from the current state every decision |
| 81 | `flip_available` | bool | 0/1 | n/a (scalar) | same derivation as self.flip_available | all zeros when the slot is empty (present = 0) | stateless: recomputed from the current state every decision |
| 82 | `is_demoed` | bool | 0/1 | n/a (scalar) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0) | stateless: recomputed from the current state every decision |
| 83 | `is_supersonic` | bool | 0/1 | n/a (scalar) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0) | stateless: recomputed from the current state every decision |
| 84-86 | `rel_pos` | uu (car - self) | /4096 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zero if either car is demolished | stateless: recomputed from the current state every decision |
| 87-89 | `rel_vel` | uu/s (car - self) | /2300 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zero if either car is demolished | stateless: recomputed from the current state every decision |
| 90-92 | `rel_pos_local` | uu (car - self) | /4096 | controlled car's local frame (components along its forward, lateral and up axes) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zero if either car is demolished | stateless: recomputed from the current state every decision |
| 93 | `distance` | uu | /4096 | n/a (scalar) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zero if either car is demolished | stateless: recomputed from the current state every decision |
| 94-96 | `ball_rel_pos` | uu (ball - car) | /4096 | team-relative world frame (orange rotated 180 deg about Z) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zeroed while that car is demolished | stateless: recomputed from the current state every decision |
| 97 | `ball_distance` | uu | /4096 | n/a (scalar) | direct from GamePacket via rlgym-compat | all zeros when the slot is empty (present = 0); zeroed while that car is demolished | stateless: recomputed from the current state every decision |

## Boost pad block

| Index | Feature | Units | Scaling | Frame | Deployment availability / reconstruction | Missing values | Reset behaviour |
|---|---|---|---|---|---|---|---|
| 242-275 | `pad_timers` | seconds until the pad is available (0 = available) | /10 | pad order = BOOST_LOCATIONS for blue, reversed for orange (the reversed order is exactly the 180 deg rotated layout; verified against RocketSim 2.2.1) | reconstructed by the runner from is_active transitions and the known 4 s / 10 s cooldowns (deploy/state_fixes.PadTimerTracker), so it does not depend on which meaning BoostPadState.timer has in a given RLBot build. A pad already inactive when the bot first sees it uses the packet timer clipped to the cooldown. | 0 (available) if a pad is missing from the packet | stateless: recomputed from the current state every decision |

