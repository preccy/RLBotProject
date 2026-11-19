"""Frozen external opponent: Necto V1, the RLGym community bot, driven from RocketSim state.

Source: https://github.com/Rolv-Arild/Necto (CC BY-NC-SA 4.0), cloned to external/Necto (git-ignored). The shipped
RLBot model rlbot-support/Necto/necto-model.pt (TorchScript) is used unchanged; Necto plays as shipped in RLBot
(argmax actions, no hard-coded kickoffs, one decision every 8 ticks). It never learns and never produces learner
experience. Checked against Necto's own necto_obs.py / agent.py by `scripts/necto_bench.py selftest`
(docs/BENCH_NECTO.md).

* Observation = Necto V1 (training/obs.py at a7462bb, same layout as rlbot-support/Necto/necto_obs.py): ball +
  players (acting car first) + 34 boost pads, 24 features each; orange mirrored (x, y negated, team flags swapped);
  q = own row + previous 8-control action; kv positions/velocities relative to the acting car. Boost and demo timers
  follow the training builder, updated once per 8-tick decision.
* State fields from rlgym 2: boost_amount / 100, on_ground, has_flip (same definition as rlgym v1), forward/up
  columns of the rotation matrix; boost pads re-ordered from RocketSim's arena order to Necto's list by position.
* Action = Necto's 5 heads -> 8 controls with Necto's own mapping (throttle = pitch, steer = yaw, or roll when
  handbrake).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NECTO_DIR = ROOT / "external" / "Necto" / "rlbot-support" / "Necto"
TICK_SKIP = 8


def necto_dir(path=None) -> Path:
    p = Path(path) if path else DEFAULT_NECTO_DIR
    return p if p.is_absolute() else ROOT / p


def necto_boost_locations(directory=None) -> np.ndarray:
    """BOOST_LOCATIONS exactly as written in Necto's necto_obs.py (parsed, not executed)."""
    tree = ast.parse((necto_dir(directory) / "necto_obs.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BOOST_LOCATIONS":
            return np.array(ast.literal_eval(node.value), dtype=float)
    raise RuntimeError("BOOST_LOCATIONS not found in necto_obs.py")


class NectoObsV1:
    INVERT = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    NORM = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, boost_locations: np.ndarray, tick_skip: int = TICK_SKIP):
        self.locs = boost_locations
        self.big = self.locs[:, 2] > 72
        self.tick_skip = tick_skip
        self.reset(2)

    def reset(self, n_players: int) -> None:
        self.boost_timers = np.zeros(len(self.locs))
        self.demo_timers = np.zeros(n_players)

    def build(self, ball: dict, players: list, boost_active: np.ndarray, prev_action: np.ndarray):
        """players[0] is the acting car. Each player: team, pos, vel, fwd, up, angvel, boost (0-1), on_ground,
        has_flip, demoed. Returns (q, kv, mask) as Necto's model expects (batch 1)."""
        P = len(players)
        qkv = np.zeros((1, 1 + P + len(self.locs), 24))
        qkv[0, 0, 3] = 1
        qkv[0, 0, 5:8], qkv[0, 0, 8:11], qkv[0, 0, 17:20] = ball["pos"], ball["vel"], ball["angvel"]
        demos = np.zeros(P)
        for n, p in enumerate(players, start=1):
            qkv[0, n, 1 if p["team"] == 0 else 2] = 1
            qkv[0, n, 5:8], qkv[0, n, 8:11] = p["pos"], p["vel"]
            qkv[0, n, 11:14], qkv[0, n, 14:17], qkv[0, n, 17:20] = p["fwd"], p["up"], p["angvel"]
            qkv[0, n, 20] = p["boost"]
            qkv[0, n, 22], qkv[0, n, 23] = p["on_ground"], p["has_flip"]
            demos[n - 1] = p["demoed"]
        b0 = 1 + P
        qkv[0, b0:, 4] = 1
        qkv[0, b0:, 5:8] = self.locs
        qkv[0, b0:, 20] = 0.12 + 0.88 * self.big
        # timers: training builder logic, once per decision
        pads = boost_active.astype(float)
        new = (pads == 1) & (self.boost_timers == 0)
        self.boost_timers[new] = 0.4 + 0.6 * self.big[new]
        self.boost_timers *= pads
        qkv[0, b0:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200
        self.boost_timers[self.boost_timers < 0] = 0
        new_d = (demos == 1) & (self.demo_timers == 0)
        self.demo_timers[new_d] = 0.3
        self.demo_timers *= demos
        qkv[0, 1:b0, 21] = self.demo_timers
        self.demo_timers -= self.tick_skip / 1200
        self.demo_timers[self.demo_timers < 0] = 0

        qkv = qkv / self.NORM
        mask = np.zeros((1, qkv.shape[1]))
        qkv[0, 1, 0] = 1  # is_main (acting car is players[0])
        if players[0]["team"] == 1:
            qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]
            qkv *= self.INVERT
        q = np.expand_dims(np.concatenate((qkv[0, 1, :], prev_action), axis=0), axis=(0, 1))
        kv = qkv
        kv[0, :, 5:11] -= q[0, 0, 5:11]
        return q, kv, mask


def necto_controls(heads: np.ndarray) -> np.ndarray:
    """Necto's agent.py mapping from 5 discrete heads to 8 controls."""
    a = heads.astype(float).copy()
    a[0] -= 1
    a[1] -= 1
    return np.array([a[0], a[1], a[0], a[1] * (1 - a[4]), a[1] * a[4], a[2], a[3], a[4]])


def car_dict(car) -> dict:
    ph = car.physics
    return {"team": int(car.team_num), "pos": np.asarray(ph.position, float), "vel": np.asarray(ph.linear_velocity, float),
            "fwd": np.asarray(ph.forward, float), "up": np.asarray(ph.up, float),
            "angvel": np.asarray(ph.angular_velocity, float), "boost": float(car.boost_amount) / 100.0,
            "on_ground": float(car.on_ground), "has_flip": float(car.has_flip), "demoed": float(car.is_demoed)}


def pad_permutation(env, locs: np.ndarray) -> np.ndarray:
    """Index into the engine's boost_pad_timers for each pad of Necto's list (matched by position)."""
    pads = env.transition_engine._arena.get_boost_pads()
    pos = np.array([[p.get_pos().x, p.get_pos().y, p.get_pos().z] for p in pads])
    perm = np.array([int(np.argmin(np.linalg.norm(pos[:, :2] - l[:2], axis=1))) for l in locs])
    err = np.linalg.norm(pos[perm, :2] - locs[:, :2], axis=1).max()
    if len(set(perm.tolist())) != len(locs) or err > 10 or (pos[perm, 2] > 72).tolist() != (locs[:, 2] > 72).tolist():
        raise RuntimeError(f"boost pad mapping failed (max xy error {err:.1f})")
    return perm


class NectoPolicy:
    def __init__(self, beta: float = 1.0, seed: int = 0, directory=None):
        import torch

        self.torch = torch
        d = necto_dir(directory)
        with open(d / "necto-model.pt", "rb") as f:
            self.actor = torch.jit.load(f)
        self.actor.eval()
        self.beta = beta
        self.gen = torch.Generator().manual_seed(seed)
        self.obs = NectoObsV1(necto_boost_locations(d))
        self.prev = np.zeros(8)

    def reset(self, n_players: int) -> None:
        self.obs.reset(n_players)
        self.prev = np.zeros(8)

    def heads(self, q, kv, mask) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            out, _ = self.actor(tuple(torch.from_numpy(x).float() for x in (q, kv, mask)))
        if self.beta == 1:
            return np.array([int(torch.argmax(l.reshape(-1))) for l in out])
        scale = np.log((self.beta + 1) / (1 - self.beta)) / np.log(3)
        return np.array([int(torch.multinomial(torch.softmax(l.reshape(-1) * scale, -1), 1, generator=self.gen))
                         for l in out])

    def act(self, state, me, others, perm) -> np.ndarray:
        ball = {"pos": np.asarray(state.ball.position), "vel": np.asarray(state.ball.linear_velocity),
                "angvel": np.asarray(state.ball.angular_velocity)}
        players = [car_dict(state.cars[a]) for a in [me] + list(others)]
        active = (np.asarray(state.boost_pad_timers)[perm] == 0)
        q, kv, mask = self.obs.build(ball, players, active, self.prev)
        self.prev = necto_controls(self.heads(q, kv, mask))
        return self.prev
