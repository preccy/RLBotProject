"""High-level replay -> obs-v1 demonstration blocks (docs/EXPERIMENT_HL_IMITATION.md, "Label construction").

Input: one rlgym-tools ReplayFrame file from Rolv-Arild/hlrlrd-parsed (30 Hz frames, 4 ticks apart).

* Policy inputs (observation at block start, tick t) use only information available at t: the ball at frame t (the
  replay sets it every frame), each car's latest TRUE replay update at or before t (update_age == 0) extrapolated
  to t, the pad timers at t, and the previous block's label as prev_action. The converter's interpolated car states
  are never used as inputs, because they were simulated with controls inferred from future motion.
* Labels (ticks t..t+7) use the converter's per-frame controls, which may rely on future motion (inferred
  pitch/yaw/roll, interpolated analog throttle/steer). They are spread over ticks and mapped with the existing
  build_demos rules (JumpMachine, dodge handling, flags). Analog sticks use the block mean, then "nearest" or
  "dither".
* Exclusions: kickoffs (to 1.0 s after the ball leaves the centre), gaps, frame spacing != 4 ticks, demolished cars,
  car snapshots older than 0.1 s, and stretches where the converter's controls do not reproduce the replay in
  RocketSim (> 150 uu or > 20 deg between consecutive true updates of the acting car).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "il_feasibility", ROOT / "external" / "rlgym-tools-2.6.5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from botcore.obs.builder import build_obs  # noqa: E402
from build_demos import JumpMachine  # noqa: E402
from il_map import T, nearest_row  # noqa: E402

TICKS = 120.0
GRAVITY = -650.0
MAX_AGE_S = 0.1 + 1e-6
KICKOFF_SKIP_TICKS = 120
REPRO_POS_UU = 150.0
REPRO_ANG_DEG = 20.0


def load_frames(path: Path):
    from rlgym_tools.rocket_league.misc.serialize import deserialize_replay_frame

    z = np.load(path)
    return [deserialize_replay_frame(v) for v in z["frames"]]


def arrays(fr):
    ids = sorted(fr[0].state.cars)
    n = len(fr)
    A = {"ids": ids, "tick": np.array([f.state.tick_count for f in fr], dtype=np.int64)}
    A["dt"] = np.append(np.diff(A["tick"]), 0)
    A["team"] = np.array([fr[0].state.cars[c].team_num for c in ids])
    A["ua"] = np.array([[f.update_age.get(c, 1.0) for c in ids] for f in fr], dtype=np.float64)
    A["act"] = np.array([[f.actions[c] for c in ids] for f in fr], dtype=np.float32)
    A["ground"] = np.array([[f.state.cars[c].on_ground for c in ids] for f in fr])
    A["flipped"] = np.array([[f.state.cars[c].has_flipped for c in ids] for f in fr])
    A["demoed"] = np.array([[f.state.cars[c].is_demoed for c in ids] for f in fr])
    A["goal"] = np.array([f.state.goal_scored for f in fr])
    A["ball"] = np.array([f.state.ball.position for f in fr], dtype=np.float64)
    # latest true update index per frame and car (-1 if none in this gameplay period)
    lt = np.full((n, len(ids)), -1, dtype=np.int64)
    period_start = np.zeros(n, dtype=np.int64)
    for i in range(n):
        new_period = i == 0 or A["tick"][i] - A["tick"][i - 1] > 6
        period_start[i] = i if new_period else period_start[i - 1]
        for c in range(len(ids)):
            if A["ua"][i, c] == 0:
                lt[i, c] = i
            elif not new_period:
                lt[i, c] = lt[i - 1, c]
    A["lt"], A["period_start"] = lt, period_start
    return A


def valid_frames(A):
    """Frames usable as part of a decision block: gameplay after the kickoff exclusion, 4-tick spacing to the next
    frame, no goal, nobody demolished."""
    n = len(A["tick"])
    ok = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        s = A["period_start"][i]
        j = i
        while j < n and A["period_start"][j] == s:
            j += 1
        moved = np.where(np.linalg.norm(A["ball"][s:j, :2], axis=1) > 5.0)[0]
        if len(moved):
            t0 = A["tick"][s + moved[0]] + KICKOFF_SKIP_TICKS
            ok[s:j] = A["tick"][s:j] >= t0
        i = j
    ok &= A["dt"] == 4
    ok &= ~A["goal"]
    ok &= ~A["demoed"].any(axis=1)
    return ok


def stretches(ok):
    out, i, n = [], 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        out.append((i, j))  # frames [i, j) each have a 4-tick step to the next frame
        i = j
    return out


# ------------------------------------------------------------------ causal observation
def _rodrigues(w, dt):
    th = np.linalg.norm(w) * dt
    if th < 1e-9:
        return np.eye(3)
    k = w / np.linalg.norm(w)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def snapshot_car(fr, A, i, c):
    """Car c as known at frame i: its latest true update, extrapolated to frame i's tick. None if too old."""
    j = A["lt"][i, c]
    if j < 0:
        return None
    age = (A["tick"][i] - A["tick"][j]) / TICKS
    if age > MAX_AGE_S:
        return None
    car = copy.deepcopy(fr[j].state.cars[A["ids"][c]])
    if age > 0:
        ph = car.physics
        pos, vel, ang = (np.array(ph.position, dtype=np.float64), np.array(ph.linear_velocity, dtype=np.float64),
                         np.array(ph.angular_velocity, dtype=np.float64))
        R = np.array(ph.rotation_mtx, dtype=np.float64)
        if car.on_ground:
            pos = pos + vel * age
        else:
            pos = pos + vel * age + 0.5 * np.array([0, 0, GRAVITY]) * age ** 2
            vel = vel + np.array([0, 0, GRAVITY]) * age
        ph.position = pos.astype(np.float32)
        ph.linear_velocity = vel.astype(np.float32)
        ph.rotation_mtx = (_rodrigues(ang, age) @ R).astype(np.float32)
        car._inverted_physics = None
    return car


def causal_state(fr, A, i):
    cars = {}
    for c, aid in enumerate(A["ids"]):
        car = snapshot_car(fr, A, i, c)
        if car is None:
            return None
        cars[aid] = car
    st = copy.copy(fr[i].state)
    st.cars = cars
    return st


# ------------------------------------------------------------------ labels
class HLJumpMachine(JumpMachine):
    """build_demos.JumpMachine with one change for 30 Hz replay data: while holding, a block keeps the jump bit only
    if the jump is still held at the block's end (and on >= 4 ticks). The original rule (held from the block's
    start on >= 4 ticks) merged a fast dodge's 4-tick release into the hold, because the re-press then starts exactly
    on the next block's first tick and is read as a continued hold. Releasing in the block where the human released
    keeps press -> release -> press patterns on time."""

    def step(self, u: np.ndarray, ground: np.ndarray):
        j = u[:, 5] > 0.5
        starts = [k for k in range(8) if j[k] and (not j[k - 1] if k > 0 else not self.last_tick_jump)]
        self.last_tick_jump = bool(j[-1])
        press = None
        if self.prev_bit:
            bit = bool(j[0] and j[-1] and j.sum() >= 4 and not starts)
            if starts:
                self.pending = (u[starts[0], 2:5], bool(ground[starts[0]]), starts[0] - 8)
        else:
            if self.pending is not None:
                bit, press = True, self.pending
                self.pending = None
            elif starts and starts[0] < 4:
                bit, press = True, (u[starts[0], 2:5], bool(ground[starts[0]]), starts[0])
            elif starts:
                bit = False
                self.pending = (u[starts[0], 2:5], bool(ground[starts[0]]), starts[0] - 8)
            else:
                bit = False
        self.prev_bit = bit
        return bit, press


def quantize(m: float, scheme: str, rng) -> float:
    if scheme == "nearest":
        return float(np.sign(m)) if abs(m) >= 0.5 else 0.0
    return float(np.sign(m)) if rng.random() < abs(m) else 0.0


def block_label(u, ground, bit, press, scheme, rng):
    """Row for one 8-tick window. Like build_demos.block_row, but analog sticks use the block mean + scheme."""
    tgt = np.zeros(8, dtype=np.float32)
    for j in range(5):
        tgt[j] = quantize(float(u[:, j].mean()), scheme, rng)
    tgt[5] = float(bit)
    tgt[6] = float(u[:, 6].mean() >= 0.5)
    tgt[7] = float(u[:, 7].mean() >= 0.5)
    g = float(np.mean(ground))
    approx, dodge = False, False
    if press is not None:
        stick, on_ground, _ = press
        tgt[2:5] = stick
        if not on_ground:
            g, dodge = 0.0, True
            approx = bool(abs(stick[1] + stick[2]) > 1)
    if bit and not dodge:
        tgt[3] = 0.0
    return nearest_row(tgt, g, dodge=dodge), approx, dodge


def repro_ok(fr, A, engine):
    """bad[i, c] = True when frame i's controls for car c are not verified to reproduce the replay."""
    n, nc = len(fr), len(A["ids"])
    good = np.zeros((n, nc), dtype=bool)
    err = []
    for c in range(nc):
        z = np.where(A["ua"][:, c] == 0)[0]
        for j, k in zip(z[:-1], z[1:]):
            if k - j > 4 or np.any(A["dt"][j:k] != 4) or A["period_start"][j] != A["period_start"][k]:
                continue
            engine.set_state(fr[j].state, {})
            acts = {aid: np.repeat(A["act"][j:k, cc], 4, axis=0) for cc, aid in enumerate(A["ids"])}
            out = engine.step(acts, {})
            a, b = out.cars[A["ids"][c]].physics, fr[k].state.cars[A["ids"][c]].physics
            dp = float(np.linalg.norm(np.asarray(a.position) - np.asarray(b.position)))
            cosang = float(np.clip(np.dot(a.forward, b.forward), -1, 1))
            dang = float(np.degrees(np.arccos(cosang)))
            err.append((dp, dang))
            good[j:k, c] = dp <= REPRO_POS_UU and dang <= REPRO_ANG_DEG
    return good, np.array(err)


def extract(path: Path, scheme: str = "dither", seed: int = 0, with_repro: bool = True, keep_frames=False):
    from rlgym.rocket_league.sim import RocketSimEngine

    fr = load_frames(path)
    A = arrays(fr)
    ok = valid_frames(A)
    if with_repro:
        good, repro_err = repro_ok(fr, A, RocketSimEngine())
    else:
        good, repro_err = np.ones((len(fr), len(A["ids"])), dtype=bool), np.zeros((0, 2))
    out = {k: [] for k in ("obs", "label", "prev", "frame", "tick", "car", "team", "approx_dir", "dodge_lost",
                           "press_inflight", "press_offset", "steer_mean", "throttle_mean", "boost_mean", "dodge")}
    stats = {"frames": len(fr), "blocks_postkickoff": 0, "blocks_usable": 0, "excl_age": 0, "excl_repro": 0}
    for c in range(len(A["ids"])):
        rng = np.random.default_rng([seed, c])
        for s, e in stretches(ok):
            machine, labels = HLJumpMachine(), []
            w = 0
            while s + 2 * w + 2 <= e:
                a = s + 2 * w
                u = np.repeat(A["act"][a:a + 2, c], 4, axis=0)
                g = np.repeat(A["ground"][a:a + 2, c], 4)
                bit, press = machine.step(u, g)
                row, ap, dodge = block_label(u, g, bit, press, scheme, rng)
                labels.append((a, row, ap, press, dodge, u))
                w += 1
            for w in range(1, len(labels)):
                a, row, ap, press, dodge, u = labels[w]
                stats["blocks_postkickoff"] += 1
                if not (good[a, c] and good[a + 1, c]):
                    stats["excl_repro"] += 1
                    continue
                st = causal_state(fr, A, a)
                if st is None:
                    stats["excl_age"] += 1
                    continue
                prev_row = labels[w - 1][1]
                o = build_obs(st, A["ids"][c], prev_action=T[prev_row])
                out["obs"].append(o); out["label"].append(row); out["prev"].append(prev_row)
                out["frame"].append(a); out["tick"].append(A["tick"][a]); out["car"].append(c)
                out["team"].append(A["team"][c]); out["approx_dir"].append(ap)
                out["press_inflight"].append(press is not None and press[2] < -1)
                out["press_offset"].append(np.nan if press is None else press[2])
                out["steer_mean"].append(float(u[:, 1].mean())); out["throttle_mean"].append(float(u[:, 0].mean()))
                out["boost_mean"].append(float(u[:, 6].mean())); out["dodge"].append(dodge)
                out["dodge_lost"].append(False)
                stats["blocks_usable"] += 1
    d = {k: np.array(v) for k, v in out.items()}
    d["obs"] = d["obs"].astype(np.float32) if len(d["obs"]) else np.zeros((0, 276), np.float32)
    # dodge preservation (as build_demos): each has_flipped rising edge inside the usable grid needs a fresh jump
    # label within one block of the block containing it
    jb = T[d["label"], 5] > 0.5 if len(d["label"]) else np.zeros(0, bool)
    pj = T[d["prev"], 5] > 0.5 if len(d["prev"]) else np.zeros(0, bool)
    onset = jb & ~pj
    n_dodge = n_kept = 0
    for c in range(len(A["ids"])):
        idx = np.where(d["car"] == c)[0]
        ticks = d["tick"][idx]
        edges = np.where(np.diff(A["flipped"][:, c].astype(int)) == 1)[0] + 1
        for k in edges:
            tk = A["tick"][k] - 4  # the press lies about one frame before the flag appears
            b = np.searchsorted(ticks, tk, side="right") - 1
            if (b < 1 or b + 1 >= len(ticks) or not (ticks[b] <= tk < ticks[b] + 8)
                    or ticks[b] - ticks[b - 1] != 8 or ticks[b + 1] - ticks[b] != 8):
                continue  # dodge not inside a contiguous run of three usable blocks: cannot be judged
            n_dodge += 1
            kept = bool(onset[idx[max(b - 1, 0):b + 2]].any())
            n_kept += kept
            if not kept:
                d["dodge_lost"][idx[max(b - 1, 0):b + 2]] = True
    stats.update(dodges_in_grid=n_dodge, dodges_pattern_kept=n_kept,
                 repro_err_median=(np.median(repro_err, axis=0).tolist() if len(repro_err) else None),
                 repro_bad_share=float(np.mean((repro_err[:, 0] > REPRO_POS_UU) | (repro_err[:, 1] > REPRO_ANG_DEG)))
                 if len(repro_err) else None)
    if keep_frames:
        return d, stats, fr, A
    return d, stats
