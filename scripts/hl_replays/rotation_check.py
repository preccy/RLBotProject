"""Replacement checks R1 (rotation equivalence) and R2 (per-player steering fidelity), plus a reproduction of the
original pooled C2 numbers. Pre-registered in docs/EXPERIMENT_HL_IMITATION.md ("Replacement check").

    .\\.venv\\Scripts\\python.exe scripts\\hl_replays\\rotation_check.py

Writes runs/hl-check/replacement.json. Exit code 0 = R1 and R2 pass.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import checks  # noqa: E402
import hlconv  # noqa: E402
from hlconv import ROOT, T  # noqa: E402

from botcore.imitation import equivalence_mask  # noqa: E402
from botcore.obs import spec as S  # noqa: E402
from rlgym.rocket_league.common_values import BOOST_LOCATIONS  # noqa: E402

OUT = ROOT / "runs" / "hl-check"
F = checks.F
D = np.diag([-1.0, -1.0, 1.0])
_L = np.array(BOOST_LOCATIONS, dtype=float)
PAD_PERM = np.array([int(np.argmin(np.linalg.norm(_L - r, axis=1))) for r in _L * np.array([-1, -1, 1])])
OBS_TOL, METRIC_TOL = 1e-4, 1e-4
FLAGS = ("press_inflight", "dodge_lost", "approx_dir", "dodge")


def _rot_phys(ph):
    ph.position = (D @ np.asarray(ph.position, float)).astype(np.float32)
    ph.linear_velocity = (D @ np.asarray(ph.linear_velocity, float)).astype(np.float32)
    ph.angular_velocity = (D @ np.asarray(ph.angular_velocity, float)).astype(np.float32)
    ph.rotation_mtx = (D @ np.asarray(ph.rotation_mtx, float)).astype(np.float32)


def rotate_frames(fr):
    out = []
    for f in fr:
        g = copy.deepcopy(f)
        st = g.state
        _rot_phys(st.ball)
        st._inverted_ball = None
        st.boost_pad_timers = np.ascontiguousarray(np.asarray(st.boost_pad_timers)[PAD_PERM])
        st._inverted_boost_pad_timers = None
        for car in st.cars.values():
            _rot_phys(car.physics)
            car.team_num = 1 - car.team_num
            car._inverted_physics = None
        out.append(g)
    return out


def steer_metric(d, c, obs_y=True, fr=None, A=None):
    """Per-player C2 statistic. obs_y=False takes y from raw converter physics at the next block's start frame."""
    xs, ys = [], []
    for i, j in checks.next_pairs(d):
        if d["car"][i] != c:
            continue
        o, n = d["obs"][i], d["obs"][j]
        if not (o[F["self.on_ground[0]"]] > .5 and n[F["self.on_ground[0]"]] > .5
                and o[F["self.lin_vel_local[0]"]] * S.LIN_VEL_SCALE > 300):
            continue
        xs.append(d["steer_mean"][i])
        if obs_y:
            ys.append(n[F["self.ang_vel_local[2]"]])
        else:
            ph = fr[d["frame"][j]].state.cars[A["ids"][c]].physics
            ys.append(float(np.dot(np.asarray(ph.angular_velocity), np.asarray(ph.rotation_mtx)[:, 2])))
    return checks.corr(xs, ys), len(xs)


def r1_compare(m, d0, A0, d1, A1):
    k0 = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(d0["car"], d0["tick"]))}
    k1 = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(d1["car"], d1["tick"]))}
    res = {"match": m, "blocks_orig": len(k0), "blocks_rot": len(k1), "borderline": 0, "divergences": [],
           "max_obs_abs_diff": 0.0}
    for key in sorted(set(k0) ^ set(k1), key=lambda x: (x[1], x[0])):
        c, t = key
        side = d0 if key in k0 else d1
        a = int(np.where(A0["tick"] == t)[0][0])
        e0 = A0["repro_err"][a:a + 2, c]; e1 = A1["repro_err"][a:a + 2, c]
        near = lambda e: bool(np.any((np.abs(e[:, 0] - hlconv.REPRO_POS_UU) <= 1) | (np.abs(e[:, 1] - hlconv.REPRO_ANG_DEG) <= 0.1)))
        if near(e0) and near(e1):
            res["borderline"] += 1
        else:
            res["divergences"].append({"player": c, "tick": t, "item": "usable set",
                                       "detail": f"only in {'original' if side is d0 else 'rotated'}; repro err orig {e0.tolist()} rot {e1.tolist()}"})
    for key in sorted(set(k0) & set(k1), key=lambda x: (x[1], x[0])):
        i, j = k0[key], k1[key]
        for item in ("label", "prev") + FLAGS:
            if d0[item][i] != d1[item][j]:
                res["divergences"].append({"player": key[0], "tick": key[1], "item": item,
                                           "detail": f"{d0[item][i]} vs {d1[item][j]}"})
        g0, g1 = d0["obs"][i][F["self.on_ground[0]"]] > .5, d1["obs"][j][F["self.on_ground[0]"]] > .5
        if not np.array_equal(equivalence_mask(T, int(d0["label"][i]), bool(g0)), equivalence_mask(T, int(d1["label"][j]), bool(g1))):
            res["divergences"].append({"player": key[0], "tick": key[1], "item": "mask", "detail": ""})
        diff = np.abs(d0["obs"][i].astype(np.float64) - d1["obs"][j])
        res["max_obs_abs_diff"] = max(res["max_obs_abs_diff"], float(diff.max()))
        if diff.max() > OBS_TOL:
            f = int(np.argmax(diff))
            res["divergences"].append({"player": key[0], "tick": key[1], "item": "obs",
                                       "detail": f"{checks.NAMES[f]}: {d0['obs'][i][f]:.6f} vs {d1['obs'][j][f]:.6f}"})
    res["steer_metric"] = {}
    for c in range(2):
        a, na = steer_metric(d0, c); b, nb = steer_metric(d1, c)
        res["steer_metric"][f"player{c}"] = {"orig_team": int(A0["team"][c]), "orig": a, "rot": b, "n": [na, nb]}
        if not (abs(a - b) <= METRIC_TOL and na == nb):
            res["divergences"].append({"player": c, "tick": None, "item": "steer metric", "detail": f"{a} vs {b} (n {na}/{nb})"})
    res["n_divergences"] = len(res["divergences"])
    res["first_divergence"] = res["divergences"][0] if res["divergences"] else None
    res["divergences"] = res["divergences"][:20]
    res["pass"] = res["n_divergences"] == 0
    return res


def r2(m, d, fr, A):
    from rlgym.rocket_league.sim import RocketSimEngine

    eng = RocketSimEngine()
    lab = {(int(c), int(t)): int(r) for c, t, r in zip(d["car"], d["tick"], d["label"])}
    out = {}
    for c in range(2):
        aid = A["ids"][c]
        errs = {k: [] for k in ("raw", "mapped", "earlier", "later")}
        for k in np.where(d["car"] == c)[0]:
            a = int(d["frame"][k]); t = int(d["tick"][k])
            if a + 4 >= len(fr) or A["ua"][a, c] != 0 or A["ua"][a + 4, c] != 0 or np.any(A["dt"][a:a + 4] != 4):
                continue
            if not (A["ground"][a, c] and A["ground"][a + 4, c]):
                continue
            ph = fr[a].state.cars[aid].physics
            if float(np.dot(ph.linear_velocity, ph.forward)) <= 300:
                continue
            need = [t - 8, t, t + 8, t + 16]
            if not all((c, x) in lab for x in need) or any(T[lab[(c, x)], 5] > .5 for x in need):
                continue
            plans = {"mapped": [lab[(c, t)], lab[(c, t + 8)]], "earlier": [lab[(c, t + 8)], lab[(c, t + 16)]],
                     "later": [lab[(c, t - 8)], lab[(c, t)]]}
            truth = fr[a + 4].state.cars[aid].physics
            for mode in ("raw", "mapped", "earlier", "later"):
                eng.set_state(fr[a].state, {})
                acts = {x: np.repeat(A["act"][a:a + 4, cc], 4, axis=0) for cc, x in enumerate(A["ids"])}
                if mode != "raw":
                    acts[aid] = np.concatenate([np.repeat(T[r][None], 8, axis=0) for r in plans[mode]])
                p = eng.step(acts, {}).cars[aid].physics
                dp = float(np.linalg.norm(np.asarray(p.position) - np.asarray(truth.position)))
                dh = float(np.degrees(np.arccos(np.clip(np.dot(p.forward, truth.forward), -1, 1))))
                errs[mode].append((dp, dh))
        med = {k: (np.median(np.array(v), axis=0).round(2).tolist() if v else None) for k, v in errs.items()}
        obs_r, n1 = steer_metric(d, c)
        raw_r, n2 = steer_metric(d, c, obs_y=False, fr=fr, A=A)
        mp, mh = med["mapped"] if med["mapped"] else (np.inf, np.inf)
        ok = (len(errs["mapped"]) > 20 and mp <= 40 and mh <= 5
              and mh < med["earlier"][1] and mh < med["later"][1] and abs(obs_r - raw_r) <= 0.15)
        out[f"player{c}"] = {"team": int(A["team"][c]), "segments": len(errs["mapped"]),
                             "median_pos_uu_heading_deg": med,
                             "steer_metric_obs": round(obs_r, 4), "steer_metric_raw_physics": round(raw_r, 4),
                             "n": n1, "pass": bool(ok)}
    return {"match": m, **out, "pass": all(v["pass"] for k, v in out.items())}


def reproduce_c2():
    sel = json.loads((ROOT / "external" / "hlrlrd" / "selection.json").read_text())
    picks = [m for m in sel["matches"] if m["split"] == "train"][:4]
    xs = {0: [], 1: []}; ys = {0: [], 1: []}
    for m in picks:
        d, _ = hlconv.extract(ROOT / "external" / "hlrlrd" / m["path"], scheme="dither", seed=m["index"])
        for i, j in checks.next_pairs(d):
            o, n = d["obs"][i], d["obs"][j]
            if o[F["self.on_ground[0]"]] > .5 and n[F["self.on_ground[0]"]] > .5 and o[F["self.lin_vel_local[0]"]] * S.LIN_VEL_SCALE > 300:
                xs[int(d["team"][i])].append(d["steer_mean"][i]); ys[int(d["team"][i])].append(n[F["self.ang_vel_local[2]"]])
    return {f"team{t}": {"r": round(checks.corr(xs[t], ys[t]), 4), "n": len(xs[t])} for t in (0, 1)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / "external" / "hlrlrd" / "selection.json").read_text())
    picks = [m for m in sel["matches"] if m["split"] == "train"][4:7]
    res = {"c2_reproduction": reproduce_c2(), "matches": [m["index"] for m in picks], "R1": [], "R2": []}
    print("C2 reproduction:", res["c2_reproduction"], flush=True)
    for m in picks:
        fr = hlconv.load_frames(ROOT / "external" / "hlrlrd" / m["path"])
        d0, st0, _, A0 = hlconv.extract_frames(fr, scheme="nearest", seed=m["index"], keep_frames=True)
        d1, st1, _, A1 = hlconv.extract_frames(rotate_frames(fr), scheme="nearest", seed=m["index"], keep_frames=True)
        r = r1_compare(m["index"], d0, A0, d1, A1)
        res["R1"].append(r)
        print(f"R1 match {m['index']}: pass {r['pass']}, blocks {r['blocks_orig']}/{r['blocks_rot']}, borderline {r['borderline']}, "
              f"max obs diff {r['max_obs_abs_diff']:.2e}, divergences {r['n_divergences']}, first {r['first_divergence']}, "
              f"steer {r['steer_metric']}", flush=True)
        q = r2(m["index"], d0, fr, A0)
        res["R2"].append(q)
        print(f"R2 match {m['index']}: {json.dumps(q)}", flush=True)
    res["R1_pass"] = all(r["pass"] for r in res["R1"])
    res["R2_pass"] = all(r["pass"] for r in res["R2"])
    (OUT / "replacement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("R1 pass", res["R1_pass"], "R2 pass", res["R2_pass"])
    return 0 if res["R1_pass"] and res["R2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
