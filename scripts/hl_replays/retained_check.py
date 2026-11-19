"""Retained-side (non-recorder) checks N1-N4 (docs/EXPERIMENT_HL_IMITATION.md, "Retained-side pilot").

    .\\.venv\\Scripts\\python.exe scripts\\hl_replays\\retained_check.py

Writes runs/hl-check/retained.json (and retained_failure.txt on failure). Exit code 0 = all pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import checks  # noqa: E402
import hlconv  # noqa: E402
import recorder  # noqa: E402
from hlconv import ROOT, T  # noqa: E402
from rotation_check import steer_metric  # noqa: E402

OUT = ROOT / "runs" / "hl-check"
CHECK_INDICES = (7, 8, 9)
MAX_LAG = 8


def segments(d, fr, A, c):
    """R2 segments of car c, with per-mode (pos, heading) errors and the N2 class."""
    from rlgym.rocket_league.sim import RocketSimEngine

    eng = RocketSimEngine()
    lab = {(int(cc), int(t)): int(r) for cc, t, r in zip(d["car"], d["tick"], d["label"])}
    aid = A["ids"][c]
    out = []
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
        rows = [lab[(c, x)] for x in need]
        plans = {"mapped": rows[1:3], "earlier": rows[2:4], "later": rows[0:2]}
        truth = fr[a + 4].state.cars[aid].physics
        err = {}
        for mode in ("raw", "mapped", "earlier", "later"):
            eng.set_state(fr[a].state, {})
            acts = {x: np.repeat(A["act"][a:a + 4, cc], 4, axis=0) for cc, x in enumerate(A["ids"])}
            if mode != "raw":
                acts[aid] = np.concatenate([np.repeat(T[r][None], 8, axis=0) for r in plans[mode]])
            p = eng.step(acts, {}).cars[aid].physics
            err[mode] = (float(np.linalg.norm(np.asarray(p.position) - np.asarray(truth.position))),
                         float(np.degrees(np.arccos(np.clip(np.dot(p.forward, truth.forward), -1, 1)))))
        st = [float(T[r, 1]) for r in rows]
        kind = "change" if len(set(st)) > 1 else ("sustained" if st[0] != 0 else "straight")
        out.append({"tick": t, "frame": a, "rows": rows, "steer_rows": st, "kind": kind, "err": err})
    return out


def med(segs, mode):
    if not segs:
        return None
    return np.median(np.array([s["err"][mode] for s in segs]), axis=0).round(3).tolist()


def timing_ok(segs, idx):
    m, e, l = med(segs, "mapped"), med(segs, "earlier"), med(segs, "later")
    return bool(m[idx] < e[idx] and m[idx] < l[idx])


def best_lag(fr, A, c):
    up = np.array([np.asarray(f.state.cars[A["ids"][c]].physics.rotation_mtx)[:, 2] for f in fr])
    av = np.array([np.asarray(f.state.cars[A["ids"][c]].physics.angular_velocity) for f in fr])
    yaw = np.einsum("ij,ij->i", up, av)
    fwd = np.array([float(np.dot(f.state.cars[A["ids"][c]].physics.linear_velocity,
                                 f.state.cars[A["ids"][c]].physics.forward)) for f in fr])
    base = np.where((A["ua"][:, c] == 0) & A["ground"][:, c] & (fwd > 300))[0]
    rs = []
    for L in range(MAX_LAG + 1):
        ks = [k for k in base if k - L >= 0 and A["period_start"][k - L] == A["period_start"][k]
              and np.all(A["dt"][k - L:k] == 4)]
        rs.append(abs(checks.corr(A["act"][np.array(ks) - L, c, 1], yaw[ks])))
    return int(np.nanargmax(rs)), [round(float(r), 4) for r in rs]


def sub(d, mask):
    return {k: v[mask] for k, v in d.items()}


def dodge_boost(d, fr, A, st, c):
    dc = sub(d, d["car"] == c)
    kept, ingrid = st["dodges_in_grid_kept_by_car"][c][1], st["dodges_in_grid_kept_by_car"][c][0]
    st2 = dict(st, dodges_pattern_kept=kept, dodges_in_grid=ingrid)
    return checks.c3_boost_dodge([dc], [(fr, A, st2)]), (ingrid, kept)


def counts(d, st, c):
    use = (d["car"] == c) & ~d["press_inflight"] & ~d["dodge_lost"]
    return {"blocks_usable_by_loader": int(use.sum()), "usable_dodge_blocks": int((use & d["dodge"]).sum()),
            "dodges_in_grid": st["dodges_in_grid_kept_by_car"][c][0],
            "dodges_in_grid_kept": st["dodges_in_grid_kept_by_car"][c][1]}


def failure_text(m, team, kind, segs, fr, A, c, d):
    bad = [s for s in segs if s["err"]["mapped"][1] > min(s["err"]["earlier"][1], s["err"]["later"][1])]
    if not bad:
        return f"match {m} retained {team}: {kind} criterion failed on medians; no single segment has aligned worse than shifted"
    s = bad[0]
    a = s["frame"]
    lines = [f"First failing sequence: match {m}, retained player ({team}), {kind} segment at tick {s['tick']}",
             "  frame  tick   analog throttle  analog steer   (converter controls, 4 ticks per frame)"]
    for f in range(max(a - 2, 0), min(a + 6, len(fr))):
        lines.append(f"  {f:6d} {A['tick'][f]:6d}   {A['act'][f, c, 0]:+.2f}            {A['act'][f, c, 1]:+.2f}"
                     + ("   <- segment start" if f == a else ""))
    lines.append("  mapped rows (thr, steer) at t-8, t, t+8, t+16: "
                 + ", ".join(f"({T[r, 0]:+.0f},{T[r, 1]:+.0f})" for r in s["rows"]))
    for mode in ("raw", "mapped", "earlier", "later"):
        lines.append(f"  16-tick re-simulation {mode:8s}: position {s['err'][mode][0]:6.1f} uu, heading {s['err'][mode][1]:5.2f} deg")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ident = {r["index"]: r for r in recorder.identify_selection(ROOT)}
    sel = json.loads((ROOT / "external" / "hlrlrd" / "selection.json").read_text())
    picks = [m for m in sel["matches"] if m["index"] in CHECK_INDICES]
    assert all(m["split"] == "train" and ident[m["index"]]["recorder_orange"] is not None for m in picks)
    res = {"matches": list(CHECK_INDICES), "players": []}
    all_segs, failures = [], []
    for m in picks:
        fr = hlconv.load_frames(ROOT / "external" / "hlrlrd" / m["path"])
        d, st, fr, A = hlconv.extract_frames(fr, scheme="nearest", seed=m["index"], keep_frames=True)
        rt = ident[m["index"]]["retained_team"]
        c = int(np.where(A["team"] == rt)[0][0])
        team = "blue" if rt == 0 else "orange"
        segs = segments(d, fr, A, c)
        all_segs += [(m["index"], team, s) for s in segs]
        ch = [s for s in segs if s["kind"] == "change"]
        su = [s for s in segs if s["kind"] == "sustained"]
        obs_r, _ = steer_metric(d, c)
        raw_r, _ = steer_metric(d, c, obs_y=False, fr=fr, A=A)
        mp = med(segs, "mapped")
        n1 = bool(len(segs) > 20 and mp[0] <= 40 and mp[1] <= 5 and timing_ok(segs, 1) and abs(obs_r - raw_r) <= 0.15)
        lag_ret, rs_ret = best_lag(fr, A, c)
        lag_rec, rs_rec = best_lag(fr, A, 1 - c)
        n4, dg = dodge_boost(d, fr, A, st, c)
        p = {"match": m["index"], "retained_team": team, "counts": counts(d, st, c),
             "N1": {"segments": len(segs), "median_pos_uu_heading_deg": {k: med(segs, k) for k in ("raw", "mapped", "earlier", "later")},
                    "steer_metric_obs": round(obs_r, 4), "steer_metric_raw_physics": round(raw_r, 4), "pass": n1},
             "N2_change": {"segments": len(ch), "median_pos_uu_heading_deg": {k: med(ch, k) for k in ("raw", "mapped", "earlier", "later")},
                           "heading_ok": timing_ok(ch, 1) if ch else None, "position_ok": timing_ok(ch, 0) if ch else None},
             "N2_sustained": {"segments": len(su), "median_pos_uu_heading_deg": {k: med(su, k) for k in ("raw", "mapped", "earlier", "later")}},
             "N2_straight_segments": sum(s["kind"] == "straight" for s in segs),
             "N3": {"retained_best_lag_frames": lag_ret, "retained_abs_r_by_lag": rs_ret,
                    "recorder_best_lag_frames": lag_rec, "recorder_abs_r_by_lag": rs_rec, "pass": lag_ret <= 3},
             "N4": n4}
        res["players"].append(p)
        print(json.dumps(p), flush=True)
        if not n1:
            failures.append(("N1", m["index"], team, "all", segs, fr, A, c, d))
        if len(ch) >= 20 and not (p["N2_change"]["heading_ok"] and p["N2_change"]["position_ok"]):
            failures.append(("N2 change", m["index"], team, "change", ch, fr, A, c, d))
        if not p["N3"]["pass"]:
            failures.append(("N3", m["index"], team, "lag", segs, fr, A, c, d))
        if not n4["pass"]:
            failures.append(("N4", m["index"], team, "dodge", segs, fr, A, c, d))
    # N2 pooled fallback and sustained turns
    few = [p for p in res["players"] if p["N2_change"]["segments"] < 20]
    pooled_ch = [s for _, _, s in all_segs if s["kind"] == "change"]
    pooled_su = [s for _, _, s in all_segs if s["kind"] == "sustained"]
    res["N2_pooled_change"] = {"segments": len(pooled_ch), "median_pos_uu_heading_deg": {k: med(pooled_ch, k) for k in ("raw", "mapped", "earlier", "later")},
                               "used_for_players": [p["match"] for p in few],
                               "pass": bool(timing_ok(pooled_ch, 0) and timing_ok(pooled_ch, 1)) if pooled_ch else False}
    if few and not res["N2_pooled_change"]["pass"]:
        failures.append(("N2 pooled change", None, None, "change", pooled_ch, None, None, None, None))
    sm = med(pooled_su, "mapped")
    res["N2_pooled_sustained"] = {"segments": len(pooled_su),
                                  "median_pos_uu_heading_deg": {k: med(pooled_su, k) for k in ("raw", "mapped", "earlier", "later")},
                                  "pass": bool(sm is not None and sm[0] <= 40 and sm[1] <= 5)}
    if not res["N2_pooled_sustained"]["pass"]:
        failures.append(("N2 sustained", None, None, "sustained", pooled_su, None, None, None, None))
    res["failures"] = [f[0] + (f" match {f[1]} {f[2]}" if f[1] is not None else "") for f in failures]
    res["all_pass"] = not failures
    (OUT / "retained.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "players"}, indent=2))
    if failures:
        name, m, team, kind, segs, fr, A, c, d = failures[0]
        txt = (failure_text(m, team, kind, segs, fr, A, c, d) if fr is not None
               else f"{name}: pooled criterion failed: {res.get('N2_pooled_change') if 'change' in name else res['N2_pooled_sustained']}")
        (OUT / "retained_failure.txt").write_text(txt, encoding="utf-8")
        print(txt)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
