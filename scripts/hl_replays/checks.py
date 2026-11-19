"""Initial label checks C1-C5 on 4 converted training-split matches (docs/EXPERIMENT_HL_IMITATION.md).

    .\\.venv\\Scripts\\python.exe scripts\\hl_replays\\checks.py

Writes runs/hl-check/checks.json and prints the passages. Exit code 0 = all checks pass, 1 = a check failed.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hlconv  # noqa: E402
from hlconv import ROOT, T, build_obs, causal_state  # noqa: E402

from botcore.deploy.live_report import feature_names  # noqa: E402
from botcore.obs import spec as S  # noqa: E402

OUT = ROOT / "runs" / "hl-check"
NAMES = feature_names()
F = {n: i for i, n in enumerate(NAMES)}
N_MATCHES, SIM_STARTS, SIM_BLOCKS = 4, 300, 7


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else float("nan")


def next_pairs(d):
    """Index pairs (i, j): same car, j is the block right after i (8 ticks later)."""
    key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(d["car"], d["tick"]))}
    return [(i, key[(int(c), int(t) + 8)]) for i, (c, t) in enumerate(zip(d["car"], d["tick"])) if (int(c), int(t) + 8) in key]


def s1_reference_sign():
    s1 = np.load(ROOT / "runs" / "demos" / "s1.npz")
    o, lab, fr, m = s1["obs"], s1["label"], s1["frame"], s1["match"]
    key = {(int(mm), int(f)): i for i, (mm, f) in enumerate(zip(m, fr))}
    ii = [(i, key[(int(mm), int(f) + 8)]) for i, (mm, f) in enumerate(zip(m, fr)) if (int(mm), int(f) + 8) in key]
    i, j = np.array(ii).T
    g = (o[i, F["self.on_ground[0]"]] > .5) & (o[j, F["self.on_ground[0]"]] > .5) & (o[i, F["self.lin_vel_local[0]"]] * S.LIN_VEL_SCALE > 300)
    return float(np.sign(corr(T[lab[i[g]], 1], o[j[g], F["self.ang_vel_local[2]"]]))), s1


def c1(ds, s1):
    ref = s1["obs"]
    lo, hi = ref.min(0), ref.max(0)
    w = 0.1 * (hi - lo)
    allobs = np.concatenate([d["obs"] for d in ds])
    out = (allobs < lo - w) | (allobs > hi + w)
    per = out.mean(0)
    top = [(NAMES[k], round(float(per[k]), 4)) for k in np.argsort(-per)[:6] if per[k] > 0]
    return {"blocks": int(len(allobs)), "obs_size": int(allobs.shape[1]), "finite": bool(np.isfinite(allobs).all()),
            "share_outside_widened_s1_range": float(out.mean()), "worst_features": top,
            "pass": bool(allobs.shape[1] == 276 and np.isfinite(allobs).all() and out.mean() < 0.001)}


def c2(ds, extras, sign):
    res = {}
    for team in (0, 1):
        st, yaw, th, acc = [], [], [], []
        for d in ds:
            for i, j in next_pairs(d):
                if d["team"][i] != team:
                    continue
                o, n = d["obs"][i], d["obs"][j]
                grounded = o[F["self.on_ground[0]"]] > .5 and n[F["self.on_ground[0]"]] > .5
                v = o[F["self.lin_vel_local[0]"]] * S.LIN_VEL_SCALE
                if grounded and v > 300:
                    st.append(d["steer_mean"][i]); yaw.append(n[F["self.ang_vel_local[2]"]])
                if grounded and 0 <= v < 1400 and d["boost_mean"][i] == 0:
                    th.append(d["throttle_mean"][i])
                    acc.append((n[F["self.lin_vel_local[0]"]] - o[F["self.lin_vel_local[0]"]]) * S.LIN_VEL_SCALE)
        res[f"team{team}"] = {"steer_vs_next_yaw_rate": corr(st, yaw), "n_steer": len(st),
                              "throttle_vs_forward_accel": corr(th, acc), "n_throttle": len(th)}
    sy = [sign * res[f"team{t}"]["steer_vs_next_yaw_rate"] for t in (0, 1)]
    ta = [res[f"team{t}"]["throttle_vs_forward_accel"] for t in (0, 1)]
    res["reference_sign_from_s1_demos"] = sign
    # causality: corrupting every frame after t leaves the observation at t unchanged
    fr, A, d = extras
    rng = np.random.default_rng(0)
    picks = rng.choice(len(d["frame"]), 60, replace=False)
    identical = 0
    for k in picks:
        a, c = int(d["frame"][k]), int(d["car"][k])
        fr2 = list(fr[:a + 1]) + [copy.deepcopy(fr[rng.integers(0, len(fr))]) for _ in range(len(fr) - a - 1)]
        A2 = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in A.items()}
        for kk in ("ua", "act", "ground", "ball"):
            A2[kk][a + 1:] = rng.permutation(A2[kk][a + 1:])
        A2["tick"][a + 1:] += 1000
        o2 = build_obs(causal_state(fr2, A2, a), A["ids"][c], prev_action=T[d["prev"][k]])
        identical += bool(np.array_equal(o2, d["obs"][k]))
    res["causality_identical"] = f"{identical}/{len(picks)}"
    res["pass"] = bool(all(x > 0.5 for x in sy) and abs(sy[0] - sy[1]) < 0.15 and all(x > 0.3 for x in ta)
                       and identical == len(picks))
    return res


def c3(matches):
    from rlgym.rocket_league.sim import RocketSimEngine

    eng = RocketSimEngine()
    rows = []
    for (dd, dn, fr, A) in matches:
        lab = {s: {(int(c), int(t)): int(r) for c, t, r in zip(d["car"], d["tick"], d["label"])} for s, d in
               (("dither", dd), ("nearest", dn))}
        rng = np.random.default_rng(1)
        cand = [k for k in range(len(dd["frame"])) if A["ua"][dd["frame"][k], dd["car"][k]] == 0
                and A["ground"][dd["frame"][k], dd["car"][k]]]
        for k in rng.permutation(cand)[:SIM_STARTS]:
            a, c = int(dd["frame"][k]), int(dd["car"][k])
            ticks = [int(A["tick"][a]) + 8 * w for w in range(SIM_BLOCKS)]
            if not all((c, t) in lab["dither"] and (c, t) in lab["nearest"] for t in ticks):
                continue
            if np.any(A["dt"][a:a + 2 * SIM_BLOCKS] != 4):
                continue
            res = {}
            for mode in ("raw", "dither", "nearest"):
                eng.set_state(fr[a].state, {})
                st, grounded = None, True
                for w in range(SIM_BLOCKS):
                    f0 = a + 2 * w
                    acts = {aid: np.repeat(A["act"][f0:f0 + 2, cc], 4, axis=0) for cc, aid in enumerate(A["ids"])}
                    if mode != "raw":
                        acts[A["ids"][c]] = np.repeat(T[lab[mode][(c, ticks[w])]][None], 8, axis=0)
                    st = eng.step(acts, {})
                    grounded &= bool(st.cars[A["ids"][c]].on_ground)
                p = st.cars[A["ids"][c]].physics
                res[mode] = (np.asarray(p.position, float), np.asarray(p.forward, float), grounded)
            jumps = any(T[lab[s][(c, t)], 5] > .5 for s in ("dither", "nearest") for t in ticks) or \
                A["act"][a:a + 2 * SIM_BLOCKS, c, 5].max() > 0
            steer_int = float(np.mean(np.abs(A["act"][a:a + 2 * SIM_BLOCKS, c, 1])))
            partial = float(np.mean((np.abs(A["act"][a:a + 2 * SIM_BLOCKS, c, 1]) > 0.1) &
                                    (np.abs(A["act"][a:a + 2 * SIM_BLOCKS, c, 1]) < 0.9)))
            row = {"grounded": res["raw"][2] and not jumps, "steer_abs_mean": steer_int, "partial_steer_share": partial}
            for mode in ("dither", "nearest"):
                row[f"{mode}_pos"] = float(np.linalg.norm(res[mode][0] - res["raw"][0]))
                row[f"{mode}_head"] = float(np.degrees(np.arccos(np.clip(np.dot(res[mode][1], res["raw"][1]), -1, 1))))
            rows.append(row)
    g = [r for r in rows if r["grounded"]]
    q = lambda key, rs: [round(float(np.quantile([r[key] for r in rs], x)), 1) for x in (0.5, 0.9)] if rs else None
    out = {"sims": len(rows), "grounded_sims": len(g), "horizon_ticks": 8 * SIM_BLOCKS}
    for mode in ("dither", "nearest"):
        out[mode] = {"grounded_pos_median_p90": q(f"{mode}_pos", g), "grounded_heading_deg_median_p90": q(f"{mode}_head", g),
                     "all_pos_median_p90": q(f"{mode}_pos", rows), "all_heading_deg_median_p90": q(f"{mode}_head", rows)}
        hi = [r for r in g if r["partial_steer_share"] >= 0.5]
        out[mode]["grounded_mostly_partial_steer_pos_median"] = q(f"{mode}_pos", hi)[0] if hi else None
    out["n_grounded_mostly_partial_steer"] = len([r for r in g if r["partial_steer_share"] >= 0.5])
    chosen = "dither" if out["dither"]["grounded_pos_median_p90"][0] < out["nearest"]["grounded_pos_median_p90"][0] else "nearest"
    out["chosen_scheme"] = chosen
    out["pass_steering"] = bool(out[chosen]["grounded_pos_median_p90"][0] <= 100
                                and out[chosen]["grounded_heading_deg_median_p90"][0] <= 10)
    return out, chosen


def c3_boost_dodge(dsel, matches_A):
    agree, consume_agree, n_cons = [], 0, 0
    ang, n_dir = [], 0
    kept = indg = 0
    for d, (fr, A, st) in zip(dsel, matches_A):
        lb = T[d["label"], 6] > .5
        agree += list(lb == (d["boost_mean"] >= .5))
        for k in range(len(d["frame"])):
            a, c = int(d["frame"][k]), int(d["car"][k])
            b0, b1 = fr[a].state.cars[A["ids"][c]].boost_amount, fr[a + 2].state.cars[A["ids"][c]].boost_amount
            if b0 >= 1 and b1 <= b0:
                n_cons += 1
                consume_agree += bool(lb[k] == (b1 < b0 - 0.5))
        for k in np.where(d["dodge"] & ~d["approx_dir"])[0]:
            r = T[d["label"][k]]
            # the press stick of this block (the replay's dodge direction): converter action at the press frame
            a, c = int(d["frame"][k]), int(d["car"][k])
            off = d["press_offset"][k]
            f = a + (int(off) // 4 if not np.isnan(off) else 0)
            stick = A["act"][f, c, 2:5] if 0 <= f < len(A["act"]) else A["act"][a, c, 2:5]
            raw = np.array([stick[0], np.clip(stick[1] + stick[2], -1, 1)])
            mapped = np.array([r[2], r[4]])
            if np.linalg.norm(raw) < 0.2 or np.linalg.norm(mapped) == 0:
                continue
            n_dir += 1
            cosv = np.dot(raw, mapped) / np.linalg.norm(raw) / np.linalg.norm(mapped)
            ang.append(float(np.degrees(np.arccos(np.clip(cosv, -1, 1)))))
        kept += st["dodges_pattern_kept"]; indg += st["dodges_in_grid"]
    ang = np.array(ang)
    out = {"boost_bit_agrees_with_block_majority": float(np.mean(agree)),
           "boost_bit_agrees_with_replay_boost_consumption (descriptive)": consume_agree / max(n_cons, 1),
           "dodge_direction_within_45deg": float(np.mean(ang <= 45)) if len(ang) else None, "dodges_direction_checked": n_dir,
           "dodge_direction_median_deg": float(np.median(ang)) if len(ang) else None,
           "dodge_pattern_kept": f"{kept}/{indg}"}
    out["pass"] = bool(out["boost_bit_agrees_with_block_majority"] >= 0.95 and out["dodge_direction_within_45deg"] is not None
                       and out["dodge_direction_within_45deg"] >= 0.90 and indg > 0 and kept / indg >= 0.90)
    return out


def c4(stats, ds):
    post = sum(s["blocks_postkickoff"] for s in stats)
    usable = sum(int(((~d["press_inflight"]) & (~d["dodge_lost"])).sum()) for d in ds)
    return {"blocks_postkickoff": post, "blocks_after_exclusions": sum(s["blocks_usable"] for s in stats),
            "blocks_used_by_loader": usable, "usable_share": usable / post,
            "excluded_repro": sum(s["excl_repro"] for s in stats), "excluded_age": sum(s["excl_age"] for s in stats),
            "repro_error_median_uu_deg": [s["repro_err_median"] for s in stats], "pass": bool(usable / post >= 0.5)}


def passages(dd, dn, fr, A):
    lines = []
    T_ = lambda r: "".join({-1: "-", 0: "0", 1: "+"}[int(v)] for v in T[r][:2])
    kn = {(int(c), int(t)): int(r) for c, t, r in zip(dn["car"], dn["tick"], dn["label"])}
    for team in (0, 1):
        c = int(np.where(A["team"] == team)[0][0])
        idx = np.where(dd["car"] == c)[0]
        for s in range(len(idx) - 6):
            run = idx[s:s + 6]
            if (np.all(np.diff(dd["tick"][run]) == 8) and all(A["ground"][dd["frame"][k], c] for k in run)
                    and np.mean(np.abs(dd["steer_mean"][run]) > 0.15) > 0.8 and np.mean(np.abs(dd["steer_mean"][run]) < 0.85) > 0.5
                    and dd["boost_mean"][run].max() == 0):
                lines.append(f"\nGrounded steering, {'blue' if team == 0 else 'orange'} perspective (tick, analog throttle/steer per frame, "
                             f"block mean steer, nearest row thr/str, dithered row thr/str, speed uu/s, local yaw rate next block)")
                for k in run:
                    a = int(dd["frame"][k])
                    sp = np.linalg.norm(fr[a].state.cars[A['ids'][c]].physics.linear_velocity)
                    nxt = [j for j in idx if dd["tick"][j] == dd["tick"][k] + 8]
                    yr = dd["obs"][nxt[0], F["self.ang_vel_local[2]"]] if nxt else float("nan")
                    lines.append(f"  {dd['tick'][k]:6d}  thr {A['act'][a, c, 0]:+.2f}/{A['act'][a + 1, c, 0]:+.2f}  str {A['act'][a, c, 1]:+.2f}/{A['act'][a + 1, c, 1]:+.2f}"
                                 f"  mean {dd['steer_mean'][k]:+.2f}  nearest {T_(kn[(c, int(dd['tick'][k]))])}  dither {T_(dd['label'][k])}"
                                 f"  speed {sp:5.0f}  yaw-rate {yr:+.3f}")
                break
    c = 0
    idx = np.where(dd["car"] == c)[0]
    for s in range(1, len(idx) - 6):
        k = idx[s]
        if dd["boost_mean"][k] > 0 and dd["boost_mean"][idx[s - 1]] == 0 and np.all(np.diff(dd["tick"][idx[s - 1:s + 5]]) == 8):
            lines.append("\nBoost use (tick, boost button per frame, label boost bit, boost amount at block start)")
            for kk in idx[s - 1:s + 5]:
                a = int(dd["frame"][kk])
                lines.append(f"  {dd['tick'][kk]:6d}  button {A['act'][a, c, 6]:.0f}{A['act'][a + 1, c, 6]:.0f}  label {T[dd['label'][kk], 6]:.0f}"
                             f"  boost {fr[a].state.cars[A['ids'][c]].boost_amount:5.1f}")
            break
    shown = 0
    for team in (0, 1):
        c = int(np.where(A["team"] == team)[0][0])
        idx = np.where(dd["car"] == c)[0]
        for s in range(3, len(idx) - 3):
            k = idx[s]
            if dd["dodge"][k] and not dd["press_inflight"][k] and np.all(np.diff(dd["tick"][idx[s - 3:s + 3]]) == 8):
                lines.append(f"\nOrdinary dodge, {'blue' if team == 0 else 'orange'} perspective (tick, jump per frame, stick p/y/r at frames, "
                             f"label jump/pitch/yaw/roll, on_ground, has_flipped)")
                for kk in idx[s - 3:s + 3]:
                    a = int(dd["frame"][kk])
                    r = T[dd["label"][kk]]
                    lines.append(f"  {dd['tick'][kk]:6d}  jump {A['act'][a, c, 5]:.0f}{A['act'][a + 1, c, 5]:.0f}  stick {np.round(A['act'][a, c, 2:5], 2)}"
                                 f"  label J{r[5]:.0f} p{r[2]:+.0f} y{r[3]:+.0f} r{r[4]:+.0f}  ground {int(A['ground'][a, c])}  flipped {int(A['flipped'][a, c])}")
                shown += 1
                break
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / "external" / "hlrlrd" / "selection.json").read_text())
    picks = [m for m in sel["matches"] if m["split"] == "train"][:N_MATCHES]
    sign, s1 = s1_reference_sign()
    ds, dns, mats, stats, frA = [], [], [], [], []
    for m in picks:
        path = ROOT / "external" / "hlrlrd" / m["path"]
        dd, st, fr, A = hlconv.extract(path, scheme="dither", seed=m["index"], keep_frames=True)
        dn, _ = hlconv.extract(path, scheme="nearest", seed=m["index"])
        ds.append(dd); dns.append(dn); mats.append((dd, dn, fr, A)); stats.append(st); frA.append((fr, A, st))
        print(f"match {m['index']}: {st}", flush=True)
    res = {"matches": [m["index"] for m in picks]}
    res["C1"] = c1(ds, s1)
    res["C2"] = c2(ds, (mats[0][2], mats[0][3], ds[0]), sign)
    res["C3_steering"], chosen = c3(mats)
    dsel = ds if chosen == "dither" else dns
    res["C3_boost_dodge"] = c3_boost_dodge(dsel, frA)
    res["C4"] = c4(stats, dsel)
    res["C3"] = {"pass": bool(res["C3_steering"]["pass_steering"] and res["C3_boost_dodge"]["pass"])}
    res["all_pass"] = bool(res["C1"]["pass"] and res["C2"]["pass"] and res["C3"]["pass"] and res["C4"]["pass"])
    txt = passages(mats[0][0] if chosen == "dither" else mats[0][1], mats[0][1], mats[0][2], mats[0][3])
    (OUT / "checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (OUT / "passages.txt").write_text(txt, encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(txt)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
