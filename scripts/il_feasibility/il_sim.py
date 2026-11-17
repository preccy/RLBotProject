"""IL feasibility step 4: re-simulate recorded human kickoffs and in-play dodges in RocketSim with
(a) RAW per-tick human controls, (b) EVENT-aligned 90-row/8-tick mapping, (c) NAIVE sample-and-hold mapping.
The bot car always replays its own recorded per-tick inputs. Initial states come from the recorded packets.

python il_sim.py <dir with il_data.pkl>
"""
import pickle
import sys
from pathlib import Path

import numpy as np

R = Path(r"C:\Users\TomPr\RLBotProject")
sys.path.insert(0, str(R / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from rlgym.rocket_league.sim import RocketSimEngine  # noqa: E402

from botcore import constants as C  # noqa: E402
from botcore.deploy.packets import CompatObsPipeline  # noqa: E402
from botcore.deploy.recording import load_recording  # noqa: E402
from botcore.scenarios.mutators import _new_car  # noqa: E402
from il_map import T, ctl, map_period  # noqa: E402

D = Path(sys.argv[1])
data = pickle.load(open(D / "il_data.pkl", "rb"))
rec = load_recording(R / "runs/live/human-argmax-11900/botcore-team1-index0.bcp")
P = [p for p in rec.packets if len(p.players) >= 2 and p.balls]
HP, BP = data["hum_pid"], data["bot_pid"]


def player(p, pid):
    return next(x for x in p.players if x.player_id == pid)


# per-frame (forward-filled) recorded series
f0, f1 = P[0].match_info.frame_num, P[-1].match_info.frame_num
by_frame = {p.match_info.frame_num: i for i, p in enumerate(P)}


def series(start_f, n):
    """Recorded per-frame data for frames start_f..start_f+n (forward-filled over missing packets)."""
    out = {"Uh": [], "Ub": [], "hpos": [], "hvel": [], "bpos": [], "ball": [], "hdodge": [], "hground": [], "hfwd": [],
           "hright": [], "htouch": [], "missing": 0}
    last = None
    for f in range(start_f, start_f + n + 1):
        i = by_frame.get(f)
        if i is None:
            out["missing"] += 1
            i = last
        last = i
        p = P[i]
        h, b = player(p, HP), player(p, BP)
        out["Uh"].append(ctl(h.last_input)); out["Ub"].append(ctl(b.last_input))
        L = h.physics.location; V = h.physics.velocity
        out["hpos"].append([L.x, L.y, L.z]); out["hvel"].append([V.x, V.y, V.z])
        out["bpos"].append([b.physics.location.x, b.physics.location.y, b.physics.location.z])
        bl = p.balls[0].physics.location; out["ball"].append([bl.x, bl.y, bl.z])
        out["hdodge"].append(h.has_dodged); out["hground"].append(str(h.air_state).endswith("OnGround"))
        out["htouch"].append(h.latest_touch.game_seconds if h.latest_touch else -1)
    for k in list(out):
        if k != "missing":
            out[k] = np.array(out[k])
    return out


# compat snapshots at the start frames (rotation matrices in the same convention the policy sees)
def snapshots(start_frames):
    pipe = CompatObsPipeline(rec.field)
    snaps = {}
    want = set(start_frames)
    for p in P:
        pipe.update(p)
        f = p.match_info.frame_num
        if f in want:
            s = {}
            for pid, key in ((HP, "human"), (BP, "bot")):
                c = pipe.state.cars[pid]
                pl = player(p, pid)
                s[key] = dict(pos=np.array(c.physics.position), vel=np.array(c.physics.linear_velocity),
                              ang=np.array(c.physics.angular_velocity), rot=np.array(c.physics.rotation_mtx),
                              boost=pl.boost, ground=str(pl.air_state).endswith("OnGround"), team=pl.team,
                              has_jumped=pl.has_jumped, has_dj=pl.has_double_jumped, has_flipped=pl.has_dodged,
                              demo=pl.demolished_timeout > 0)
            bl = p.balls[0].physics
            s["ball"] = dict(pos=np.array([bl.location.x, bl.location.y, bl.location.z]),
                             vel=np.array([bl.velocity.x, bl.velocity.y, bl.velocity.z]),
                             ang=np.array([bl.angular_velocity.x, bl.angular_velocity.y, bl.angular_velocity.z]))
            s["pads"] = np.array(pipe.pads.timers(), dtype=np.float32)  # BOOST_LOCATIONS order
            snaps[f] = s
    return snaps


ENGINE = RocketSimEngine(rlbot_delay=False)
_arena_pads = None


def make_state(s):
    global _arena_pads
    gs = ENGINE.create_base_state()
    for key in ("human", "bot"):
        c = s[key]; car = _new_car(c["team"])
        car.physics.position = c["pos"].astype(np.float32); car.physics.linear_velocity = c["vel"].astype(np.float32)
        car.physics.angular_velocity = c["ang"].astype(np.float32); car.physics.rotation_mtx = c["rot"].astype(np.float32)
        car.boost_amount = float(c["boost"]); car.on_ground = bool(c["ground"])
        car.wheels_with_contact = (bool(c["ground"]),) * 4
        car.has_jumped = bool(c["has_jumped"]); car.has_double_jumped = bool(c["has_dj"]); car.has_flipped = bool(c["has_flipped"])
        gs.cars[key] = car
    gs.ball.position = s["ball"]["pos"].astype(np.float32); gs.ball.linear_velocity = s["ball"]["vel"].astype(np.float32)
    gs.ball.angular_velocity = s["ball"]["ang"].astype(np.float32)
    ENGINE.set_state(gs, {})
    if _arena_pads is None:  # arena pad index -> BOOST_LOCATIONS index, by position
        from rlgym.rocket_league.common_values import BOOST_LOCATIONS; locs = np.array(BOOST_LOCATIONS)[:, :2]
        _arena_pads = [int(np.argmin(np.linalg.norm(locs - np.array([pd.get_pos().x, pd.get_pos().y]), axis=1)))
                       for pd in ENGINE._arena.get_boost_pads()]
    gs.boost_pad_timers = np.array([s["pads"][j] for j in _arena_pads], dtype=np.float32)
    ENGINE.set_state(gs, {})


def simulate(s, Uh, Ub):
    make_state(s)
    out = {"pos": [], "vel": [], "flip": [], "fwd": [], "right": [], "touch": [], "ball": []}
    for k in range(len(Uh)):
        st = ENGINE.step({"human": Uh[k:k + 1].astype(np.float32), "bot": Ub[k:k + 1].astype(np.float32)}, {})
        c = st.cars["human"]
        out["pos"].append(np.array(c.physics.position)); out["vel"].append(np.array(c.physics.linear_velocity))
        out["flip"].append(bool(c.has_flipped)); out["fwd"].append(np.array(c.physics.forward)); out["right"].append(np.array(c.physics.right))
        out["touch"].append(c.ball_touches > 0); out["ball"].append(np.array(st.ball.position))
    return {k: np.array(v) for k, v in out.items()}


def _record(st, out):
    c = st.cars["human"]
    out["pos"].append(np.array(c.physics.position)); out["vel"].append(np.array(c.physics.linear_velocity))
    out["flip"].append(bool(c.has_flipped)); out["fwd"].append(np.array(c.physics.forward)); out["right"].append(np.array(c.physics.right))
    out["touch"].append(c.ball_touches > 0); out["ball"].append(np.array(st.ball.position))


def simulate_blocks(s, Ub, rec_pos, rec_vel, plan_rows=None, raw=None):
    """Closed-loop block-by-block simulation with a state restore at every block boundary.
    plan_rows given: TRACK mode - per block choose the table row (same jump bit as the event plan; any jump row on
    jump blocks) whose 8-tick result best matches the recorded position/velocity. raw given: RAW-RESTORED control."""
    make_state(s)
    out = {"pos": [], "vel": [], "flip": [], "fwd": [], "right": [], "touch": [], "ball": []}
    n = len(Ub)
    rows_used = []
    for b in range(int(np.ceil(n / 8))):
        lo, hi = b * 8, min((b + 1) * 8, n)
        saved = ENGINE._get_state()
        if raw is not None:
            cands = [None]
        else:
            jb = T[plan_rows[b], 5] > 0.5
            cands = [r for r in range(len(T)) if (T[r, 5] > 0.5) == jb]
        best, best_err = None, None
        for r in cands:
            ENGINE.set_state(saved, {})
            Uh = raw[lo:hi] if r is None else np.repeat(T[r][None], hi - lo, axis=0)
            for k in range(hi - lo):
                st = ENGINE.step({"human": Uh[k:k + 1].astype(np.float32), "bot": Ub[lo + k:lo + k + 1].astype(np.float32)}, {})
            c = st.cars["human"]
            e = np.linalg.norm(np.array(c.physics.position) - rec_pos[hi - 1]) + 0.1 * np.linalg.norm(
                np.array(c.physics.linear_velocity) - rec_vel[hi - 1])
            if best_err is None or e < best_err:
                best, best_err = r, e
        ENGINE.set_state(saved, {})
        Uh = raw[lo:hi] if best is None else np.repeat(T[best][None], hi - lo, axis=0)
        rows_used.append(best)
        for k in range(hi - lo):
            st = ENGINE.step({"human": Uh[k:k + 1].astype(np.float32), "bot": Ub[lo + k:lo + k + 1].astype(np.float32)}, {})
            _record(st, out)
    res = {k: np.array(v) for k, v in out.items()}
    res["rows"] = rows_used
    return res


def dodge_metrics(pos_vel_flip, fwd_right=None):
    vel, flip = pos_vel_flip["vel"], pos_vel_flip["flip"]
    idx = np.where(flip)[0]
    if len(idx) == 0:
        return None, None
    k = int(idx[0])
    k2 = min(k + 6, len(vel) - 1)
    dv = vel[k2] - vel[max(k - 1, 0)]
    f, r = fwd_right if fwd_right is not None else (pos_vel_flip["fwd"][max(k - 1, 0)], pos_vel_flip["right"][max(k - 1, 0)])
    return k, float(np.degrees(np.arctan2(dv @ r, dv @ f)))


def first_exceed(err, thr):
    w = np.where(err > thr)[0]
    return int(w[0]) if len(w) else None


def run_segment(label, start_f, n, s, t_end_ref=None, kickoff=False):
    rs = series(start_f, n)
    Uh_raw, Ub = rs["Uh"][1:], rs["Ub"][1:]  # controls of frame k+1 produce state k+1
    ground = rs["hground"][1:]
    ev = T[np.repeat(map_period(Uh_raw, ground, "event"), 8)[:len(Uh_raw)]]
    # diagnostic only (NOT expressible by the table): EVENT timing, but the human's own stick during airborne jump presses
    evdir = ev.copy()
    air_jump = (ev[:, 5] > 0.5) & ~ground
    evdir[air_jump, 2:5] = Uh_raw[air_jump, 2:5]
    variants = {"RAW": Uh_raw, "EVENT": ev, "EVDIR": evdir,
                "NAIVE": T[np.repeat(map_period(Uh_raw, ground, "naive"), 8)[:len(Uh_raw)]]}
    # first tick where a mapped control differs from the raw control on a channel that matters in that state
    first_diff = {}
    for name in ("EVENT", "NAIVE"):
        Um = variants[name]
        rel = np.where(ground[:, None], np.array([1, 1, 0, 0, 0, 1, 1, 1], bool), np.array([0, 0, 1, 1, 1, 1, 1, 0], bool))
        d = (Um != Uh_raw) & rel
        idx = np.where(d.any(axis=1))[0]
        first_diff[name] = None if not len(idx) else (int(idx[0]), [c for c, x in zip(
            ["throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"], d[idx[0]]) if x])
    ball0 = rs["ball"][0]
    moved = np.where(np.linalg.norm(rs["ball"][1:] - ball0, axis=1) > 5)[0]
    rec_ball_touch = int(moved[0]) if len(moved) else None
    rec_pos, rec_vel = rs["hpos"][1:], rs["hvel"][1:]
    # recorded dodge: first frame with has_dodged; direction from the same velocity-change method
    rd = np.where(rs["hdodge"][1:])[0]
    rec_k = int(rd[0]) if len(rd) else None
    lines = []
    if rec_k is not None:
        # recorded car frame at the dodge from the RAW simulation is not available; use velocity heading as forward
        sim_raw = None
    rows = {}
    rec_pos, rec_vel = rs["hpos"][1:], rs["hvel"][1:]
    plan = map_period(Uh_raw, ground, "event")
    variants["RAWRS"] = None
    variants["TRACK"] = None
    for name, Uh in variants.items():
        if name == "RAWRS":
            sim = simulate_blocks(s, Ub, rec_pos, rec_vel, raw=Uh_raw)
        elif name == "TRACK":
            sim = simulate_blocks(s, Ub, rec_pos, rec_vel, plan_rows=plan)
            Uh = Uh_raw
        else:
            sim = simulate(s, Uh, Ub)
        err = np.linalg.norm(sim["pos"] - rec_pos, axis=1)
        k, ang = dodge_metrics(sim)
        dist = np.linalg.norm(sim["pos"] - sim["ball"], axis=1)
        arr = first_exceed(-dist, -220) if kickoff else None  # first tick within 220 uu of the ball
        rows[name] = dict(dodge_tick=k, dodge_angle=ang, div10=first_exceed(err, 10), div100=first_exceed(err, 100),
                          err_at_dodge=None if (k is None or rec_k is None) else float(err[min(rec_k, len(err) - 1)]),
                          arrive=arr, speed_at_arrive=None if arr is None else float(np.linalg.norm(sim["vel"][arr])),
                          speed_1s=float(np.linalg.norm(sim["vel"][min(119, len(sim["vel"]) - 1)])), sim=sim)
    # recorded reference numbers
    rec_dist = np.linalg.norm(rec_pos - rs["ball"][1:], axis=1)
    rec_arr = first_exceed(-rec_dist, -220) if kickoff else None
    car_gap = np.linalg.norm(rec_pos - rs["bpos"][1:], axis=1)
    contact = first_exceed(-car_gap, -200)
    # recorded dodge angle using the RAW sim's car axes at that tick (car orientation is not in this series)
    rec_ang = None
    if rec_k is not None:
        raw = rows["RAW"]["sim"]
        dv = rec_vel[min(rec_k + 6, len(rec_vel) - 1)] - rec_vel[max(rec_k - 1, 0)]
        f, r = raw["fwd"][max(rec_k - 1, 0)], raw["right"][max(rec_k - 1, 0)]
        rec_ang = float(np.degrees(np.arctan2(dv @ r, dv @ f)))
    return dict(label=label, rec_dodge=rec_k, rec_angle=rec_ang, rec_arrive=rec_arr,
                rec_speed_arrive=None if rec_arr is None else float(np.linalg.norm(rec_vel[rec_arr])),
                rec_speed_1s=float(np.linalg.norm(rec_vel[min(119, len(rec_vel) - 1)])),
                car_contact=contact, missing=rs["missing"], variants=rows, first_diff=first_diff,
                rec_ball_touch=rec_ball_touch)


def summarise(rs, kickoff):
    have = [r for r in rs if r["rec_dodge"] is not None]
    print(f"  segments {len(rs)}, with a recorded dodge {len(have)}")
    for name in ("RAW", "RAWRS", "TRACK", "EVENT", "EVDIR", "NAIVE"):
        ok = [r for r in have if r["variants"][name]["dodge_tick"] is not None]
        shift = np.array([r["variants"][name]["dodge_tick"] - r["rec_dodge"] for r in ok])
        aerr = np.array([abs(r["variants"][name]["dodge_angle"] - r["rec_angle"]) for r in ok])
        d10 = [r["variants"][name]["div10"] for r in rs]; d100 = [r["variants"][name]["div100"] for r in rs]
        line = (f"  {name:5s} dodge executed {len(ok)}/{len(have)}; dodge tick shift median {np.median(shift):+.0f} "
                f"(range {shift.min():+d}..{shift.max():+d}); |angle err| median {np.median(aerr):.0f} deg (max {aerr.max():.0f})")
        if kickoff:
            arr = [(r["variants"][name]["arrive"], r["rec_arrive"], r["variants"][name]["speed_at_arrive"], r["rec_speed_arrive"])
                   for r in rs if r["variants"][name]["arrive"] is not None and r["rec_arrive"] is not None]
            da = np.array([a - b for a, b, _, _ in arr]); ds = np.array([c - d for _, _, c, d in arr])
            line += (f"\n        arrival (<220 uu of ball) shift median {np.median(da):+.0f} ticks (range {da.min():+d}..{da.max():+d}), "
                     f"speed at arrival diff median {np.median(ds):+.0f} uu/s (range {ds.min():+.0f}..{ds.max():+.0f}) over {len(arr)} kickoffs")
        line += (f"\n        first >10 uu divergence median tick {np.median([x for x in d10 if x is not None]) if any(x is not None for x in d10) else float('nan'):.0f}"
                 f" ({sum(x is None for x in d10)} never); >100 uu: {sum(x is not None for x in d100)}/{len(rs)} segments, median tick "
                 f"{np.median([x for x in d100 if x is not None]) if any(x is not None for x in d100) else float('nan'):.0f}")
        if kickoff:
            after_touch = sum(1 for r in rs if r["variants"][name]["div100"] is not None and r["rec_ball_touch"] is not None
                              and r["variants"][name]["div100"] >= r["rec_ball_touch"])
            line += f"; of those, at/after the first recorded ball touch: {after_touch}"
        print(line)
    for name in ("EVENT", "NAIVE"):
        fd = [r["first_diff"][name] for r in rs if r["first_diff"][name] is not None]
        from collections import Counter
        print(f"  {name} first control difference vs RAW: median tick {np.median([t for t, _ in fd]):.0f}; channel counts "
              f"{dict(Counter(c for _, cs in fd for c in cs))}")


def fmt(v, unit=""):
    return "-" if v is None else (f"{v:.0f}{unit}" if isinstance(v, float) else f"{v}{unit}")


def report(r, kickoff):
    print(f"\n{r['label']}: recorded dodge @{fmt(r['rec_dodge'])}t angle {fmt(r['rec_angle'],'deg')}"
          + (f", arrive @{fmt(r['rec_arrive'])}t speed {fmt(r['rec_speed_arrive'])}" if kickoff else f", speed@1s {fmt(r['rec_speed_1s'])}")
          + f" | cars within 200uu @{fmt(r['car_contact'])}t | missing frames {r['missing']}")
    for name, v in r["variants"].items():
        print(f"   {name:5s} dodge {'YES' if v['dodge_tick'] is not None else 'NO '} @{fmt(v['dodge_tick'])}t angle {fmt(v['dodge_angle'],'deg'):>6s}"
              + (f" arrive @{fmt(v['arrive'])}t speed {fmt(v['speed_at_arrive'])}" if kickoff else f" speed@1s {fmt(v['speed_1s'])}")
              + f" | diverges >10uu @{fmt(v['div10'])}t, >100uu @{fmt(v['div100'])}t")


if __name__ == "__main__":
    frames, phases = [], []
    for p in P:
        frames.append(p.match_info.frame_num); phases.append(str(p.match_info.match_phase).split(".")[-1])
    kick_starts = [i for i in range(1, len(P)) if phases[i] == "Kickoff" and phases[i - 1] != "Kickoff"]
    starts = [frames[i - 1] for i in kick_starts]  # last countdown packet: cars at spawn, frozen
    # in-play dodges: grounded >= 30 ticks before the first press, bot and ball far away during the segment
    seqs = data["seq"]
    U, ground, fr = data["U"], data["ground"], data["frames"]
    inplay = []
    for x in seqs:
        s0, e0 = x["period"]
        a = s0 + x["t1"]; b = s0 + x["t2"]
        if x["t1"] < 40 or b + 60 >= e0 or not all(ground[a - 30:a]):
            continue
        seg = range(a - 30, b + 60)
        pk = [P[i] for i in seg]
        dbot = min(np.linalg.norm(np.array([player(p, HP).physics.location.x - player(p, BP).physics.location.x,
                                             player(p, HP).physics.location.y - player(p, BP).physics.location.y])) for p in pk)
        dball = min(np.linalg.norm(np.array([player(p, HP).physics.location.x - p.balls[0].physics.location.x,
                                              player(p, HP).physics.location.y - p.balls[0].physics.location.y,
                                              player(p, HP).physics.location.z - p.balls[0].physics.location.z])) for p in pk)
        if dbot > 800 and dball > 400:
            inplay.append((fr[a - 30], b - a + 90))
    sel = inplay[:: max(1, len(inplay) // 6)][:6]
    snaps = snapshots(starts + [f for f, _ in sel])
    results = {"kickoffs": [], "inplay": []}
    print(f"===== KICKOFFS ({len(starts)}), 360 ticks (3 s) from the end of the countdown")
    for j, f in enumerate(starts):
        r = run_segment(f"kickoff {j + 1} (frame {f})", f, 360, snaps[f], kickoff=True)
        results["kickoffs"].append(r)
    summarise(results["kickoffs"], True)
    k = results["kickoffs"]
    print("  recorded: first ball touch (anyone) median tick", np.median([r["rec_ball_touch"] for r in k if r["rec_ball_touch"] is not None]),
          "| human arrival median", np.median([r["rec_arrive"] for r in k if r["rec_arrive"] is not None]),
          "| cars within 200 uu during the 3 s:", sum(r["car_contact"] is not None for r in k))
    print(f"\n===== IN-PLAY DODGES ({len(sel)} of {len(inplay)} clean candidates: grounded start, bot > 800 uu, ball > 400 uu)")
    for f, n in sel:
        r = run_segment(f"in-play dodge (frame {f}, t={f/120:.1f}s)", f, n, snaps[f])
        report(r, False); results["inplay"].append(r)
    summarise(results["inplay"], False)
    for grp in results.values():
        for r in grp:
            for v in r["variants"].values():
                v.pop("sim", None)
    pickle.dump(results, open(D / "il_sim.pkl", "wb"))
