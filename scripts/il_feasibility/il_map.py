"""IL feasibility (no training): human-perspective observations + 90-row/8-tick action mapping + loss metrics.

python il_map.py <outdir>     -> writes <outdir>/il_data.pkl and prints the report
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(r"C:\Users\TomPr\RLBotProject")
sys.path.insert(0, str(R / "src"))
from rlbot import flat  # noqa: E402

from botcore.actions.table import get_action_table  # noqa: E402
from botcore.deploy.live_report import feature_names  # noqa: E402
from botcore.deploy.packets import CompatObsPipeline  # noqa: E402
from botcore.deploy.recording import load_recording  # noqa: E402

T = get_action_table().controls  # throttle, steer, pitch, yaw, roll, jump, boost, handbrake
CH = ["throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"]
W_GROUND = np.array([1, 1, 0, 0, 0, 3, 1, 0.5])
W_AIR = np.array([0.1, 0, 1, 1, 1, 3, 1, 0])
NAMES = feature_names()


def ctl(c) -> np.ndarray:
    return np.array([c.throttle, c.steer, c.pitch, c.yaw, c.roll, float(c.jump), float(c.boost), float(c.handbrake)],
                    dtype=np.float32)


def nearest_row(target: np.ndarray, on_ground, dodge=None) -> int:
    """on_ground: bool, or the fraction of the block's ticks spent on the ground (mixed blocks use both weight sets).
    dodge: None keeps the original rule (any airborne jump block); True/False states explicitly whether this block
    starts a dodge, the only case where the game's yaw+roll sum sets the direction and yaw may be moved onto roll."""
    t = target.copy()
    gf = float(on_ground)
    if (dodge if dodge is not None else (gf < 1 and t[5] == 1)):  # dodge: the game uses yaw+roll for the side component; table jump rows have yaw=0
        t[4] = np.clip(t[3] + t[4], -1, 1)
        t[3] = 0
    w = (W_GROUND if gf == 1 else W_AIR if gf == 0 else np.maximum(W_GROUND, W_AIR)) + 0.01
    return int(np.argmin((np.abs(T - t) * w).sum(axis=1)))


def press_segments(jump: np.ndarray):
    segs, k = [], 0
    while k < len(jump):
        if jump[k]:
            s = k
            while k < len(jump) and jump[k]:
                k += 1
            segs.append((s, k))  # [s, k)
        else:
            k += 1
    return segs


def map_period(U: np.ndarray, ground: np.ndarray, mode: str) -> np.ndarray:
    """Map one kickoff-to-goal period of per-tick human controls to one table row per 8-tick block."""
    n_blocks = int(np.ceil(len(U) / 8))
    rows = np.zeros(n_blocks, dtype=int)
    if mode == "naive":  # sample-and-hold at the block start
        for b in range(n_blocks):
            rows[b] = nearest_row(U[b * 8], float(np.mean(ground[b * 8:(b + 1) * 8])))
        return rows
    # event-aligned: majority for every channel, jump presses re-timed onto blocks preserving press/release order
    jump_blocks = np.zeros(n_blocks, dtype=bool)
    dir_at = {}
    prev_end = -2
    for s, e in press_segments(U[:, 5] > 0.5):
        start = max(int(round(s / 8)), prev_end + 2 if prev_end >= 0 else 0)  # >= 1 released block in between
        length = max(1, int(round((e - s) / 8)))
        if start >= n_blocks:
            break
        jump_blocks[start:start + length] = True
        dir_at[start] = s
        prev_end = start + length - 1
    for b in range(n_blocks):
        blk = U[b * 8:(b + 1) * 8]
        tgt = np.array([Counter(np.round(blk[:, j], 3)).most_common(1)[0][0] for j in range(8)], dtype=np.float32)
        tgt[5] = float(jump_blocks[b])
        g = float(np.mean(ground[b * 8:(b + 1) * 8]))
        if b in dir_at:  # a press starts here: use the stick at the press tick (sets the dodge direction)
            tgt[2:5] = U[dir_at[b], 2:5]
            if not ground[dir_at[b]]:
                g = 0.0
        rows[b] = nearest_row(tgt, g)
    return rows


def main(out: Path):
    rec = load_recording(R / "runs/live/human-argmax-11900/botcore-team1-index0.bcp")
    pipe = CompatObsPipeline(rec.field)
    P = rec.packets
    bot_dec = {f: (a, o) for f, a, o in zip(rec.decision_frame, rec.decision_action, rec.decision_obs)}
    # ---------------- pass 1: per-packet human data and bot-obs reconstruction check
    frames, phases, U, ground, hum_pid, bot_pid = [], [], [], [], None, None
    dodge_dirs = {}
    bot_check = []
    prev_bot = None
    prev_dodged = False
    for p in P:
        if len(p.players) < 2 or not p.balls:
            continue
        h = next(x for x in p.players if x.team == 0)
        b = next(x for x in p.players if x.team == 1)
        hum_pid, bot_pid = h.player_id, b.player_id
        pipe.update(p)
        f = p.match_info.frame_num
        ph = str(p.match_info.match_phase).split(".")[-1]
        frames.append(f); phases.append(ph); U.append(ctl(h.last_input))
        ground.append(str(h.air_state).endswith("OnGround"))
        if h.has_dodged and not prev_dodged:
            dodge_dirs[f] = (h.dodge_dir.x, h.dodge_dir.y)
        prev_dodged = h.has_dodged
        if f in bot_dec:
            a, o = bot_dec[f]
            prev = o[NAMES.index("self.prev_action[0]"):NAMES.index("self.prev_action[0]") + 8]
            mine = pipe.obs(bot_pid, prev_action=prev)
            bot_check.append(float(np.abs(mine - o).max()))
    frames, U, ground = np.array(frames), np.array(U), np.array(ground)
    bc = np.array(bot_check)
    print(f"[obs pipeline check] rebuilt the bot's own observation at {len(bc)} decisions: max abs diff median "
          f"{np.median(bc):.2e}, 99th pct {np.percentile(bc, 99):.2e}, exact (<1e-5) {np.mean(bc < 1e-5):.1%}")
    # ---------------- periods: kickoff start -> last active packet before the next countdown
    periods = []
    k = 0
    while k < len(phases):
        if phases[k] == "Kickoff" and (k == 0 or phases[k - 1] != "Kickoff"):
            e = k
            while e < len(phases) and phases[e] in ("Kickoff", "Active"):
                e += 1
            periods.append((k, e))
            k = e
        else:
            k += 1
    # ---------------- mapping + tick-level loss
    res = {}
    for mode in ("naive", "event"):
        agree = {c: [] for c in CH}
        rows_all = []
        for s, e in periods:
            u, g = U[s:e], ground[s:e]
            rows = map_period(u, g, mode)
            rows_all.append(rows)
            m = T[np.repeat(rows, 8)[:len(u)]]
            for j, c in enumerate(CH):
                rel = (g & (W_GROUND[j] > 0.2)) | (~g & (W_AIR[j] > 0.2))
                if j in (3, 4):  # yaw/roll in the air: compare the yaw+roll sum used for dodges only on press ticks
                    pass
                agree[c].append((m[rel, j] == u[rel, j]))
        res[mode] = {c: float(np.mean(np.concatenate(v))) for c, v in agree.items()}
        res[mode + "_rows"] = rows_all
    print("\n[tick-level agreement on state-relevant channels] (ground: throttle/steer/jump/boost/handbrake; air: pitch/yaw/roll/jump/boost)")
    for c in CH:
        print(f"  {c:9s} naive {res['naive'][c]:.1%}   event-aligned {res['event'][c]:.1%}")
    # ---------------- jump/dodge sequences
    seq = []
    for s, e in periods:
        u, g = U[s:e], ground[s:e]
        segs = press_segments(u[:, 5] > 0.5)
        for i in range(1, len(segs)):
            (a0, a1), (b0, b1) = segs[i - 1], segs[i]
            if g[max(a0 - 1, 0)] and not g[min(b0, len(g) - 1)] and (s + b0 + 1) < len(frames) and \
                    any(frames[s + b0] <= f <= frames[s + b0] + 2 for f in dodge_dirs):
                seq.append({"first_press": a1 - a0, "release": b0 - a1, "jump_to_dodge": b0 - a0, "period": (s, e), "t1": a0, "t2": b0})
    jd = np.array([x["jump_to_dodge"] for x in seq]); fp = np.array([x["first_press"] for x in seq]); rl = np.array([x["release"] for x in seq])
    print(f"\n[human ground jump -> dodge sequences] n={len(seq)}: first press {np.median(fp):.0f} ticks (min {fp.min()}, max {fp.max()}), "
          f"release {np.median(rl):.0f} (min {rl.min()}), jump->dodge {np.median(jd):.0f} ticks (min {jd.min()}, 10th pct {np.percentile(jd,10):.0f}, max {jd.max()})")
    print(f"  share with jump->dodge < 16 ticks (the minimum the 8-tick interface allows: press, release, press): {np.mean(jd < 16):.0%}")
    # mapped re-timing (event mode) -- symbolic: when does the mapped dodge press happen?
    delays, lost_naive = [], 0
    for x in seq:
        (s, e) = x["period"]
        for mode in ("event", "naive"):
            rows = map_period(U[s:e], ground[s:e], mode)
            jb = T[rows, 5] > 0.5
            b1 = None
            presses = [b for b in range(len(jb)) if jb[b] and (b == 0 or not jb[b - 1])]
            # mapped press pair around this sequence
            first = [b for b in presses if abs(b * 8 - x["t1"]) <= 8]
            if not first:
                if mode == "naive": lost_naive += 1
                continue
            second = [b for b in presses if b > first[0] and b * 8 <= x["t2"] + 24]
            if mode == "event" and second:
                delays.append(second[0] * 8 - x["t2"])
            if mode == "naive" and not second:
                lost_naive += 1
    dl = np.array(delays)
    print(f"  event-aligned mapping keeps {len(dl)}/{len(seq)} press-release-press patterns; dodge press shifted by "
          f"median {np.median(dl):+.0f} ticks (range {dl.min():+d}..{dl.max():+d}); naive sample-and-hold loses {lost_naive}/{len(seq)}")
    # ---------------- dodge direction representability
    ang = []
    for f, (x, y) in dodge_dirs.items():
        a = np.degrees(np.arctan2(y, x))
        cand = np.degrees(np.arctan2([0, 1, 1, 1, 0, -1, -1, -1], [1, 1, 0, -1, -1, -1, 0, 1]))
        ang.append(np.min(np.abs((a - cand + 180) % 360 - 180)))
    ang = np.array(ang)
    print(f"\n[dodge direction] {len(ang)} human dodges: exactly representable (8 table directions) {np.mean(ang < 1):.0%}; "
          f"others off by {np.unique(np.round(ang[ang >= 1]))} deg")
    # ---------------- human-perspective observations at event-mode decisions
    pipe2 = CompatObsPipeline(rec.field)
    idx_of = {f: i for i, f in enumerate(frames)}
    dec = {}
    for (s, e), rows in zip(periods, res["event_rows"]):
        for b, r in enumerate(rows):
            if s + b * 8 < e:
                dec[frames[s + b * 8]] = (r, rows[b - 1] if b > 0 else None)
    obs, labels, checks = [], [], []
    for p in P:
        if len(p.players) < 2 or not p.balls:
            continue
        pipe2.update(p)
        f = p.match_info.frame_num
        if f in dec:
            r, prev = dec[f]
            o = pipe2.obs(hum_pid, prev_action=None if prev is None else T[prev])
            h = next(x for x in p.players if x.team == 0)
            if h.demolished_timeout > 0 or o[NAMES.index("self.is_demoed[0]")] > 0:
                continue
            checks.append(abs(o[NAMES.index("self.pos[0]")] * 4096 - h.physics.location.x) +
                          abs(o[NAMES.index("self.pos[1]")] * 4096 - h.physics.location.y))
            obs.append(o); labels.append(r)
    obs = np.array(obs)
    print(f"\n[human-perspective observations] {len(obs)} decisions (8-tick, event-aligned labels); finite {np.isfinite(obs).all()}, "
          f"|obs| max {np.abs(obs).max():.2f}; self position vs packet (not demolished) max error {max(checks):.3f} uu (blue = identity frame); "
          f"opponent slot present {np.mean(obs[:, NAMES.index('car2.present[0]')] > 0):.0%}")
    lc = Counter(labels)
    print(f"  labels: {len(lc)} distinct rows used; top-5 share {sum(v for _, v in lc.most_common(5)) / len(labels):.0%}; "
          f"dodge-row (air jump with direction) decisions {sum(1 for i, l in enumerate(labels) if T[l, 5] and (T[l, 2] or T[l, 4]))}")
    pickle.dump({"frames": frames, "U": U, "ground": ground, "periods": periods, "seq": seq, "dodge_dirs": dodge_dirs,
                 "rows_event": res["event_rows"], "rows_naive": res["naive_rows"], "hum_pid": hum_pid, "bot_pid": bot_pid},
                open(out / "il_data.pkl", "wb"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
