"""Possession occasions in recorded live 1v1 matches (BotCore packet recordings, *.bcp).

    .\\.venv\\Scripts\\python.exe scripts\\possession_analysis.py runs\\beast\\ctrl-13300 [--first 2]

Definitions (fixed before the Beast recordings were inspected):
* Touch events come from each player's `latest_touch.game_seconds` changing.
* **Open play:** match phase Active, and at least KICKOFF_EXCLUDE_S after the kickoff went live.
* **BotCore wins the ball:** a BotCore touch whose previous touch was the opponent's, or after the ball had been
  untouched for at least LOOSE_S.
* **Room to act:** at that moment the opponent is at least ROOM_DIST uu from the ball, and its estimated time to the
  ball (distance / closing speed) is at least ROOM_ETA_S.
* **Pressure** is the opponent's distance, closing speed and time to the ball at BotCore's touch.
* **First touch** is the ball velocity FIRST_TOUCH_DT after the touch: speed, component towards the opponent goal,
  vertical speed, and whether a straight-line projection reaches the goal mouth within 3 s.
* **Next:** the next touch by anyone (who, and after how long), and whether BotCore's shot counter (Rocket League's
  own stat) rose or BotCore scored within SHOT_WINDOW_S.

Timestamps are given as the in-game clock (time remaining, as on the scoreboard) and as seconds elapsed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KICKOFF_EXCLUDE_S = 4.0
LOOSE_S = 2.0
ROOM_DIST = 1000.0
ROOM_ETA_S = 0.5
FIRST_TOUCH_DT = 1 / 30
SHOT_WINDOW_S = 5.0
GOAL_Y, GOAL_HALF_W = 5120.0, 892.755


def vec(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=float)


def clock(remaining: float) -> str:
    r = max(0, int(np.ceil(remaining)))
    return f"{r // 60}:{r % 60:02d}"


def analyse(bcp: Path) -> dict:
    from rlbot import flat

    from botcore.deploy.recording import load_recording

    rec = load_recording(bcp)
    team = int(rec.meta.get("team", 0))
    sign = 1.0 if team == 0 else -1.0  # direction of the opponent goal along y
    pk = [p for p in rec.packets if len(p.players) >= 2 and len(p.balls) >= 1]
    me = next(i for i, p in enumerate(pk[-1].players) if p.team == team)
    opp = 1 - me
    t = np.array([p.match_info.seconds_elapsed for p in pk])
    active = np.array([p.match_info.match_phase == flat.MatchPhase.Active for p in pk])

    # kickoff "live" times: first Active packet after a non-Active stretch that included a Kickoff phase
    live_at, saw_kickoff = [], False
    for i, p in enumerate(pk):
        ph = p.match_info.match_phase
        if ph == flat.MatchPhase.Kickoff:
            saw_kickoff = True
        elif ph == flat.MatchPhase.Active and saw_kickoff:
            live_at.append(t[i])
            saw_kickoff = False

    # touch events
    touches, last = [], {me: None, opp: None}
    for i, p in enumerate(pk):
        for who in (me, opp):
            lt = p.players[who].latest_touch
            if lt is None:
                continue
            gs = round(lt.game_seconds, 4)
            if last[who] is not None and gs != last[who]:
                touches.append((gs, who, i))
            last[who] = gs
    touches.sort()

    def idx_at(sec: float) -> int:
        return int(min(np.searchsorted(t, sec), len(pk) - 1))

    def shots_goals(i: int):
        s = pk[i].players[me].score_info
        return s.shots, s.goals

    occasions = []
    for k, (gs, who, i_seen) in enumerate(touches):
        if who != me:
            continue
        i = idx_at(gs)
        if not active[i]:
            continue
        lv = [x for x in live_at if x <= gs]
        if not lv or gs - lv[-1] < KICKOFF_EXCLUDE_S:
            continue
        prev = touches[k - 1] if k > 0 else None
        if prev is not None and prev[0] >= lv[-1]:
            won = prev[1] == opp or gs - prev[0] >= LOOSE_S
        else:
            won = gs - lv[-1] >= LOOSE_S
        if not won:
            continue
        p = pk[i]
        ball, car, o = p.balls[0].physics, p.players[me].physics, p.players[opp].physics
        bpos, opos, ovel = vec(ball.location), vec(o.location), vec(o.velocity)
        d_opp = float(np.linalg.norm(opos - bpos))
        closing = float(np.dot(ovel, (bpos - opos) / max(d_opp, 1.0)))
        eta = d_opp / closing if closing > 1.0 else float("inf")
        room = d_opp >= ROOM_DIST and eta >= ROOM_ETA_S
        # first touch: ball velocity just after
        j = idx_at(gs + FIRST_TOUCH_DT)
        b2 = pk[j].balls[0].physics
        bp2, bv2 = vec(b2.location), vec(b2.velocity)
        fwd = float(bv2[1] * sign)
        on_target = False
        if fwd > 0:
            tg = (sign * GOAL_Y - bp2[1]) / bv2[1]
            on_target = 0 < tg < 3.0 and abs(bp2[0] + bv2[0] * tg) < GOAL_HALF_W
        # next touch by anyone, and shot/goal within the window
        nxt = next((x for x in touches[k + 1:] if x[0] > gs), None)
        s0, g0 = shots_goals(i)
        jw = idx_at(gs + SHOT_WINDOW_S)
        window = range(i, jw + 1)
        s1 = max(shots_goals(m)[0] for m in window)
        g1 = max(shots_goals(m)[1] for m in window)
        opp_goal = max(pk[m].players[opp].score_info.goals for m in window) > pk[i].players[opp].score_info.goals
        our_follow = 0
        for x in touches[k + 1:]:
            if x[1] != me:
                break
            our_follow += 1
        # possession sequence: until the opponent's first touch after this occasion (or the phase leaves Active)
        end_i = next((m for m in range(i + 1, len(pk)) if not active[m]), len(pk) - 1)
        lost = next((x for x in touches[k + 1:] if x[1] == opp and x[0] > gs), None)
        if lost is not None and idx_at(lost[0]) <= end_i:
            li = idx_at(lost[0])
            lb = vec(pk[li].balls[0].physics.location)
            seq_end = {"how": "opponent touch", "after_s": round(lost[0] - gs, 2),
                       "botcore_dist_at_loss": round(float(np.linalg.norm(vec(pk[li].players[me].physics.location) - lb))),
                       "ball_y_toward_opp_goal_at_loss": round(float(lb[1] * sign))}
        else:
            li = end_i
            seq_end = {"how": "play stopped (goal or end)", "after_s": round(float(t[li] - gs), 2)}
        span = range(i, li + 1)
        seq_end["botcore_touches"] = 1 + sum(1 for x in touches[k + 1:] if x[1] == me and gs < x[0] <= t[li])
        seq_end["shot_before_loss"] = max(shots_goals(m)[0] for m in span) > s0
        seq_end["goal_before_loss"] = max(shots_goals(m)[1] for m in span) > g0
        seq_end["max_ball_y_toward_opp_goal"] = round(max(float(pk[m].balls[0].physics.location.y * sign) for m in span))
        n_info = None
        if nxt is not None:
            q = pk[idx_at(nxt[0])]
            nb = vec(q.balls[0].physics.location)
            n_info = {"who": "BotCore" if nxt[1] == me else "opponent", "after_s": round(nxt[0] - gs, 2),
                      "botcore_dist_to_ball_then": round(float(np.linalg.norm(vec(q.players[me].physics.location) - nb))),
                      "opp_dist_to_ball_then": round(float(np.linalg.norm(vec(q.players[opp].physics.location) - nb))),
                      "ball_y_progress_toward_opp_goal": round(float((nb[1] - bpos[1]) * sign))}
        occasions.append({
            "clock": clock(p.match_info.game_time_remaining), "seconds_elapsed": round(float(gs), 2),
            "overtime": bool(p.match_info.is_overtime),
            "ball_y_toward_opp_goal": round(float(bpos[1] * sign)), "ball_height": round(float(bpos[2])),
            "botcore_speed": round(float(np.linalg.norm(vec(car.velocity)))), "botcore_boost": int(p.players[me].boost),
            "botcore_on_ground": p.players[me].air_state == flat.AirState.OnGround,
            "won_from": "opponent touch" if prev is not None and prev[1] == opp and prev[0] >= lv[-1] else "loose ball",
            "pressure": {"opp_dist": round(d_opp), "opp_closing_speed": round(closing),
                         "opp_eta_s": None if eta == float("inf") else round(eta, 2),
                         "opp_goal_side": bool((opos[1] - bpos[1]) * sign > 0)},
            "room_to_act": room,
            "first_touch": {"ball_speed": round(float(np.linalg.norm(bv2))), "toward_opp_goal": round(fwd),
                            "vertical": round(float(bv2[2])), "projected_on_target": bool(on_target)},
            "botcore_consecutive_touches_after": our_follow,
            "sequence": seq_end,
            "next_touch": n_info,
            "shot_within_5s": s1 > s0, "goal_within_5s": g1 > g0, "conceded_within_5s": bool(opp_goal),
        })
    return {"recording": str(bcp), "botcore_team": team, "occasions": occasions}


def describe(o: dict) -> str:
    ft = o["first_touch"]
    if ft["projected_on_target"]:
        act = "shot on target (projected)"
    elif ft["vertical"] > 600 and ft["toward_opp_goal"] < 800:
        act = "popped the ball up"
    elif ft["toward_opp_goal"] >= 800:
        act = "hit it forward"
    elif ft["toward_opp_goal"] > -300:
        act = "soft/sideways touch"
    else:
        act = "hit it backwards"
    return f"{act} ({ft['ball_speed']} uu/s, {ft['toward_opp_goal']:+d} toward goal)"


def outcome(o: dict) -> str:
    n = o["next_touch"]
    if o["goal_within_5s"]:
        return "goal"
    if o["shot_within_5s"]:
        return "shot"
    if n is None:
        return "no further touch"
    if n["who"] == "BotCore":
        return "kept possession, no shot"
    if n["after_s"] < 1.0:
        return "opponent challenged within 1 s"
    if n["botcore_dist_to_ball_then"] > 1000:
        return "opponent collected; BotCore >1000 uu away"
    return "opponent collected; BotCore close"


def sequence_end(o: dict) -> str:
    s = o["sequence"]
    if s["goal_before_loss"]:
        return "ended in a BotCore goal"
    if s["shot_before_loss"]:
        return "ended in a BotCore shot"
    if s["how"] != "opponent touch":
        return "play stopped, no shot"
    if s["botcore_dist_at_loss"] <= 500:
        return "lost in a contested touch (BotCore within 500 uu)"
    return "opponent collected cleanly (BotCore >500 uu away)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir")
    ap.add_argument("--first", type=int, default=2)
    args = ap.parse_args()
    sd = Path(args.series_dir)
    out = {"definitions": {"KICKOFF_EXCLUDE_S": KICKOFF_EXCLUDE_S, "LOOSE_S": LOOSE_S, "ROOM_DIST": ROOM_DIST,
                           "ROOM_ETA_S": ROOM_ETA_S, "SHOT_WINDOW_S": SHOT_WINDOW_S}, "matches": {}}
    for d in sorted(x for x in sd.glob("m*-*") if x.is_dir()):
        bcp = sorted(d.glob("botcore-team*-index*.bcp"))
        if not bcp:
            continue
        a = analyse(bcp[0])
        room = [o for o in a["occasions"] if o["room_to_act"]]
        for o in a["occasions"]:
            o["first_touch_label"], o["outcome"], o["sequence_end"] = describe(o), outcome(o), sequence_end(o)
        a["n_won"], a["n_room"] = len(a["occasions"]), len(room)
        a["first"] = room[:args.first]
        out["matches"][d.name] = a
        print(f"\n== {d.name}: {len(a['occasions'])} open-play wins, {len(room)} with room to act")
        for o in room[:args.first]:
            pr, n = o["pressure"], o["next_touch"]
            print(f"  {o['clock']} ({o['seconds_elapsed']}s) from {o['won_from']}; opp {pr['opp_dist']} uu, "
                  f"closing {pr['opp_closing_speed']}, eta {pr['opp_eta_s']}, goal-side {pr['opp_goal_side']}; "
                  f"ball y {o['ball_y_toward_opp_goal']}; {o['first_touch_label']}; "
                  f"next {n and (n['who'], n['after_s'])}; -> {o['outcome']} | seq: {o['sequence_end']} "
                  f"after {o['sequence']['after_s']}s, {o['sequence']['botcore_touches']} touches, "
                  f"max y {o['sequence']['max_ball_y_toward_opp_goal']}")
    allroom = [o for m in out["matches"].values() for o in m["occasions"] if o["room_to_act"]]
    tally = {}
    for o in allroom:
        tally[o["outcome"]] = tally.get(o["outcome"], 0) + 1
    out["all_room_occasions_outcomes"] = tally
    out["all_room_first_touch"] = {}
    for o in allroom:
        lab = o["first_touch_label"].split(" (")[0]
        out["all_room_first_touch"][lab] = out["all_room_first_touch"].get(lab, 0) + 1
    out["all_room_sequence_end"] = {}
    for o in allroom:
        out["all_room_sequence_end"][o["sequence_end"]] = out["all_room_sequence_end"].get(o["sequence_end"], 0) + 1
    print("\nall occasions with room:", len(allroom), tally, out["all_room_first_touch"], out["all_room_sequence_end"])
    (sd / "possession.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
