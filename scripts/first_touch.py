"""First touches under pressure in simulated matches (docs/EXPERIMENT_HL_IMITATION.md, "Behaviour").

    .\\.venv\\Scripts\\python.exe scripts\\first_touch.py run      # play and record the matches
    .\\.venv\\Scripts\\python.exe scripts\\first_touch.py report   # tables, examples and HTML clips

Matches: each subject (HL candidate, CTRL candidate, and 13300 itself as the baseline) plays frozen 13300 on the same
colour-swapped fresh seeds (offsets 21,700+), 5-minute matches with evaluate.py's rules (kickoff after every goal; no
overtime here). Our policies sample. State is logged every decision step (8 ticks).

Definitions (fixed before any of these matches is played):
* **Subject wins the ball:** a subject touch in open play (>= 4 s after the kickoff reset) whose previous touch was
  the opponent's, or after the ball was untouched for >= 2 s.
* **Under pressure:** at that touch the opponent is within 2,500 uu of the ball.
* **The defender closes / backs off:** the opponent's mean velocity component towards the ball over the 0.5 s before
  the touch. >= +500 uu/s is "closes", <= 0 is "backs off", and anything in between is "holds".
* **First-touch outcome:**
  * ball speed and its component towards the opponent goal, one step (8 ticks) after the touch;
  * the next touch by anyone within 3 s: subject (kept), opponent (lost) or none;
  * a goal for within 5 s;
  * a goal against within 10 s;
  * subject state at the touch: speed, boost, grounded, and a dodge started in the previous 0.5 s.
* **Examples:** fixed rule, no picking. In the first two seed pairs, for each subject, the first "closes" and the
  first "backs off" event in each match. Each gets an HTML clip from 3 s before to 4 s after.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "runs" / "hl-eval" / "first-touch"
OFFSET, PAIRS = 21_700, 20
STEP_S = 8 / 120
GOAL_Y = 5120.0


def subjects() -> dict:
    sc = json.loads((ROOT / "runs" / "hl-eval" / "screen.json").read_text(encoding="utf-8"))
    return {"HL": ROOT / "runs" / "s6-hl" / "exports" / f"iter_{sc['HL']['candidate']:06d}",
            "CTRL": ROOT / "runs" / "s6-ctrl" / "exports" / f"iter_{sc['CTRL']['candidate']:06d}",
            "13300": ROOT / "runs" / "reference" / "S5C-iter13300"}


def play(args) -> dict:
    subj_path, opp_path, seed, subj_blue = args
    import torch

    from botcore import constants as C
    from botcore.env import EnvSpec, build_env
    from botcore.evaluate import EVAL_SEED_BASE
    from botcore.export import PolicyRunner, load_bundle

    torch.set_num_threads(1)
    s = EVAL_SEED_BASE + seed
    st_team = C.BLUE_TEAM if subj_blue else C.ORANGE_TEAM
    run = {C.BLUE_TEAM: PolicyRunner(load_bundle(subj_path if subj_blue else opp_path, device="cpu"), seed=2 * s),
           C.ORANGE_TEAM: PolicyRunner(load_bundle(opp_path if subj_blue else subj_path, device="cpu"), seed=2 * s + 1)}
    spec = EnvSpec(team_size=1, action_repeat=8, rlbot_delay=True, no_touch_timeout_seconds=1e9,
                   episode_timeout_seconds=1e9, scenario_weights={"kickoff": 1.0}, reward_weights={})
    env = build_env(spec, s)
    obs = env.reset()
    log, score, since_reset = [], [0, 0], 0
    for step in range(int(300 / STEP_S)):
        acts = {a: np.array([run[env.state.cars[a].team_num].act(obs[a])]) for a in env.agents}
        obs, _, term, _ = env.step(acts)
        st = env.state
        since_reset += 1
        row = {"t": round((step + 1) * STEP_S, 3), "since_reset": since_reset, "ball": np.round(st.ball.position, 1).tolist(),
               "ball_v": np.round(st.ball.linear_velocity, 1).tolist(), "score": list(score), "goal": None}
        for car in st.cars.values():
            k = "subj" if car.team_num == st_team else "opp"
            row[k] = {"pos": np.round(car.physics.position, 1).tolist(), "vel": np.round(car.physics.linear_velocity, 1).tolist(),
                      "yaw": round(float(car.physics.yaw), 3), "boost": round(float(car.boost_amount), 1),
                      "ground": bool(car.on_ground), "flipped": bool(car.has_flipped), "touch": bool(car.ball_touches > 0),
                      "team": int(car.team_num)}
        if any(term.values()):
            scorer = int(st.scoring_team)
            score[scorer] += 1
            row["goal"] = "for" if scorer == st_team else "against"
            row["score"] = list(score)
            obs = env.reset()
            since_reset = 0
        log.append(row)
    env.close()
    return {"seed": seed, "subj_blue": subj_blue, "subj_team": int(st_team), "log": log}


def events(m: dict) -> list:
    L = m["log"]
    sign = 1.0 if m["subj_team"] == 0 else -1.0  # opponent goal direction along y
    out, last_touch, last_touch_i = [], None, -10 ** 9
    for i, r in enumerate(L):
        if r["since_reset"] <= 1:
            last_touch, last_touch_i = None, i
        s, o = r["subj"], r["opp"]
        if s["touch"] and o["touch"]:
            who = "both"
        elif s["touch"]:
            who = "subj"
        elif o["touch"]:
            who = "opp"
        else:
            continue
        open_play = r["since_reset"] * STEP_S >= 4.0
        wins = who == "subj" and (last_touch == "opp" or (i - last_touch_i) * STEP_S >= 2.0 or last_touch is None)
        prev_owner = last_touch
        last_touch, last_touch_i = who, i
        if not (open_play and wins):
            continue
        ball = np.array(r["ball"])
        opos = np.array(o["pos"])
        dist = float(np.linalg.norm(opos - ball))
        if dist > 2500:
            continue
        w = L[max(0, i - 7):i + 1]  # ~0.5 s
        close = float(np.mean([np.dot(np.array(x["opp"]["vel"]), (np.array(x["ball"]) - np.array(x["opp"]["pos"])) /
                                      (np.linalg.norm(np.array(x["ball"]) - np.array(x["opp"]["pos"])) + 1e-6)) for x in w]))
        kind = "closes" if close >= 500 else "backs off" if close <= 0 else "holds"
        nxt = L[i + 1] if i + 1 < len(L) else r
        bv = np.array(nxt["ball_v"])
        nt = None
        for j in range(i + 2, min(len(L), i + 1 + int(3 / STEP_S))):
            if L[j]["since_reset"] <= 1:
                break
            if L[j]["subj"]["touch"] or L[j]["opp"]["touch"]:
                nt = ("kept" if L[j]["subj"]["touch"] and not L[j]["opp"]["touch"] else "lost" if L[j]["opp"]["touch"] and not L[j]["subj"]["touch"] else "50/50", round((j - i) * STEP_S, 2))
                break
        gf = any(x["goal"] == "for" for x in L[i:i + int(5 / STEP_S)])
        ga = any(x["goal"] == "against" for x in L[i:i + int(10 / STEP_S)])
        dodge = any(L[j]["subj"]["flipped"] and not L[j - 1]["subj"]["flipped"] for j in range(max(1, i - 7), i + 1))
        out.append({"i": i, "t": r["t"], "kind": kind, "opp_closing_uu_s": round(close), "opp_dist": round(dist),
                    "opp_goal_side": bool(sign * (opos[1] - ball[1]) > 0), "prev_touch": prev_owner,
                    "subj_speed": round(float(np.linalg.norm(s["vel"]))), "subj_boost": s["boost"], "subj_ground": s["ground"],
                    "dodge_0_5s": dodge, "ball_height": round(float(ball[2])),
                    "ball_speed_after": round(float(np.linalg.norm(bv))), "ball_to_goal_after": round(float(sign * bv[1])),
                    "next_touch": nt[0] if nt else "none", "next_touch_s": nt[1] if nt else None,
                    "goal_for_5s": gf, "goal_against_10s": ga})
    return out


def run() -> None:
    subs = subjects()
    opp = subs["13300"]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, p in subs.items():
        for k in range(PAIRS):
            for blue in (True, False):
                f = OUT / "matches" / f"{name}-{OFFSET + k}-{'blue' if blue else 'orange'}.json"
                if not f.is_file():
                    jobs.append(((str(p), str(opp), OFFSET + k, blue), f))
    (OUT / "matches").mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} matches to play", flush=True)
    with ProcessPoolExecutor(11) as ex:
        for (_, f), r in zip(jobs, ex.map(play, [a for a, _ in jobs])):
            f.write_text(json.dumps(r), encoding="utf-8")


def clip(m: dict, ev: dict, title: str, path: Path) -> None:
    from botcore.viz import _car_frame, write_html

    L = m["log"]
    frames = []
    for r in L[max(0, ev["i"] - 45):ev["i"] + 61]:
        cars = [_car_frame(r[k]["pos"], r[k]["yaw"], r[k]["team"], r[k]["boost"], r[k]["ground"]) for k in ("subj", "opp")]
        frames.append([round(r["t"], 2), [round(v) for v in r["ball"]], cars, r["score"]])
    write_html({"title": title, "frames": frames}, path)


def report() -> dict:
    subs = list(subjects())
    res, examples = {}, []
    for name in subs:
        ev_all = []
        for k in range(PAIRS):
            for blue in (True, False):
                f = OUT / "matches" / f"{name}-{OFFSET + k}-{'blue' if blue else 'orange'}.json"
                m = json.loads(f.read_text(encoding="utf-8"))
                ev = events(m)
                ev_all += ev
                if k < 2:
                    for kind in ("closes", "backs off"):
                        e = next((x for x in ev if x["kind"] == kind), None)
                        if e:
                            side = "blue" if blue else "orange"
                            html = OUT / "clips" / f"{name}-{OFFSET + k}-{side}-{kind.replace(' ', '')}.html"
                            clip(m, e, f"{name} ({side}) vs 13300, seed offset {OFFSET + k}, t={e['t']:.1f}s: defender {kind}", html)
                            examples.append({"subject": name, "seed": OFFSET + k, "side": side,
                                             "clip": str(html.relative_to(ROOT)), **{x: y for x, y in e.items() if x != "i"}})
        tab = {}
        for kind in ("closes", "holds", "backs off"):
            es = [e for e in ev_all if e["kind"] == kind]
            if not es:
                continue
            n = len(es)
            tab[kind] = {"events": n, "per_match": round(n / (2 * PAIRS), 2),
                         "kept": round(sum(e["next_touch"] == "kept" for e in es) / n, 3),
                         "lost": round(sum(e["next_touch"] == "lost" for e in es) / n, 3),
                         "fifty_fifty": round(sum(e["next_touch"] == "50/50" for e in es) / n, 3),
                         "no_touch_3s": round(sum(e["next_touch"] == "none" for e in es) / n, 3),
                         "goal_for_5s": round(sum(e["goal_for_5s"] for e in es) / n, 3),
                         "goal_against_10s": round(sum(e["goal_against_10s"] for e in es) / n, 3),
                         "median_ball_speed_after": float(np.median([e["ball_speed_after"] for e in es])),
                         "median_ball_to_goal_after": float(np.median([e["ball_to_goal_after"] for e in es])),
                         "dodge_touch_share": round(sum(e["dodge_0_5s"] for e in es) / n, 3),
                         "airborne_touch_share": round(sum(not e["subj_ground"] for e in es) / n, 3),
                         "median_subject_speed": float(np.median([e["subj_speed"] for e in es])),
                         "zero_boost_share": round(sum(e["subj_boost"] < 1 for e in es) / n, 3)}
        res[name] = tab
    out = {"tables": res, "examples": examples}
    (OUT / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
