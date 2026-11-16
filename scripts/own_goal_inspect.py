"""Inspect own-goal sequences of a policy: car state at its last touch and reward contributions over the 3 s before.

    .\\.venv\\Scripts\\python.exe scripts\\own_goal_inspect.py <policy bundle> <opponent bundle> [n_seeds] [seed_offset] [out_dir]

For every own goal (the policy's team touched the ball last, then conceded; window = last 10 s) it records, in the policy's team frame:
* at the last touch: car position relative to the ball (goal side or attacking side of the ball), height, airborne,
  upside down (car up-vector z < 0), mid-flip, speed, facing (cos between car forward and the direction to its
  own goal), ball speed towards its own goal right after the touch;
* the policy's reward contributions (training weights) summed over the window and at the touch decision.

Heuristic categories (each sequence gets the first that applies):
* control/recovery: at the last touch the car is upside down, mid-flip, or airborne with nose pointing down;
* positioning: car on the attacking side of the ball (ball between car and own goal) while facing its own goal;
* other: anything else (e.g. deflections while defending from the goal side).
Reward exploitation is assessed separately: did the window's shaping reward (everything except the goal terms)
come out positive despite the own goal, and how much of it came from the touch itself?
Writes JSON with every sequence plus HTML replays of a few representative ones.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from botcore import constants as C  # noqa: E402
from botcore.config import EVAL_SEED_BASE  # noqa: E402
from botcore.env import EnvSpec, build_env  # noqa: E402
from botcore.export import PolicyRunner, load_bundle  # noqa: E402
from botcore.viz import write_html  # noqa: E402

WINDOW = 150  # decisions = 10 s (slow rollers can take several seconds after the last touch)
TRAIN_WEIGHTS = {"goal_scored": 3.0, "goal_conceded": 3.0, "ball_touch": 0.5, "velocity_to_ball": 0.5,
                 "ball_to_goal": 1.5, "boost_economy": 0.1}  # long-run / cell D weights


def team_frame(v, team):
    v = np.asarray(v, dtype=np.float64)
    return v * np.array([-1.0, -1.0, 1.0]) if team == C.ORANGE_TEAM else v


def main() -> None:
    policy, opponent = Path(sys.argv[1]), Path(sys.argv[2])
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    offset = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    out_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else ROOT / "runs" / "own_goals"
    out_dir.mkdir(parents=True, exist_ok=True)
    pb, ob = load_bundle(policy), load_bundle(opponent)
    sequences, goals_scored, matches = [], 0, 0
    for s in range(n_seeds):
        seed = EVAL_SEED_BASE + offset + s
        for p_team in (C.BLUE_TEAM, C.ORANGE_TEAM):
            matches += 1
            runners = {p_team: PolicyRunner(pb, seed=seed * 2), 1 - p_team: PolicyRunner(ob, seed=seed * 2 + 1)}
            spec = EnvSpec(team_size=1, action_repeat=8, rlbot_delay=True, no_touch_timeout_seconds=1e9,
                           episode_timeout_seconds=1e9, scenario_weights={"kickoff": 1.0},
                           reward_weights=TRAIN_WEIGHTS)
            env = build_env(spec, seed)
            obs = env.reset()
            me = next(a for a in env.agents if env.state.cars[a].team_num == p_team)
            window = deque(maxlen=WINDOW)
            last_toucher = None
            t = 0
            for _ in range(int(300 * 120 / 8)):
                st = env.state
                actions = {a: np.array([runners[st.cars[a].team_num].act(obs[a])]) for a in env.agents}
                obs, rewards, term, _ = env.step(actions)
                t += 1
                st = env.state
                car = st.cars[me]
                idx = env.agents.index(me)
                breakdown = {k: float(v[idx]) for k, v in env.reward_fn.last_breakdown.items()}
                touched = car.ball_touches > 0
                for a, c in st.cars.items():
                    if c.ball_touches > 0:
                        last_toucher = c.team_num
                rot = np.asarray(car.physics.rotation_mtx)
                window.append({
                    "t": round(t * 8 / 120, 2), "action": int(actions[me][0]), "touch": bool(touched),
                    "car_pos": team_frame(car.physics.position, p_team).round(0).tolist(),
                    "car_vel": team_frame(car.physics.linear_velocity, p_team).round(0).tolist(),
                    "car_forward": team_frame(rot[:, 0], p_team).round(3).tolist(),
                    "car_up_z": round(float(rot[2, 2]), 3), "on_ground": bool(car.on_ground),
                    "flipping": bool(car.is_flipping), "has_flip": bool(car.has_flip),
                    "ball_pos": team_frame(st.ball.position, p_team).round(0).tolist(),
                    "ball_vel": team_frame(st.ball.linear_velocity, p_team).round(0).tolist(),
                    "reward": breakdown,
                    "frame": [round(t * 8 / 120, 2), [round(float(v)) for v in st.ball.position],
                              [[round(float(x)) for x in c.physics.position] + [round(float(c.physics.yaw), 3),
                                int(c.team_num), int(round(c.boost_amount)), int(bool(c.on_ground))]
                               for c in st.cars.values()], [0, 0]],
                })
                if any(term.values()):
                    scorer = st.scoring_team
                    if scorer == p_team:
                        goals_scored += 1
                    elif last_toucher == p_team:
                        sequences.append(analyse(list(window), seed, p_team))
                    window.clear()
                    last_toucher = None
                    obs = env.reset()
            env.close()
    categories = Counter(s["category"] for s in sequences)
    exploit = [s for s in sequences if s["window_shaping_reward"] > 0]
    summary = {
        "policy": str(policy), "opponent": str(opponent), "matches": matches, "goals_scored": goals_scored,
        "own_goals": len(sequences), "categories": dict(categories),
        "own_goals_with_positive_window_shaping": len(exploit),
        "own_goals_without_touch_in_window": sum(1 for s in sequences if s["last_touch"] is None),
        "mean_shaping_from_last_touch_to_goal": round(float(np.mean([s["shaping_from_last_touch_to_goal"] for s in sequences if s["last_touch"]])), 3) if any(s["last_touch"] for s in sequences) else None,
        "mean_window_shaping_reward": round(float(np.mean([s["window_shaping_reward"] for s in sequences])), 3) if sequences else None,
        "mean_touch_reward_at_last_touch": round(float(np.mean([s["last_touch_reward"]["ball_touch"] for s in sequences if s["last_touch_reward"]])), 3) if sequences else None,
        "mean_last_touch": {k: round(float(np.mean([s["last_touch"][k] for s in sequences if s["last_touch"]])), 3)
                            for k in ("height", "speed", "facing_own_goal_cos", "ball_speed_to_own_goal_after",
                                      "dist_to_own_goal")} if sequences else None,
        "last_touch_flags": {k: int(sum(bool(s["last_touch"] and s["last_touch"][k]) for s in sequences))
                             for k in ("airborne", "upside_down", "flipping", "attacking_side_of_ball")} if sequences else None,
    }
    (out_dir / "own_goals.json").write_text(json.dumps({"summary": summary, "sequences": [
        {k: v for k, v in s.items() if k != "frames"} for s in sequences]}, indent=2), encoding="utf-8")
    # replays of up to two sequences per category
    shown = Counter()
    for i, s in enumerate(sequences):
        if shown[s["category"]] >= 2:
            continue
        shown[s["category"]] += 1
        write_html({"title": f"Own goal {i}: {s['category']} (seed {s['seed']}, policy {'blue' if s['team'] == 0 else 'orange'})",
                    "frames": s["frames"]}, out_dir / f"own_goal_{i}_{s['category']}.html")
    print(json.dumps(summary, indent=2))


def analyse(window: list, seed: int, team: int) -> dict:
    touch_idx = [i for i, w in enumerate(window) if w["touch"]]
    last = None
    last_reward = None
    category = "other"
    if touch_idx:
        w = window[touch_idx[-1]]
        nxt = window[min(touch_idx[-1] + 1, len(window) - 1)]
        car, ball = np.array(w["car_pos"]), np.array(w["ball_pos"])
        own_goal = np.array([0.0, -C.BACK_NET_Y, C.GOAL_HEIGHT / 2])
        to_goal = own_goal - car
        fwd = np.array(w["car_forward"])
        ball_to_goal_dir = (own_goal - np.array(nxt["ball_pos"]))
        ball_to_goal_dir /= max(np.linalg.norm(ball_to_goal_dir), 1e-6)
        last = {
            "decisions_before_goal": len(window) - 1 - touch_idx[-1],
            "height": float(car[2]), "airborne": not w["on_ground"], "upside_down": w["car_up_z"] < 0,
            "flipping": w["flipping"], "speed": float(np.linalg.norm(w["car_vel"])),
            "facing_own_goal_cos": float(np.dot(fwd, to_goal / max(np.linalg.norm(to_goal), 1e-6))),
            "attacking_side_of_ball": bool(car[1] > ball[1]),  # ball between car and own goal (team frame: own goal at -y)
            "ball_speed_to_own_goal_after": float(np.dot(np.array(nxt["ball_vel"]), ball_to_goal_dir)),
            "dist_to_own_goal": float(np.linalg.norm(ball - own_goal)),
        }
        last_reward = w["reward"]
        nose_down = (not w["on_ground"]) and w["car_forward"][2] < -0.5
        if w["car_up_z"] < 0 or w["flipping"] or nose_down:
            category = "control_recovery"
        elif last["attacking_side_of_ball"] and last["facing_own_goal_cos"] > 0.3:
            category = "positioning"
    shaping = sum(v for w in window for k, v in w["reward"].items() if not k.startswith("goal_"))
    since_touch = window[touch_idx[-1]:] if touch_idx else []
    post_touch_shaping = sum(v for w in since_touch for k, v in w["reward"].items() if not k.startswith("goal_"))
    return {"seed": seed, "team": team, "category": category, "last_touch": last, "last_touch_reward": last_reward,
            "window_shaping_reward": float(shaping),
            "shaping_from_last_touch_to_goal": float(post_touch_shaping),
            "window_reward_by_component": {k: round(sum(w["reward"].get(k, 0.0) for w in window), 4)
                                           for k in TRAIN_WEIGHTS},
            "frames": [w["frame"] for w in window]}


if __name__ == "__main__":
    main()
