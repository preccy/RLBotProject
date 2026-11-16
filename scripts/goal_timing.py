"""Diagnostic: time from each kickoff to the goal, and who scored, in RocketSim matches.

    .\\.venv\\Scripts\\python.exe scripts\\goal_timing.py <policy A> <policy B> [n_seeds] [seed_offset]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from botcore import constants as C  # noqa: E402
from botcore.config import EVAL_SEED_BASE  # noqa: E402
from botcore.env import EnvSpec, build_env  # noqa: E402
from botcore.export import PolicyRunner, load_bundle  # noqa: E402


def main() -> None:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    offset = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    ba, bb = load_bundle(a), load_bundle(b)
    times = {"A": [], "B": []}
    kickoff_touch_first = {"A": 0, "B": 0}
    for s in range(n_seeds):
        seed = EVAL_SEED_BASE + offset + s
        for a_team in (C.BLUE_TEAM, C.ORANGE_TEAM):
            runners = {a_team: PolicyRunner(ba, seed=seed * 2), 1 - a_team: PolicyRunner(bb, seed=seed * 2 + 1)}
            spec = EnvSpec(team_size=1, action_repeat=8, rlbot_delay=True, no_touch_timeout_seconds=1e9,
                           episode_timeout_seconds=1e9, scenario_weights={"kickoff": 1.0}, reward_weights={})
            env = build_env(spec, seed)
            obs = env.reset()
            since, first = 0, None
            for _ in range(int(300 * 120 / 8)):
                st = env.state
                acts = {ag: np.array([runners[st.cars[ag].team_num].act(obs[ag])]) for ag in env.agents}
                obs, _, term, _ = env.step(acts)
                since += 1
                if first is None:
                    for c in env.state.cars.values():
                        if c.ball_touches > 0:
                            first = "A" if c.team_num == a_team else "B"
                if any(term.values()):
                    side = "A" if env.state.scoring_team == a_team else "B"
                    times[side].append(since * 8 / 120)
                    if first:
                        kickoff_touch_first[first] += 1
                    since, first = 0, None
                    obs = env.reset()
            env.close()
    for side in ("A", "B"):
        t = np.array(times[side])
        if len(t):
            print(f"{side} ({a.name if side == 'A' else b.name}): {len(t)} goals; seconds from kickoff to goal: "
                  f"median {np.median(t):.1f}, 10% {np.percentile(t, 10):.1f}, 90% {np.percentile(t, 90):.1f}; "
                  f"within 5 s: {np.mean(t <= 5):.0%}, within 10 s: {np.mean(t <= 10):.0%}")
        else:
            print(f"{side}: no goals")
    print(f"first touch after kickoff (episodes ending in a goal): {kickoff_touch_first}")


if __name__ == "__main__":
    main()
