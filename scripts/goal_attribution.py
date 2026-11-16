"""Diagnostic: who last touched the ball before each goal (own goals vs real goals), in RocketSim matches.

    .\\.venv\\Scripts\\python.exe scripts\\goal_attribution.py <policy bundle> <opponent bundle> [n_seeds] [seed_offset]
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
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    offset = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    ba, bb = load_bundle(a), load_bundle(b)
    counts = {"A_scored": 0, "A_own_goal": 0, "B_scored": 0, "B_own_goal": 0, "untouched": 0}
    for s in range(n_seeds):
        seed = EVAL_SEED_BASE + offset + s
        for a_team in (C.BLUE_TEAM, C.ORANGE_TEAM):
            runners = {a_team: PolicyRunner(ba, seed=seed * 2), 1 - a_team: PolicyRunner(bb, seed=seed * 2 + 1)}
            spec = EnvSpec(team_size=1, action_repeat=8, rlbot_delay=True, no_touch_timeout_seconds=1e9,
                           episode_timeout_seconds=1e9, scenario_weights={"kickoff": 1.0}, reward_weights={})
            env = build_env(spec, seed)
            obs = env.reset()
            last = None
            for _ in range(int(300 * 120 / 8)):
                st = env.state
                acts = {ag: np.array([runners[st.cars[ag].team_num].act(obs[ag])]) for ag in env.agents}
                obs, _, term, _ = env.step(acts)
                for ag, car in env.state.cars.items():
                    if car.ball_touches > 0:
                        last = car.team_num
                if any(term.values()):
                    scorer = env.state.scoring_team
                    side = "A" if scorer == a_team else "B"
                    if last is None:
                        counts["untouched"] += 1
                    elif last == scorer:
                        counts[f"{side}_scored"] += 1
                    else:  # the conceding team touched it last
                        counts[f"{'B' if side == 'A' else 'A'}_own_goal"] += 1
                    last = None
                    obs = env.reset()
            env.close()
    print(f"A={a.name} B={b.name}, {2 * n_seeds} matches x 300 s: {counts}")


if __name__ == "__main__":
    main()
