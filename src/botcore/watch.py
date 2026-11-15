"""Optional single-environment viewer using RLViser (never used during training or benchmarks)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from botcore import constants as C

ROOT = Path(__file__).resolve().parents[2]
RLVISER_HELP = (
    "RLViser is not installed. Download rlviser.exe from https://github.com/VirxEC/rlviser/releases and put it in "
    f"{ROOT}. rlviser-py starts it from the current working directory on the first render call."
)


def watch(bundle_a: Path, bundle_b: Optional[Path], seconds: float, scenario: str, seed: int,
          deterministic: bool = False, speed: float = 1.0, dry_run: bool = False, log=print) -> None:
    from botcore.env import EnvSpec, build_env
    from botcore.export import PolicyRunner, load_bundle

    blue = load_bundle(bundle_a)
    orange = load_bundle(bundle_b) if bundle_b else blue
    if blue.action_repeat != orange.action_repeat:
        raise ValueError("both bundles must use the same action_repeat")
    renderer = None
    if not dry_run:
        if not (ROOT / "rlviser.exe").is_file():
            raise FileNotFoundError(RLVISER_HELP)
        os.chdir(ROOT)
        from rlgym.rocket_league.rlviser import RLViserRenderer

        renderer = RLViserRenderer(tick_rate=C.TICKS_PER_SECOND / blue.action_repeat)
    spec = EnvSpec(team_size=1, action_repeat=blue.action_repeat, rlbot_delay=blue.rlbot_delay,
                   no_touch_timeout_seconds=30.0, episode_timeout_seconds=1e9,
                   scenario_weights={scenario: 1.0}, reward_weights={})
    env = build_env(spec, seed, renderer=renderer)
    runners = {C.BLUE_TEAM: PolicyRunner(blue, deterministic=deterministic, seed=seed),
               C.ORANGE_TEAM: PolicyRunner(orange, deterministic=deterministic, seed=seed + 1)}
    obs = env.reset()
    step_dt = blue.action_repeat / C.TICKS_PER_SECOND / speed
    n_steps = int(seconds * C.TICKS_PER_SECOND / blue.action_repeat)
    goals = {0: 0, 1: 0}
    next_time = time.perf_counter()
    try:
        for _ in range(n_steps):
            actions = {a: np.array([runners[env.state.cars[a].team_num].act(obs[a])]) for a in env.agents}
            obs, _, term, trunc = env.step(actions)
            if renderer is not None:
                env.render()
                next_time += step_dt
                time.sleep(max(0.0, next_time - time.perf_counter()))
            if any(term.values()):
                goals[env.state.scoring_team] += 1
            if any(term.values()) or any(trunc.values()):
                obs = env.reset()
    finally:
        env.close()
    log(f"watched {seconds:.0f}s of game time; goals blue {goals[0]} - orange {goals[1]}"
        + (" (dry run, nothing rendered)" if dry_run else ""))
