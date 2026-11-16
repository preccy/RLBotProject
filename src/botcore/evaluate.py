"""Headless head-to-head evaluation of two compatible policies in RocketSim.

Protocol (fixed, recorded in every report):
* For each evaluation seed s, two matches are played: A blue vs B orange, then B blue vs A orange. Both
  matches use the same seed, so kickoff spawn sequences and each side's sampling stream are matched.
* Seeds are EVAL_SEED_BASE + i (i = 0..n-1): a frozen benchmark set disjoint from every training seed.
* A match: kickoff, play until a goal, kickoff again; the clock counts simulated game time only.
  Regulation is `match_seconds`. If tied, overtime continues until the next goal, capped at
  `overtime_max_seconds`; a match still tied after that is a draw. No goal replays, no countdown and no
  "ball must touch the ground" rule at zero (simplifications vs. real Rocket League).
* Sampling unit for uncertainty is the colour-swapped seed PAIR (the two matches share a seed and are
  correlated); goals inside a match are never treated as independent samples. The interval is a
  percentile bootstrap over pairs of the pair-mean score (win = 1, draw = 0.5, loss = 0); when every pair is
  a clean sweep (all 1 or all 0) the exact Clopper-Pearson bound over pairs is used instead of [1, 1].
* Deterministic (argmax) vs stochastic (sampled) inference is chosen per side and recorded.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from botcore import constants as C
from botcore.config import EVAL_SEED_BASE

EVAL_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class MatchRules:
    team_size: int = 1
    match_seconds: float = 300.0
    overtime_max_seconds: float = 120.0


@dataclass(frozen=True)
class Side:
    bundle: str  # export bundle directory
    deterministic: bool = False


@dataclass
class MatchResult:
    seed: int
    blue: str  # "A" or "B"
    goals_blue: int
    goals_orange: int
    seconds_played: float
    overtime: bool
    # behaviour diagnostics per colour: {"blue": {...}, "orange": {...}} (not used for judging strength)
    behaviour: Dict[str, Dict[str, float]] = None

    def score_for(self, side: str) -> float:
        mine, theirs = (self.goals_blue, self.goals_orange) if self.blue == side else (self.goals_orange, self.goals_blue)
        return 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0

    def goal_diff_for(self, side: str) -> int:
        diff = self.goals_blue - self.goals_orange
        return diff if self.blue == side else -diff


_CACHE: Dict[str, object] = {}


def _runner(side: Side, seed: int):
    from botcore.export import PolicyRunner, load_bundle

    bundle = _CACHE.get(side.bundle)
    if bundle is None:
        bundle = load_bundle(side.bundle, device="cpu")
        _CACHE[side.bundle] = bundle
    return PolicyRunner(bundle, device="cpu", deterministic=side.deterministic, seed=seed)


def play_match(seed: int, blue: Side, orange: Side, blue_label: str, rules: MatchRules) -> MatchResult:
    """Play one full match. Pure function of its arguments (deterministic given the seed)."""
    import torch

    from botcore.env import EnvSpec, build_env

    torch.set_num_threads(1)
    runners = {C.BLUE_TEAM: _runner(blue, seed * 2 + 0), C.ORANGE_TEAM: _runner(orange, seed * 2 + 1)}
    repeats = {r.bundle.action_repeat for r in runners.values()}
    delays = {r.bundle.rlbot_delay for r in runners.values()}
    if len(repeats) != 1 or len(delays) != 1:
        raise ValueError(f"both policies must use the same action_repeat and input delay (got {repeats}, {delays})")
    action_repeat = repeats.pop()
    spec = EnvSpec(
        team_size=rules.team_size, action_repeat=action_repeat, rlbot_delay=delays.pop(),
        no_touch_timeout_seconds=1e9, episode_timeout_seconds=1e9,
        scenario_weights={"kickoff": 1.0}, reward_weights={},
    )
    env = build_env(spec, seed)
    obs = env.reset()
    agents = list(env.agents)
    teams = {a: env.state.cars[a].team_num for a in agents}
    by_team = {t: [a for a in agents if teams[a] == t] for t in (C.BLUE_TEAM, C.ORANGE_TEAM)}

    goals = {C.BLUE_TEAM: 0, C.ORANGE_TEAM: 0}
    regulation_ticks = int(round(rules.match_seconds * C.TICKS_PER_SECOND))
    overtime_ticks = int(round(rules.overtime_max_seconds * C.TICKS_PER_SECOND))
    ticks = 0
    overtime = False
    acc = {t: {"touches": 0.0, "speed": 0.0, "ball_dist": 0.0, "airborne": 0.0, "boost": 0.0, "supersonic": 0.0,
               "n": 0, "own_goals": 0} for t in (C.BLUE_TEAM, C.ORANGE_TEAM)}
    last_toucher = None  # team that touched the ball last (own-goal attribution)
    while True:
        actions = {}
        for team, team_agents in by_team.items():
            chosen = runners[team].act_batch(np.stack([obs[a] for a in team_agents]))
            for a, idx in zip(team_agents, chosen):
                actions[a] = np.array([idx])
        obs, _, terminated, _ = env.step(actions)
        ticks += action_repeat
        state = env.state
        ball = np.asarray(state.ball.position)
        for a in agents:
            car, d = state.cars[a], acc[teams[a]]
            if car.ball_touches > 0:
                last_toucher = teams[a]
            d["touches"] += float(car.ball_touches > 0)
            d["speed"] += float(np.linalg.norm(car.physics.linear_velocity))
            d["ball_dist"] += float(np.linalg.norm(np.asarray(car.physics.position) - ball))
            d["airborne"] += float(not car.on_ground)
            d["boost"] += float(car.boost_amount)
            d["supersonic"] += float(car.is_supersonic)
            d["n"] += 1
        if any(terminated.values()):
            goals[env.state.scoring_team] += 1
            if last_toucher is not None and last_toucher != env.state.scoring_team:
                acc[last_toucher]["own_goals"] += 1
            last_toucher = None
            if overtime:
                break
            obs = env.reset()
        if not overtime and ticks >= regulation_ticks:
            if goals[C.BLUE_TEAM] != goals[C.ORANGE_TEAM] or overtime_ticks == 0:
                break
            overtime = True
        if overtime and ticks >= regulation_ticks + overtime_ticks:
            break
    env.close()
    minutes = ticks / C.TICKS_PER_SECOND / 60.0
    behaviour = {}
    for team, name in ((C.BLUE_TEAM, "blue"), (C.ORANGE_TEAM, "orange")):
        d, n = acc[team], max(1, acc[team]["n"])
        cars = len(by_team[team])
        behaviour[name] = {"touch_decisions_per_min": d["touches"] / cars / minutes, "mean_speed": d["speed"] / n,
                           "mean_ball_distance": d["ball_dist"] / n, "airborne_fraction": d["airborne"] / n,
                           "mean_boost": d["boost"] / n, "supersonic_fraction": d["supersonic"] / n,
                           "own_goals_per_match": float(d["own_goals"])}
    return MatchResult(seed, blue_label, goals[C.BLUE_TEAM], goals[C.ORANGE_TEAM], ticks / C.TICKS_PER_SECOND, overtime,
                       behaviour)


def _play_job(args) -> MatchResult:
    return play_match(*args)


def eval_seeds(n_pairs: int, offset: int = 0) -> List[int]:
    return [EVAL_SEED_BASE + offset + i for i in range(n_pairs)]


def bootstrap_ci(values: np.ndarray, n_resamples: int = 5000, level: float = 0.95, seed: int = 0,
                 unit_interval: bool = False) -> tuple[float, float]:
    """Percentile bootstrap of the mean. With `unit_interval` (values in [0, 1], e.g. pair scores), a sample that is
    entirely 1 (or entirely 0) gets the exact Clopper-Pearson bound instead of the degenerate [1, 1]: no resample can
    contain a loss, but the true rate may still be below 1. For any [0, 1] variable with mean m, P(X = 1) <= m, so
    P(all n = 1) <= m^n and the bound ((1 - level) / 2)^(1/n) is valid (conservative) for the mean score too."""
    if len(values) == 0:
        return float("nan"), float("nan")
    values = np.asarray(values, dtype=float)
    if unit_interval and (np.all(values == 1.0) or np.all(values == 0.0)):
        bound = ((1 - level) / 2) ** (1.0 / len(values))
        return (bound, 1.0) if values[0] == 1.0 else (0.0, 1.0 - bound)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def summarise(results: List[MatchResult], seeds: List[int]) -> dict:
    by_seed: Dict[int, List[MatchResult]] = {}
    for r in results:
        by_seed.setdefault(r.seed, []).append(r)
    pair_scores = np.array([np.mean([r.score_for("A") for r in by_seed[s]]) for s in seeds])
    pair_gd = np.array([np.mean([r.goal_diff_for("A") for r in by_seed[s]]) for s in seeds])
    scores = [r.score_for("A") for r in results]
    return {
        "matches": len(results),
        "seed_pairs": len(seeds),
        "A_wins": int(sum(s == 1.0 for s in scores)),
        "draws": int(sum(s == 0.5 for s in scores)),
        "A_losses": int(sum(s == 0.0 for s in scores)),
        "A_goals_for": int(sum(r.goals_blue if r.blue == "A" else r.goals_orange for r in results)),
        "A_goals_against": int(sum(r.goals_orange if r.blue == "A" else r.goals_blue for r in results)),
        "A_score": float(np.mean(scores)),
        "A_score_ci95_pair_bootstrap": bootstrap_ci(pair_scores, unit_interval=True),
        "A_goal_diff_per_match": float(np.mean(pair_gd)),
        "A_goal_diff_ci95_pair_bootstrap": bootstrap_ci(pair_gd, seed=1),
        "overtime_matches": int(sum(r.overtime for r in results)),
        "A_score_as_blue": float(np.mean([r.score_for("A") for r in results if r.blue == "A"])),
        "A_score_as_orange": float(np.mean([r.score_for("A") for r in results if r.blue == "B"])),
        "decisive_pairs": int(sum(abs(v - 0.5) > 1e-9 for v in pair_scores)),
        "behaviour": _behaviour_summary(results),
    }


def _behaviour_summary(results: List[MatchResult]) -> dict:
    """Mean behaviour diagnostics for side A and side B across matches (diagnostic only)."""
    out = {}
    for side in ("A", "B"):
        rows = [r.behaviour["blue" if r.blue == side else "orange"] for r in results if r.behaviour]
        if rows:
            out[side] = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
    return out


def evaluate(a: Side, b: Side, rules: MatchRules, n_pairs: int, workers: Optional[int] = None,
             seed_offset: int = 0) -> dict:
    from botcore.export import load_bundle

    bundles = {"A": load_bundle(a.bundle), "B": load_bundle(b.bundle)}
    seeds = eval_seeds(n_pairs, seed_offset)
    jobs = []
    for s in seeds:
        jobs.append((s, a, b, "A", rules))
        jobs.append((s, b, a, "B", rules))
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    t0 = time.perf_counter()
    if workers == 1:
        results = [_play_job(j) for j in jobs]
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_play_job, jobs))
    elapsed = time.perf_counter() - t0
    return {
        "protocol_version": EVAL_PROTOCOL_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules": asdict(rules) | {
            "kickoff_after_goal": True, "goal_replays": False, "ball_must_touch_ground_at_zero": False,
            "draw_if_tied_after_overtime": True, "simulator": "RocketSim via rlgym RocketSimEngine",
        },
        "seeds": seeds,
        "seed_range_note": f"evaluation seeds start at {EVAL_SEED_BASE}; training seeds are always below it",
        "sides": {
            name: {
                "bundle": str(Path(side.bundle).resolve()),
                "weights_sha256": bundles[name].manifest["weights_sha256"],
                "source": bundles[name].manifest.get("source"),
                "inference": "deterministic (argmax)" if side.deterministic else "stochastic (sampled)",
                "action_repeat": bundles[name].action_repeat,
            }
            for name, side in (("A", a), ("B", b))
        },
        "summary": summarise(results, seeds),
        "matches": [asdict(r) for r in results],
        "wall_seconds": elapsed,
        "interpretation": (
            "Relative strength between these two policies under these rules in RocketSim only. Not Rocket League "
            "MMR or rank. With few seed pairs the interval is wide; this validates the harness, not playing strength."
        ),
    }


def format_summary(report: dict) -> str:
    s = report["summary"]
    lo, hi = s["A_score_ci95_pair_bootstrap"]
    a, b = report["sides"]["A"], report["sides"]["B"]
    return (
        f"A = {a['bundle']} [{a['weights_sha256'][:12]}, {a['inference']}]\n"
        f"B = {b['bundle']} [{b['weights_sha256'][:12]}, {b['inference']}]\n"
        f"{s['matches']} matches ({s['seed_pairs']} colour-swapped seed pairs), "
        f"{report['rules']['match_seconds']:.0f}s + up to {report['rules']['overtime_max_seconds']:.0f}s overtime\n"
        f"A: {s['A_wins']}W {s['draws']}D {s['A_losses']}L, goals {s['A_goals_for']}-{s['A_goals_against']}, "
        f"score {s['A_score']:.3f} (95% CI over seed pairs {lo:.3f}-{hi:.3f}), "
        f"blue {s['A_score_as_blue']:.3f} / orange {s['A_score_as_orange']:.3f}\n"
        f"wall time {report['wall_seconds']:.1f}s"
    )


# --------------------------------------------------------------------------------------- promotion
PROMOTION_MIN_PAIRS = 100
PROMOTION_MIN_LOWER_BOUND = 0.5  # candidate must beat best-verified with the CI lower bound above 0.5
REGRESSION_MAX_DROP = 0.05  # vs every other benchmark opponent: score must be >= 0.5 - this


def check_promotion(candidate_sha: str, best_sha: Optional[str], reports: List[dict]) -> List[str]:
    """Return reasons the candidate may NOT be promoted (empty list = promotion allowed)."""
    problems = []
    if not reports:
        return ["no evaluation reports given"]
    versus_best = None
    for rep in reports:
        if rep["sides"]["A"]["weights_sha256"] != candidate_sha:
            problems.append(f"report {rep.get('created_at')} does not have the candidate as side A")
            continue
        s = rep["summary"]
        if s["seed_pairs"] < PROMOTION_MIN_PAIRS:
            problems.append(f"report vs {rep['sides']['B']['weights_sha256'][:12]} has {s['seed_pairs']} seed pairs "
                            f"(< {PROMOTION_MIN_PAIRS} required)")
        if best_sha is not None and rep["sides"]["B"]["weights_sha256"] == best_sha:
            versus_best = rep
            if s["A_score_ci95_pair_bootstrap"][0] <= PROMOTION_MIN_LOWER_BOUND:
                problems.append(f"vs best-verified: CI lower bound {s['A_score_ci95_pair_bootstrap'][0]:.3f} "
                                f"is not above {PROMOTION_MIN_LOWER_BOUND}")
        elif s["A_score"] < 0.5 - REGRESSION_MAX_DROP:
            problems.append(f"regression vs {rep['sides']['B']['weights_sha256'][:12]}: score {s['A_score']:.3f}")
    if best_sha is not None and versus_best is None:
        problems.append("no report comparing the candidate with the current best-verified policy")
    return problems
