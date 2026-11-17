"""Evaluations and decision for the VTB comparison, exactly as pre-registered in docs/EXPERIMENT_VTB.md.

    .\\.venv\\Scripts\\python.exe scripts\\vtb_eval.py [eval|diag|decide|all] [final_iteration]

Idempotent: existing reports are reused. All policies sample their actions (stochastic), identically for both branches.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from long1_eval import bootstrap_mean_ci, own_goals_by_pair, side_stats  # noqa: E402

OUT = ROOT / "runs" / "vtb-eval"
REF = ROOT / "runs" / "reference"
START = REF / "S2N-iter9900"  # best_verified and deployed at the start
S1L = REF / "S1L-iter2263"
R0 = REF / "R0-random-init"
M2REF = REF / "M2-D-s202-iter160"
RUNS = {"CTRL": ROOT / "runs" / "s3-ctrl", "EXP": ROOT / "runs" / "s3-novtb"}
FINAL_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 11900
DIAG = dict(pairs=40, seed_offset=0)
DEFENCE_OFFSET, DEFENCE_N, DEFENCE_SECONDS = 13000, 200, 15.0
GOALSIDE_OFFSET, GOALSIDE_PAIRS = 14000, 20


def bundle(branch: str, iteration: int = FINAL_ITER) -> Path:
    from botcore.export import bundle_from_checkpoint

    run = RUNS[branch]
    out = run / "exports" / f"iter_{iteration:06d}"
    if not out.exists():
        bundle_from_checkpoint(run / "checkpoints" / f"iter_{iteration:06d}", out)
    return out


def run_eval(name: str, a: Path, b: Path, pairs: int, seed_offset: int) -> dict:
    from botcore.evaluate import MatchRules, Side, evaluate, format_summary

    path = OUT / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rep = evaluate(Side(str(a)), Side(str(b)), MatchRules(), n_pairs=pairs, seed_offset=seed_offset)
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[{name}]\n{format_summary(rep)}", flush=True)
    return rep


def evals() -> dict:
    e, c = bundle("EXP"), bundle("CTRL")
    reps = {"h2h": run_eval("h2h-EXP-vs-CTRL", e, c, pairs=150, seed_offset=10000)}
    for br, b in (("CTRL", c), ("EXP", e)):
        reps[f"{br}_vs_START"] = run_eval(f"{br}-vs-S2N9900", b, START, pairs=150, seed_offset=11000)
        reps[f"{br}_vs_S1L"] = run_eval(f"{br}-vs-S1L2263", b, S1L, pairs=150, seed_offset=12000)
        reps[f"{br}_vs_R0"] = run_eval(f"{br}-vs-R0", b, R0, **DIAG)
        reps[f"{br}_vs_M2REF"] = run_eval(f"{br}-vs-M2REF", b, M2REF, **DIAG)
    return reps


# ------------------------------------------------------------------ supporting diagnostics
def _defence_episode(args) -> int:
    """+1 defender scored, -1 conceded, 0 neither within DEFENCE_SECONDS. Both policies sample."""
    defender, attacker, seed = args
    import torch

    from botcore import constants as C
    from botcore.env import EnvSpec, build_env
    from botcore.evaluate import EVAL_SEED_BASE
    from botcore.export import PolicyRunner, load_bundle

    torch.set_num_threads(1)
    spec = EnvSpec(team_size=1, action_repeat=8, rlbot_delay=True, no_touch_timeout_seconds=1e9,
                   episode_timeout_seconds=1e9, scenario_weights={"save": 0.5, "low_boost_defence": 0.5},
                   reward_weights={})
    s = EVAL_SEED_BASE + seed
    env = build_env(spec, s)
    obs = env.reset()
    # The ball always starts moving towards the defender's goal: -y for a blue defender, +y for orange.
    def_team = C.BLUE_TEAM if float(env.state.ball.linear_velocity[1]) < 0 else C.ORANGE_TEAM
    runners = {def_team: PolicyRunner(load_bundle(defender, device="cpu"), seed=2 * s),
               1 - def_team: PolicyRunner(load_bundle(attacker, device="cpu"), seed=2 * s + 1)}
    for _ in range(int(DEFENCE_SECONDS * 15)):
        actions = {a: np.array([runners[env.state.cars[a].team_num].act(obs[a])]) for a in env.agents}
        obs, _, term, _ = env.step(actions)
        if any(term.values()):
            out = 1 if env.state.scoring_team == def_team else -1
            env.close()
            return out
    env.close()
    return 0


def _goalside_match(args) -> dict:
    """One 300 s match from kickoff (goals reset to kickoff); fraction of decisions each team's car is goal-side."""
    a_path, b_path, seed, a_blue = args
    import torch

    from botcore import constants as C
    from botcore.env import EnvSpec, build_env
    from botcore.evaluate import EVAL_SEED_BASE
    from botcore.export import PolicyRunner, load_bundle

    torch.set_num_threads(1)
    spec = EnvSpec(team_size=1, action_repeat=8, rlbot_delay=True, no_touch_timeout_seconds=1e9,
                   episode_timeout_seconds=1e9, scenario_weights={"kickoff": 1.0}, reward_weights={})
    s = EVAL_SEED_BASE + seed
    env = build_env(spec, s)
    obs = env.reset()
    a_team = C.BLUE_TEAM if a_blue else C.ORANGE_TEAM
    runners = {a_team: PolicyRunner(load_bundle(a_path, device="cpu"), seed=2 * s + (0 if a_blue else 1)),
               1 - a_team: PolicyRunner(load_bundle(b_path, device="cpu"), seed=2 * s + (1 if a_blue else 0))}
    goalside = {"A": 0, "B": 0}
    n = 0
    for _ in range(300 * 15):
        actions = {ag: np.array([runners[env.state.cars[ag].team_num].act(obs[ag])]) for ag in env.agents}
        obs, _, term, _ = env.step(actions)
        by = float(env.state.ball.position[1])
        for ag in env.agents:
            car = env.state.cars[ag]
            sign = 1.0 if car.team_num == C.BLUE_TEAM else -1.0  # own goal at -sign*5120
            side = "A" if car.team_num == a_team else "B"
            goalside[side] += int(sign * float(car.physics.position[1]) < sign * by)
        n += 1
        if any(term.values()):
            obs = env.reset()
    env.close()
    return {"seed": seed, "A": goalside["A"] / n, "B": goalside["B"] / n}


def diagnostics() -> dict:
    path = OUT / "diagnostics.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    pols = {"S2N9900": START, "CTRL": bundle("CTRL"), "EXP": bundle("EXP")}
    out: dict = {"defence": {}, "goalside_vs_S2N9900": {}}
    seeds = [DEFENCE_OFFSET + i for i in range(DEFENCE_N)]
    with ProcessPoolExecutor(12) as ex:
        for name, p in pols.items():
            res = np.array(list(ex.map(_defence_episode, [(str(p), str(START), s) for s in seeds])))
            conceded = (res == -1).astype(float)
            mean, lo, hi = bootstrap_mean_ci(conceded)
            out["defence"][name] = {"episodes": len(res), "conceded_rate": round(mean, 3),
                                    "ci95": [round(lo, 3), round(hi, 3)], "defender_scored": int((res == 1).sum()),
                                    "neither": int((res == 0).sum())}
            print(name, "defence", out["defence"][name], flush=True)
        for name in ("CTRL", "EXP"):
            jobs = [(str(pols[name]), str(START), GOALSIDE_OFFSET + i, blue)
                    for i in range(GOALSIDE_PAIRS) for blue in (True, False)]
            rows = list(ex.map(_goalside_match, jobs))
            a = np.array([r["A"] for r in rows])
            b = np.array([r["B"] for r in rows])
            out["goalside_vs_S2N9900"][name] = {"matches": len(rows), "candidate": round(float(a.mean()), 3),
                                                "S2N9900": round(float(b.mean()), 3)}
            print(name, "goal-side", out["goalside_vs_S2N9900"][name], flush=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def behaviour(rep: dict, side: str) -> dict:
    keys = ("touch_decisions_per_min", "airborne_fraction", "mean_speed", "supersonic_fraction", "mean_boost",
            "mean_ball_distance", "own_goals_per_match")
    rows = [m["behaviour"]["blue" if m["blue"] == side else "orange"] for m in rep["matches"]]
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}


def decide() -> dict:
    reps = evals()
    h = side_stats(reps["h2h"], "A")
    verdict = "EXP better" if h["ci95"][0] > 0.5 else "EXP worse" if h["ci95"][1] < 0.5 else "inconclusive"
    result: dict = {"final_iteration": FINAL_ITER, "action_selection": "sampled (stochastic) for every policy",
                    "head_to_head_EXP_vs_CTRL": h, "verdict": verdict, "branches": {}}
    a, o = own_goals_by_pair(reps["h2h"], "A"), own_goals_by_pair(reps["h2h"], "B")
    mean, lo, hi = bootstrap_mean_ci(np.array([a[k] - o[k] for k in sorted(a)]))
    result["h2h_own_goal_diff_EXP_minus_CTRL_per_match"] = {"mean": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    qualifies = {}
    for br in ("CTRL", "EXP"):
        vs = {k: side_stats(reps[f"{br}_{k}"], "A") for k in ("vs_START", "vs_S1L", "vs_R0", "vs_M2REF")}
        regression_ok = vs["vs_R0"]["score"] >= 0.45 and vs["vs_M2REF"]["score"] >= 0.45
        qualifies[br] = vs["vs_START"]["ci95"][0] > 0.5 and regression_ok
        result["branches"][br] = {**vs, "behaviour_vs_START": {"candidate": behaviour(reps[f"{br}_vs_START"], "A"),
                                                               "S2N9900": behaviour(reps[f"{br}_vs_START"], "B")},
                                  "regression_checks_pass": regression_ok, "qualifies_for_promotion": qualifies[br]}
    if qualifies["CTRL"] and qualifies["EXP"]:
        candidate = "EXP" if verdict == "EXP better" else "CTRL"
    elif qualifies["CTRL"] or qualifies["EXP"]:
        candidate = "CTRL" if qualifies["CTRL"] else "EXP"
    else:
        candidate = None
    result["promotion_candidate"] = candidate
    result["best_verified_after"] = f"{candidate}-iter{FINAL_ITER}" if candidate else "S2N-iter9900 (unchanged)"
    (OUT / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("eval", "all"):
        evals()
    if what in ("decide", "all"):
        decide()
    if what in ("diag", "all"):
        diagnostics()
