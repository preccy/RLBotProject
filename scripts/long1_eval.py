"""Evaluations and decision for long run 1, exactly as pre-registered in docs/EXPERIMENT_LONG1.md.

    .\\.venv\\Scripts\\python.exe scripts\\long1_eval.py [midpoint|final|decide|all]

Idempotent: existing reports are reused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN = ROOT / "runs" / "s1-long"
OUT = RUN / "eval"
R0 = ROOT / "runs" / "reference" / "R0-random-init"
M2REF = ROOT / "runs" / "reference" / "M2-D-s202-iter160"
MID_ITER, FINAL_ITER = 1200, 2263
DIAG = dict(pairs=40, seed_offset=0)
FRESH = dict(pairs=150, seed_offset=1000)


def bundle(iteration: int) -> Path:
    from botcore.export import bundle_from_checkpoint

    out = RUN / "exports" / f"iter_{iteration:06d}"
    if not out.exists():
        bundle_from_checkpoint(RUN / "checkpoints" / f"iter_{iteration:06d}", out)
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


def side_stats(rep: dict, side: str = "A") -> dict:
    """Per-match aggregates for one side: goals, own goals scored into own net, touches/min, airborne fraction."""
    gf = ga = own = 0
    touches, airborne = [], []
    for m in rep["matches"]:
        colour = "blue" if m["blue"] == side else "orange"
        other = "orange" if colour == "blue" else "blue"
        mine = m["goals_blue"] if colour == "blue" else m["goals_orange"]
        theirs = m["goals_orange"] if colour == "blue" else m["goals_blue"]
        gf += mine
        ga += theirs
        own += m["behaviour"][colour]["own_goals_per_match"]
        touches.append(m["behaviour"][colour]["touch_decisions_per_min"])
        airborne.append(m["behaviour"][colour]["airborne_fraction"])
    n = len(rep["matches"])
    s = rep["summary"]
    lo, hi = s["A_score_ci95_pair_bootstrap"]
    if side == "A":
        wdl, score, ci = f"{s['A_wins']}-{s['draws']}-{s['A_losses']}", s["A_score"], [lo, hi]
    else:
        wdl, score, ci = f"{s['A_losses']}-{s['draws']}-{s['A_wins']}", 1 - s["A_score"], [1 - hi, 1 - lo]
    return {"matches": n, "W-D-L": wdl, "score": round(score, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)],
            "goals_for": gf, "goals_against": ga, "own_goals_per_match": round(own / n, 3),
            "own_goal_share_of_conceded": round(own / ga, 3) if ga else None,
            "touches_per_min": round(float(np.mean(touches)), 2), "airborne_fraction": round(float(np.mean(airborne)), 3)}


def own_goals_by_pair(rep: dict, side: str = "A") -> dict:
    per = {}
    for m in rep["matches"]:
        colour = "blue" if m["blue"] == side else "orange"
        per.setdefault(m["seed"], []).append(m["behaviour"][colour]["own_goals_per_match"])
    return {s: float(np.mean(v)) for s, v in per.items()}


def bootstrap_mean_ci(values: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def midpoint() -> None:
    b = bundle(MID_ITER)
    run_eval("diag-mid-vs-R0", b, R0, **DIAG)
    run_eval("diag-mid-vs-M2REF", b, M2REF, **DIAG)


def final() -> None:
    b = bundle(FINAL_ITER)
    run_eval("diag-final-vs-R0", b, R0, **DIAG)
    run_eval("diag-final-vs-M2REF", b, M2REF, **DIAG)
    run_eval("decision-final-vs-M2REF", b, M2REF, **FRESH)
    run_eval("decision-final-vs-R0", b, R0, **FRESH)
    run_eval("decision-M2REF-vs-R0", M2REF, R0, **FRESH)


def decide() -> dict:
    load = lambda n: json.loads((OUT / f"{n}.json").read_text(encoding="utf-8"))
    vs_m2, vs_r0, ref_r0 = load("decision-final-vs-M2REF"), load("decision-final-vs-R0"), load("decision-M2REF-vs-R0")
    f_m2, f_r0, m_r0 = side_stats(vs_m2), side_stats(vs_r0), side_stats(ref_r0)
    improvement = f_m2["ci95"][0] > 0.5 and f_m2["score"] >= 0.60
    regression = f_m2["ci95"][1] < 0.5
    beats_random = f_r0["ci95"][0] > 0.5
    a, b = own_goals_by_pair(vs_r0), own_goals_by_pair(ref_r0)
    diff = np.array([a[s] - b[s] for s in sorted(a)])
    mean, lo, hi = bootstrap_mean_ci(diff)
    own_goal_reduction = hi < 0
    verdict = "demonstrated improvement" if improvement else "regression" if regression else "inconclusive"
    result = {
        "final_vs_M2REF": f_m2, "final_vs_R0": f_r0, "M2REF_vs_R0": m_r0,
        "M2REF_side_in_final_vs_M2REF": side_stats(vs_m2, "B"),
        "own_goal_difference_final_minus_M2REF_per_match": {"mean": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)],
                                                            "pairs": len(diff)},
        "criteria": {"improvement_vs_M2": improvement, "beats_random": beats_random,
                     "own_goal_reduction": own_goal_reduction, "regression_vs_M2": regression},
        "verdict": verdict,
        "continuation_gate": "continue" if (beats_random or (improvement and own_goal_reduction)) else "stop and investigate",
        "promotion": "best_verified" if beats_random else ("candidate" if improvement else "none"),
    }
    diag = {}
    for n in ("diag-mid-vs-R0", "diag-mid-vs-M2REF", "diag-final-vs-R0", "diag-final-vs-M2REF"):
        if (OUT / f"{n}.json").is_file():
            diag[n] = side_stats(load(n))
    result["diagnostics_40_pairs"] = diag
    (OUT / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("midpoint", "all"):
        midpoint()
    if what in ("final", "all"):
        final()
    if what in ("decide", "all"):
        decide()
