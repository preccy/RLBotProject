"""Evaluations and decision for long run 2, exactly as pre-registered in docs/EXPERIMENT_LONG2.md.

    .\\.venv\\Scripts\\python.exe scripts\\long2_eval.py [final|decide|all] [final_iteration]

Idempotent: existing reports are reused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from long1_eval import bootstrap_mean_ci, own_goals_by_pair, side_stats  # noqa: E402

RUN = ROOT / "runs" / "s1-long2"
OUT = RUN / "eval"
START = ROOT / "runs" / "reference" / "S1L-iter2263"  # also the current best_verified
NIGHT_START = ROOT / "runs" / "reference" / "S2L-iter3600"  # frozen start of the overnight continuation
R0 = ROOT / "runs" / "reference" / "R0-random-init"
M2REF = ROOT / "runs" / "reference" / "M2-D-s202-iter160"
FINAL_ITER = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] in ("final", "decide", "all") else 6113
FRESH = dict(pairs=150, seed_offset=4000)
DIAG = dict(pairs=40, seed_offset=0)


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


def final() -> None:
    b = bundle(FINAL_ITER)
    run_eval("decision-final-vs-START", b, START, **FRESH)
    run_eval("diag-final-vs-R0", b, R0, **DIAG)
    run_eval("diag-final-vs-M2REF", b, M2REF, **DIAG)


def behaviour(rep: dict, side: str) -> dict:
    keys = ("touch_decisions_per_min", "airborne_fraction", "mean_speed", "supersonic_fraction", "mean_boost",
            "mean_ball_distance", "own_goals_per_match")
    rows = [m["behaviour"]["blue" if m["blue"] == side else "orange"] for m in rep["matches"]]
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys if k in rows[0]}


def decide() -> dict:
    load = lambda n: json.loads((OUT / f"{n}.json").read_text(encoding="utf-8"))
    rep = load("decision-final-vs-START")
    f, s = side_stats(rep, "A"), side_stats(rep, "B")
    improvement = f["ci95"][0] > 0.5
    regression = f["ci95"][1] < 0.5
    diag = {n: side_stats(load(n)) for n in ("diag-final-vs-R0", "diag-final-vs-M2REF")}
    regression_checks_pass = all(d["score"] >= 0.45 for d in diag.values())
    a, b = own_goals_by_pair(rep, "A"), own_goals_by_pair(rep, "B")
    diff = np.array([a[k] - b[k] for k in sorted(a)])
    mean, lo, hi = bootstrap_mean_ci(diff)
    result = {
        "final_iteration": FINAL_ITER,
        "final_vs_START": f, "START_side": s,
        "own_goal_difference_final_minus_START_per_match": {"mean": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)],
                                                             "pairs": len(diff), "note": "descriptive only"},
        "behaviour_in_decision_matches": {"FINAL": behaviour(rep, "A"), "START": behaviour(rep, "B")},
        "regression_checks_40_pairs": diag,
        "criteria": {"improvement_continued": improvement, "regression": regression,
                     "regression_checks_pass": regression_checks_pass},
        "verdict": "improvement continued" if improvement else "regression" if regression else "inconclusive",
        "retain_as_best_verified": "FINAL" if (improvement and regression_checks_pass) else "START",
    }
    (OUT / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def diag(iteration: int, seed_offset: int = 0) -> dict:
    """Overnight diagnostic: checkpoint vs the frozen overnight start, 40 pairs. Not a decision input."""
    rep = run_eval(f"night-diag-iter{iteration}-vs-S2L3600-off{seed_offset}", bundle(iteration), NIGHT_START,
                   pairs=40, seed_offset=seed_offset)
    out = {"iteration": iteration, "seed_offset": seed_offset, "candidate": side_stats(rep, "A"),
           "start": side_stats(rep, "B"), "behaviour": {"candidate": behaviour(rep, "A"), "start": behaviour(rep, "B")}}
    print(json.dumps(out, indent=1))
    return out


def overnight(iteration: int) -> dict:
    """Overnight final evaluation (fresh seeds), as written in docs/EXPERIMENT_LONG2.md before the night started."""
    b = bundle(iteration)
    reps = {
        "vs_S2L3600": run_eval(f"night-final-iter{iteration}-vs-S2L3600", b, NIGHT_START, pairs=150, seed_offset=4000),
        "vs_best_S1L2263": run_eval(f"night-final-iter{iteration}-vs-S1L2263", b, START, pairs=150, seed_offset=5000),
        "diag_vs_R0": run_eval(f"night-final-iter{iteration}-vs-R0", b, R0, **DIAG),
        "diag_vs_M2REF": run_eval(f"night-final-iter{iteration}-vs-M2REF", b, M2REF, **DIAG),
    }
    out = {k: side_stats(r, "A") for k, r in reps.items()}
    for k in ("vs_S2L3600", "vs_best_S1L2263"):
        a, o = own_goals_by_pair(reps[k], "A"), own_goals_by_pair(reps[k], "B")
        diff = np.array([a[x] - o[x] for x in sorted(a)])
        mean, lo, hi = bootstrap_mean_ci(diff)
        out[k + "_own_goal_diff_per_match"] = {"mean": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        out[k + "_behaviour"] = {"candidate": behaviour(reps[k], "A"), "opponent": behaviour(reps[k], "B")}
    out["criteria"] = {
        "improved_vs_overnight_start": out["vs_S2L3600"]["ci95"][0] > 0.5,
        "improved_vs_best_verified": out["vs_best_S1L2263"]["ci95"][0] > 0.5,
        "regression_checks_pass": out["diag_vs_R0"]["score"] >= 0.45 and out["diag_vs_M2REF"]["score"] >= 0.45,
    }
    out["promote"] = out["criteria"]["improved_vs_best_verified"] and out["criteria"]["regression_checks_pass"]
    (OUT / f"night-final-iter{iteration}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "diag":
        diag(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        sys.exit(0)
    if what == "overnight":
        overnight(int(sys.argv[2]))
        sys.exit(0)
    if what in ("final", "all"):
        final()
    if what in ("decide", "all"):
        decide()
