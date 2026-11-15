"""Run experiment M2 exactly as pre-registered in docs/EXPERIMENT_M2.md. Idempotent: finished runs and existing
evaluation reports are skipped, so it can be restarted after an interruption.

    .\\.venv\\Scripts\\python.exe scripts\\run_m2.py [train|evaluate|summary|all]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CELLS = {"A": (1e-4, 10.0), "B": (3e-4, 10.0), "C": (1e-4, 3.0), "D": (3e-4, 3.0)}
SEEDS = (101, 202)
ITERATIONS = 160
PAIRS = 40
R0 = ROOT / "runs" / "reference" / "R0-random-init"
E60 = ROOT / "runs" / "reference" / "E60-probe-4.4M"
OUT = ROOT / "runs" / "m2"
HEAD_TO_HEAD = {  # factor -> list of (better-level cell, other cell)
    "learning_rate_3e-4_vs_1e-4": [("D", "C"), ("B", "A")],
    "goal_3_vs_10": [("D", "B"), ("C", "A")],
}


def run_name(cell: str, seed: int) -> str:
    return f"m2-{cell}-s{seed}"


def final_checkpoint(cell: str, seed: int) -> Path:
    return ROOT / "runs" / run_name(cell, seed) / "checkpoints" / f"iter_{ITERATIONS:06d}"


def train() -> None:
    for seed in SEEDS:
        for cell in CELLS:
            if final_checkpoint(cell, seed).is_dir():
                print(f"[skip] {run_name(cell, seed)} already finished")
                continue
            run_dir = ROOT / "runs" / run_name(cell, seed)
            if run_dir.exists():
                cmd = [sys.executable, "-m", "botcore", "resume", "--run", str(run_dir)]
            else:
                cmd = [sys.executable, "-m", "botcore", "train", "--config",
                       str(ROOT / "configs" / "experiments" / "m2" / f"{cell}.toml"),
                       "--name", run_name(cell, seed), "--seed", str(seed)]
            t0 = time.time()
            print(f"[train] {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
            print(f"[train] {run_name(cell, seed)} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def bundle(cell: str, seed: int) -> Path:
    from botcore.export import bundle_from_checkpoint

    out = ROOT / "runs" / run_name(cell, seed) / "exports" / f"iter_{ITERATIONS:06d}"
    if not out.exists():
        bundle_from_checkpoint(final_checkpoint(cell, seed), out)
    return out


def _eval(name: str, a: Path, b: Path, pairs: int = PAIRS, seed_offset: int = 0) -> dict:
    from botcore.evaluate import MatchRules, Side, evaluate, format_summary

    path = OUT / "eval" / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate(Side(str(a)), Side(str(b)), MatchRules(), n_pairs=pairs, seed_offset=seed_offset)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] {name}\n{format_summary(report)}", flush=True)
    return report


def evaluate_all() -> None:
    for seed in SEEDS:
        for cell in CELLS:
            b = bundle(cell, seed)
            _eval(f"{run_name(cell, seed)}_vs_R0", b, R0)
            _eval(f"{run_name(cell, seed)}_vs_E60", b, E60)
        for factor, pairs in HEAD_TO_HEAD.items():
            for better, other in pairs:
                _eval(f"h2h-s{seed}-{better}_vs_{other}", bundle(better, seed), bundle(other, seed))


def _score(report: dict) -> dict:
    s = report["summary"]
    lo, hi = s["A_score_ci95_pair_bootstrap"]
    return {"score": round(s["A_score"], 3), "ci95": [round(lo, 3), round(hi, 3)],
            "W-D-L": f"{s['A_wins']}-{s['draws']}-{s['A_losses']}", "goals": f"{s['A_goals_for']}-{s['A_goals_against']}",
            "behaviour_A": {k: round(v, 3) for k, v in s.get("behaviour", {}).get("A", {}).items()}}


def summary() -> dict:
    load = lambda name: json.loads((OUT / "eval" / f"{name}.json").read_text(encoding="utf-8"))
    table = {}
    for cell, (lr, goal) in CELLS.items():
        for seed in SEEDS:
            name = run_name(cell, seed)
            metrics = [json.loads(l) for l in open(ROOT / "runs" / name / "metrics.jsonl", encoding="utf-8")]
            last = metrics[-1]
            table[name] = {
                "lr": lr, "goal": goal,
                "vs_R0": _score(load(f"{name}_vs_R0")), "vs_E60": _score(load(f"{name}_vs_E60")),
                "beats_random": load(f"{name}_vs_R0")["summary"]["A_score_ci95_pair_bootstrap"][0] > 0.5,
                "train_final": {"entropy": round(last["ppo"]["entropy"], 3), "kl": round(last["ppo"]["approx_kl"], 5),
                                "goal_fraction": round(last["episodes"]["goal_fraction"] or 0, 3),
                                "touch_raw_per_step": round(last["reward_per_agent_step"]["raw"]["ball_touch"], 5),
                                "v2ball_raw_per_step": round(last["reward_per_agent_step"]["raw"]["velocity_to_ball"], 5),
                                "wall_minutes": round(sum(m["timing_s"]["total"] for m in metrics) / 60, 1)},
            }
    factors = {}
    for factor, pairs in HEAD_TO_HEAD.items():
        results = []
        for seed in SEEDS:
            for better, other in pairs:
                rep = load(f"h2h-s{seed}-{better}_vs_{other}")
                s = rep["summary"]
                results.append({"comparison": f"{better} vs {other} (seed {seed})", "score": round(s["A_score"], 3),
                                "ci95": [round(x, 3) for x in s["A_score_ci95_pair_bootstrap"]]})
        favour = [r["score"] > 0.5 for r in results]
        against = [r["score"] < 0.5 for r in results]
        excl_for = sum(r["ci95"][0] > 0.5 for r in results)
        excl_against = sum(r["ci95"][1] < 0.5 for r in results)
        if all(favour) and excl_for >= 2:
            verdict = "established: first-named level is better"
        elif all(against) and excl_against >= 2:
            verdict = "established: second-named level is better"
        else:
            verdict = "not established"
        factors[factor] = {"head_to_heads": results, "verdict": verdict}
    best = max(table, key=lambda n: table[n]["vs_R0"]["score"])
    out = {"runs": table, "factors": factors, "highest_score_vs_R0": best}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("train", "all"):
        train()
    if what in ("evaluate", "all"):
        evaluate_all()
    if what in ("summary", "all"):
        summary()
