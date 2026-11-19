"""Evaluations for the Necto-mix experiment, exactly as pre-registered in docs/EXPERIMENT_NECTO_MIX.md.

    .\\.venv\\Scripts\\python.exe scripts\\necto_mix_eval.py [necto|transfer|decide|all]

Idempotent: finished matches/reports are reused. Our policies always sample their actions.
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

OUT = ROOT / "runs" / "necto-mix-eval"
REF = ROOT / "runs" / "reference"
START = REF / "S4I-iter12900"
R0, M2REF = REF / "R0-random-init", REF / "M2-D-s202-iter160"
RUNS = {"CTRL": ROOT / "runs" / "s5-ctrl", "NECTO": ROOT / "runs" / "s5-necto"}
FINAL = 13900
NECTO_OFFSET, NECTO_PAIRS = 18_100, 40


def bundle(arm: str, it: int = FINAL) -> Path:
    from botcore.export import bundle_from_checkpoint

    out = RUNS[arm] / "exports" / f"iter_{it:06d}"
    if not out.exists():
        bundle_from_checkpoint(RUNS[arm] / "checkpoints" / f"iter_{it:06d}", out)
    return out


# ------------------------------------------------------------------ 1. against Necto
def necto() -> dict:
    from necto_bench import play

    from botcore.evaluate import bootstrap_ci

    pols = {"CTRL": bundle("CTRL"), "NECTO": bundle("NECTO"), "IL-12900 (reference)": START}
    mdir = OUT / "necto-matches"
    mdir.mkdir(parents=True, exist_ok=True)
    jobs, files = [], []
    for name, b in pols.items():
        for i in range(NECTO_PAIRS):
            for blue in (True, False):
                f = mdir / f"{name.split()[0]}-{NECTO_OFFSET + i}-{'blue' if blue else 'orange'}.json"
                files.append((name, f))
                if not f.is_file():
                    jobs.append(((str(b), blue, NECTO_OFFSET + i, None), f))
    print(f"{len(jobs)} Necto matches to play", flush=True)
    with ProcessPoolExecutor(11) as ex:
        for (args, f), r in zip(jobs, ex.map(play, [a for a, _ in jobs])):
            f.write_text(json.dumps(r), encoding="utf-8")
    seeds = [NECTO_OFFSET + i for i in range(NECTO_PAIRS)]
    rows = {name: [json.loads(f.read_text(encoding="utf-8")) for n, f in files if n == name] for name in pols}
    score = lambda r: 1.0 if r["goals_for"] > r["goals_against"] else 0.5 if r["goals_for"] == r["goals_against"] else 0.0
    per_pair = {}
    out = {"opponent": "Necto V1 (argmax, as shipped)", "seeds": f"{seeds[0]}-{seeds[-1]} (+EVAL_SEED_BASE)",
           "our_inference": "sampled"}
    for name, rs in rows.items():
        pp = {k: np.array([np.mean([f(r) for r in rs if r["seed"] == s]) for s in seeds]) for k, f in
              (("margin", lambda r: r["goals_for"] - r["goals_against"]), ("gf", lambda r: r["goals_for"]),
               ("ga", lambda r: r["goals_against"]), ("score", score))}
        per_pair[name] = pp
        out[name] = {"matches": len(rs),
                     "W-D-L": f"{sum(score(r) == 1 for r in rs)}-{sum(score(r) == .5 for r in rs)}-"
                              f"{sum(score(r) == 0 for r in rs)}",
                     "goals_for": int(sum(r["goals_for"] for r in rs)),
                     "goals_against": int(sum(r["goals_against"] for r in rs)),
                     "goal_margin_per_match": float(pp["margin"].mean()),
                     "goal_margin_ci95": bootstrap_ci(pp["margin"], seed=1),
                     "score": float(np.mean([score(r) for r in rs])),
                     "behaviour_ours": {k: float(np.mean([r["behaviour"]["ours"][k] for r in rs]))
                                        for k in rs[0]["behaviour"]["ours"]}}

    def paired(a, b):
        res = {}
        for k, label in (("margin", "goal_margin"), ("gf", "goals_scored"), ("ga", "goals_conceded")):
            d = per_pair[a][k] - per_pair[b][k]
            res[label] = {"diff_per_match": float(d.mean()), "ci95": bootstrap_ci(d, seed=11)}
        return res

    out["paired_NECTO_minus_CTRL"] = paired("NECTO", "CTRL")
    out["paired_NECTO_minus_IL12900"] = paired("NECTO", "IL-12900 (reference)")
    out["paired_CTRL_minus_IL12900"] = paired("CTRL", "IL-12900 (reference)")
    lo, hi = out["paired_NECTO_minus_CTRL"]["goal_margin"]["ci95"]
    out["verdict_against_necto"] = "helped" if lo > 0 else "hurt" if hi < 0 else "inconclusive"
    (OUT / "necto.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return out


# ------------------------------------------------------------------ 2. transfer + promotion checks
def run_eval(name, a, b, pairs, seed_offset):
    from botcore.evaluate import MatchRules, Side, evaluate, format_summary

    path = OUT / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rep = evaluate(Side(str(a)), Side(str(b)), MatchRules(), n_pairs=pairs, seed_offset=seed_offset)
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[{name}]\n{format_summary(rep)}", flush=True)
    return rep


def transfer() -> dict:
    reps = {"h2h-NECTO-vs-CTRL": run_eval("h2h-NECTO-vs-CTRL", bundle("NECTO"), bundle("CTRL"), 100, 19_000)}
    for arm in RUNS:
        reps[f"{arm}-vs-IL12900"] = run_eval(f"{arm}-vs-IL12900", bundle(arm), START, 100, 19_200)
        reps[f"{arm}-vs-R0"] = run_eval(f"{arm}-vs-R0", bundle(arm), R0, 40, 0)
        reps[f"{arm}-vs-M2REF"] = run_eval(f"{arm}-vs-M2REF", bundle(arm), M2REF, 40, 0)
    return reps


def summary(rep: dict) -> dict:
    s = rep["summary"]
    return {"pairs": s["seed_pairs"], "W-D-L": f"{s['A_wins']}-{s['draws']}-{s['A_losses']}",
            "score": round(s["A_score"], 3), "ci95": [round(x, 3) for x in s["A_score_ci95_pair_bootstrap"]],
            "goals": f"{s['A_goals_for']}-{s['A_goals_against']}",
            "goal_diff_per_match": round(s["A_goal_diff_per_match"], 2),
            "goal_diff_ci95": [round(x, 2) for x in s["A_goal_diff_ci95_pair_bootstrap"]],
            "own_goals_per_match_A": round(s["behaviour"]["A"]["own_goals_per_match"], 3),
            "touches_per_min_A": round(s["behaviour"]["A"]["touch_decisions_per_min"], 2)}


def decide() -> dict:
    from botcore.evaluate import check_promotion
    from botcore.export import load_bundle

    nec = json.loads((OUT / "necto.json").read_text(encoding="utf-8"))
    reps = transfer()
    h2h = summary(reps["h2h-NECTO-vs-CTRL"])
    lo, hi = h2h["ci95"]
    out = {"action_selection": "sampled for every BotCore policy",
           "against_necto": {k: nec[k] for k in ("paired_NECTO_minus_CTRL", "verdict_against_necto")},
           "h2h_NECTO_vs_CTRL": h2h,
           "transfer_verdict": "NECTO better" if lo > 0.5 else "NECTO worse" if hi < 0.5 else "inconclusive",
           "arms": {}}
    best_sha = load_bundle(START).manifest["weights_sha256"]
    for arm in RUNS:
        vs = {k: summary(reps[f"{arm}-vs-{k}"]) for k in ("IL12900", "R0", "M2REF")}
        sha = load_bundle(bundle(arm)).manifest["weights_sha256"]
        problems = check_promotion(sha, best_sha, [reps[f"{arm}-vs-IL12900"]])
        regression_ok = vs["R0"]["score"] >= 0.45 and vs["M2REF"]["score"] >= 0.45
        out["arms"][arm] = {**{f"vs_{k}": v for k, v in vs.items()}, "weights_sha256": sha[:12],
                            "check_promotion_problems": problems, "regression_checks_pass": regression_ok,
                            "qualifies_for_promotion": not problems and regression_ok}
    q = [a for a in RUNS if out["arms"][a]["qualifies_for_promotion"]]
    if len(q) == 2:
        cand = "NECTO" if out["transfer_verdict"] == "NECTO better" else "CTRL"
    else:
        cand = q[0] if q else None
    out["promotion_candidate"] = cand
    out["best_verified_after"] = f"{cand}-iter{FINAL}" if cand else "IL-iter12900 (unchanged)"
    (OUT / "decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("necto", "all"):
        necto()
    if mode in ("transfer", "all"):
        transfer()
    if mode in ("decide", "all"):
        decide()
