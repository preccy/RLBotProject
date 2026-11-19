"""Fresh confirmation of CTRL-13300 (docs/EXPERIMENT_NECTO_MIX.md, "Follow-up"). Evaluation only.

    .\\.venv\\Scripts\\python.exe scripts\\confirm_13300.py

Idempotent: finished reports and Necto matches are reused. Our policies always sample their actions.
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

OUT = ROOT / "runs" / "necto-mix-eval" / "confirm-13300"
REF = ROOT / "runs" / "reference"
IL = REF / "S4I-iter12900"
R0, M2REF = REF / "R0-random-init", REF / "M2-D-s202-iter160"
CAND = ROOT / "runs" / "s5-ctrl" / "exports" / "iter_013300"
CAND_SHA, IL_SHA = "0e3fd5323e0a", "b1dbba3f4585"
H2H_OFFSET, H2H_PAIRS = 20_000, 100
NECTO_OFFSET, NECTO_PAIRS = 20_200, 40


def check_identity() -> None:
    from botcore.export import load_bundle

    for path, sha in ((CAND, CAND_SHA), (IL, IL_SHA)):
        got = load_bundle(path).manifest["weights_sha256"][:12]
        if got != sha:
            raise SystemExit(f"identity check failed for {path}: {got} != {sha}")


def run_eval(name, a, b, pairs, offset):
    from botcore.evaluate import MatchRules, Side, evaluate, format_summary

    path = OUT / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    rep = evaluate(Side(str(a)), Side(str(b)), MatchRules(), n_pairs=pairs, seed_offset=offset)
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[{name}]\n{format_summary(rep)}", flush=True)
    return rep


def necto() -> dict:
    from necto_bench import play

    from botcore.evaluate import bootstrap_ci

    pols = {"CTRL-13300": CAND, "IL-12900": IL}
    mdir = OUT / "necto-matches"
    mdir.mkdir(parents=True, exist_ok=True)
    seeds = [NECTO_OFFSET + i for i in range(NECTO_PAIRS)]
    jobs, files = [], []
    for name, b in pols.items():
        for s in seeds:
            for blue in (True, False):
                f = mdir / f"{name}-{s}-{'blue' if blue else 'orange'}.json"
                files.append((name, f))
                if not f.is_file():
                    jobs.append(((str(b), blue, s, None), f))
    print(f"{len(jobs)} Necto matches to play", flush=True)
    with ProcessPoolExecutor(11) as ex:
        for (_, f), r in zip(jobs, ex.map(play, [a for a, _ in jobs])):
            f.write_text(json.dumps(r), encoding="utf-8")
    rows = {n: [json.loads(f.read_text(encoding="utf-8")) for m, f in files if m == n] for n in pols}
    score = lambda r: 1.0 if r["goals_for"] > r["goals_against"] else 0.5 if r["goals_for"] == r["goals_against"] else 0.0
    keys = (("margin", lambda r: r["goals_for"] - r["goals_against"]), ("gf", lambda r: r["goals_for"]),
            ("ga", lambda r: r["goals_against"]))
    pp, out = {}, {"opponent": "Necto V1 (argmax, as shipped)", "seeds": f"{seeds[0]}-{seeds[-1]} (+EVAL_SEED_BASE)",
                   "our_inference": "sampled"}
    for n, rs in rows.items():
        pp[n] = {k: np.array([np.mean([f(r) for r in rs if r["seed"] == s]) for s in seeds]) for k, f in keys}
        out[n] = {"matches": len(rs),
                  "W-D-L": f"{sum(score(r) == 1 for r in rs)}-{sum(score(r) == .5 for r in rs)}-"
                           f"{sum(score(r) == 0 for r in rs)}",
                  "goals_for": int(sum(r["goals_for"] for r in rs)),
                  "goals_against": int(sum(r["goals_against"] for r in rs)),
                  "goal_margin_per_match": float(pp[n]["margin"].mean()),
                  "goal_margin_ci95": bootstrap_ci(pp[n]["margin"], seed=1),
                  "behaviour_ours": {k: float(np.mean([r["behaviour"]["ours"][k] for r in rs]))
                                     for k in rs[0]["behaviour"]["ours"]}}
    diff = {}
    for k, label in (("margin", "goal_margin"), ("gf", "goals_scored"), ("ga", "goals_conceded")):
        d = pp["CTRL-13300"][k] - pp["IL-12900"][k]
        diff[label] = {"diff_per_match": float(d.mean()), "ci95": bootstrap_ci(d, seed=11)}
    out["paired_13300_minus_12900"] = diff
    out["clear_regression_vs_necto"] = diff["goal_margin"]["ci95"][1] < 0
    (OUT / "necto.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return out


def main() -> None:
    from botcore.evaluate import check_promotion
    from botcore.export import load_bundle

    OUT.mkdir(parents=True, exist_ok=True)
    check_identity()
    h2h = run_eval("h2h-CTRL13300-vs-IL12900", CAND, IL, H2H_PAIRS, H2H_OFFSET)
    nec = necto()
    s = h2h["summary"]
    lo = s["A_score_ci95_pair_bootstrap"][0]
    problems = check_promotion(load_bundle(CAND).manifest["weights_sha256"], load_bundle(IL).manifest["weights_sha256"],
                               [h2h])
    out = {"h2h": {"pairs": s["seed_pairs"], "W-D-L": f"{s['A_wins']}-{s['draws']}-{s['A_losses']}",
                   "score": s["A_score"], "ci95": s["A_score_ci95_pair_bootstrap"],
                   "goals": f"{s['A_goals_for']}-{s['A_goals_against']}",
                   "goal_diff_per_match": s["A_goal_diff_per_match"],
                   "goal_diff_ci95": s["A_goal_diff_ci95_pair_bootstrap"]},
           "check_promotion_problems": problems, "qualifies_h2h": lo > 0.5 and not problems,
           "against_necto": nec["paired_13300_minus_12900"],
           "clear_regression_vs_necto": nec["clear_regression_vs_necto"]}
    if out["qualifies_h2h"]:
        reg = {k: run_eval(f"CTRL13300-vs-{k}", CAND, p, 40, 0)["summary"]["A_score"] for k, p in
               (("R0", R0), ("M2REF", M2REF))}
        out["regression_scores"] = reg
        out["regression_checks_pass"] = all(v >= 0.45 for v in reg.values())
    ok = out["qualifies_h2h"] and out.get("regression_checks_pass", False)
    if not ok:
        out["outcome"] = "IL-12900 stays best_verified and deployed (13300 does not qualify)"
    elif out["clear_regression_vs_necto"]:
        out["outcome"] = "13300 qualifies head-to-head but regresses clearly vs Necto: IL-12900 stays deployed"
    else:
        out["outcome"] = "13300 qualifies: All-Star series required before install"
    (OUT / "decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
