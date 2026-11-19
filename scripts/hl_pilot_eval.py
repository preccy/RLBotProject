"""Evaluations for the HL replay imitation pilot, as pre-registered in docs/EXPERIMENT_HL_IMITATION.md.

    .\\.venv\\Scripts\\python.exe scripts\\hl_pilot_eval.py [screen|confirm|h2h|necto|regression|heldout|decide|all]

Idempotent: finished reports and matches are reused. Our policies always sample their actions.
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

OUT = ROOT / "runs" / "hl-eval"
REF = ROOT / "runs" / "reference"
BASE = REF / "S5C-iter13300"
BASE_SHA = "0e3fd5323e0a"
R0, M2REF = REF / "R0-random-init", REF / "M2-D-s202-iter160"
RUNS = {"CTRL": ROOT / "runs" / "s6-ctrl", "HL": ROOT / "runs" / "s6-hl"}
SCREEN_ITERS = (13550, 13800, 14050, 14300)
SCREEN_OFFSET, SCREEN_PAIRS = 21_000, 40
CONFIRM_OFFSET, CONFIRM_PAIRS = 21_100, 100
H2H_OFFSET, H2H_PAIRS = 21_300, 100
NECTO_OFFSET, NECTO_PAIRS = 21_500, 40


def bundle(arm: str, it: int) -> Path:
    from botcore.export import bundle_from_checkpoint

    out = RUNS[arm] / "exports" / f"iter_{it:06d}"
    if not out.exists():
        bundle_from_checkpoint(RUNS[arm] / "checkpoints" / f"iter_{it:06d}", out)
    return out


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


def summary(rep: dict) -> dict:
    s = rep["summary"]
    return {"pairs": s["seed_pairs"], "W-D-L": f"{s['A_wins']}-{s['draws']}-{s['A_losses']}",
            "score": round(s["A_score"], 4), "ci95": [round(x, 4) for x in s["A_score_ci95_pair_bootstrap"]],
            "goals": f"{s['A_goals_for']}-{s['A_goals_against']}",
            "goal_diff_per_match": round(s["A_goal_diff_per_match"], 3),
            "goal_diff_ci95": [round(x, 3) for x in s["A_goal_diff_ci95_pair_bootstrap"]]}


def check_base() -> None:
    from botcore.export import load_bundle

    got = load_bundle(BASE).manifest["weights_sha256"][:12]
    if got != BASE_SHA:
        raise SystemExit(f"13300 reference identity check failed: {got}")


# ------------------------------------------------------------------ candidate selection
def screen() -> dict:
    out = {}
    for arm in RUNS:
        rows = []
        for it in SCREEN_ITERS:
            s = summary(run_eval(f"screen-{arm}-{it}-vs-13300", bundle(arm, it), BASE, SCREEN_PAIRS, SCREEN_OFFSET))
            rows.append({"iter": it, **s})
        best = max(rows, key=lambda r: (r["score"], r["goal_diff_per_match"], r["iter"]))
        out[arm] = {"rows": rows, "candidate": best["iter"]}
    (OUT / "screen.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


def cands() -> dict:
    sc = json.loads((OUT / "screen.json").read_text(encoding="utf-8"))
    return {arm: bundle(arm, sc[arm]["candidate"]) for arm in RUNS}


def confirm() -> dict:
    return {arm: summary(run_eval(f"confirm-{arm}-vs-13300", b, BASE, CONFIRM_PAIRS, CONFIRM_OFFSET))
            for arm, b in cands().items()}


def h2h() -> dict:
    c = cands()
    return summary(run_eval("h2h-HL-vs-CTRL", c["HL"], c["CTRL"], H2H_PAIRS, H2H_OFFSET))


def regression() -> dict:
    return {arm: {k: summary(run_eval(f"{arm}-vs-{k}", b, p, 40, 0)) for k, p in (("R0", R0), ("M2REF", M2REF))}
            for arm, b in cands().items()}


# ------------------------------------------------------------------ Necto
def necto() -> dict:
    from necto_bench import play

    from botcore.evaluate import bootstrap_ci

    pols = {**cands(), "13300": BASE}
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
                  "goal_margin_per_match": round(float(pp[n]["margin"].mean()), 3),
                  "goal_margin_ci95": bootstrap_ci(pp[n]["margin"], seed=1)}

    def paired(a, b):
        res = {}
        for k, label in (("margin", "goal_margin"), ("gf", "goals_scored"), ("ga", "goals_conceded")):
            d = pp[a][k] - pp[b][k]
            res[label] = {"diff_per_match": round(float(d.mean()), 3), "ci95": bootstrap_ci(d, seed=11)}
        return res

    out["paired_HL_minus_13300"] = paired("HL", "13300")
    out["paired_CTRL_minus_13300"] = paired("CTRL", "13300")
    out["paired_HL_minus_CTRL"] = paired("HL", "CTRL")
    (OUT / "necto.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


# ------------------------------------------------------------------ held-out imitation fit (descriptive)
def heldout() -> dict:
    from il_pilot_eval import heldout_metrics

    hold = ROOT / "runs" / "demos" / "hl1-holdout.npz"
    hm = sorted(set(np.load(hold)["match"].tolist()))
    kbm = ROOT / "runs" / "demos" / "s1.npz"
    out = {}
    for name, b in {"13300": BASE, **cands()}.items():
        out[name] = {"HL held-out (15 replays, retained side)": heldout_metrics(b, hold, hm),
                     "KBM m03 (held out)": heldout_metrics(b, kbm, [2])}
    (OUT / "heldout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


# ------------------------------------------------------------------ decision
def decide() -> dict:
    from botcore.evaluate import check_promotion
    from botcore.export import load_bundle

    sc = json.loads((OUT / "screen.json").read_text(encoding="utf-8"))
    c = cands()
    conf, reg, hh = confirm(), regression(), h2h()
    nec = json.loads((OUT / "necto.json").read_text(encoding="utf-8"))
    out = {"candidates": {a: sc[a]["candidate"] for a in RUNS}, "confirm_vs_13300": conf, "regression": reg,
           "h2h_HL_vs_CTRL": hh, "arms": {}}
    for arm, b in c.items():
        problems = check_promotion(load_bundle(b).manifest["weights_sha256"], load_bundle(BASE).manifest["weights_sha256"],
                                   [json.loads((OUT / f"confirm-{arm}-vs-13300.json").read_text(encoding="utf-8"))])
        q = (conf[arm]["ci95"][0] > 0.5 and not problems
             and all(reg[arm][k]["score"] >= 0.45 for k in ("R0", "M2REF")))
        flag = nec[f"paired_{arm}_minus_13300"]["goal_margin"]["ci95"][1] < 0
        out["arms"][arm] = {"qualifies": bool(q), "check_promotion_problems": problems, "necto_regression_flag": bool(flag)}
    qual = [a for a in RUNS if out["arms"][a]["qualifies"]]
    if len(qual) == 2:
        pick = "HL" if hh["ci95"][0] > 0.5 else "CTRL"
    else:
        pick = qual[0] if qual else None
    if pick is None:
        out["outcome"] = "no candidate qualifies: CTRL-13300 stays best_verified and deployed"
    elif out["arms"][pick]["necto_regression_flag"]:
        out["outcome"] = f"{pick} qualifies but is flagged for regression against Necto: not deployed; 13300 stays deployed"
    else:
        out["outcome"] = f"{pick} qualifies: becomes best_verified; All-Star series required before deployment"
    out["promoted"] = pick
    (OUT / "decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


if __name__ == "__main__":
    check_base()
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"screen": screen, "confirm": confirm, "h2h": h2h, "necto": necto, "regression": regression,
             "heldout": heldout, "decide": decide}
    for name in (steps if stage == "all" else [stage]):
        steps[name]()
