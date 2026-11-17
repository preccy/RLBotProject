"""Evaluations for the imitation pilot, exactly as pre-registered in docs/EXPERIMENT_IL_PILOT.md.

    .\\.venv\\Scripts\\python.exe scripts\\il_pilot_eval.py [preflight|heldout|dodges|matches|decide|all]

Idempotent where it runs matches (existing reports are reused). Every policy samples its actions.
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

from long1_eval import bootstrap_mean_ci, side_stats  # noqa: E402

OUT = ROOT / "runs" / "il-eval"
REF = ROOT / "runs" / "reference"
START = REF / "S3C-iter11900"
START_CKPT = REF / "S3C-iter11900-checkpoint"
R0, M2REF = REF / "R0-random-init", REF / "M2-D-s202-iter160"
RUNS = {"CTRL": ROOT / "runs" / "s4-ctrl", "IL": ROOT / "runs" / "s4-il"}
FINAL = 12900
DEMOS = ROOT / "runs" / "demos" / "s1.npz"
PILOT0 = ROOT / "runs" / "demos" / "pilot0-argmax-match.npz"


def bundle(arm: str, it: int = FINAL) -> Path:
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


# ------------------------------------------------------------------ held-out human actions
def heldout_metrics(bundle_dir: Path, npz: Path, matches) -> dict:
    import torch

    from botcore.actions.table import get_action_table
    from botcore.deploy.live_report import feature_names
    from botcore.export import load_bundle
    from botcore.imitation import class_nll, load_demos

    N = feature_names()
    obs, labels, masks = load_demos(str(npz), matches, N.index("self.on_ground[0]"))
    T = get_action_table().controls
    d = np.load(npz)
    sel = np.isin(d["match"], list(matches)) & ~d["press_inflight"] & ~d["dodge_lost"]
    prev = d["prev"][sel]
    actor = load_bundle(bundle_dir, device="cpu").actor
    with torch.no_grad():
        lg = actor(torch.as_tensor(obs))
        nll = class_nll(lg, torch.as_tensor(masks)).numpy()
        p = torch.softmax(lg, -1).numpy()
    on_g = obs[:, N.index("self.on_ground[0]")] > 0.5
    jump, pj = T[labels, 5] > 0.5, T[prev, 5] > 0.5
    first, dodge = jump & ~pj & on_g, jump & ~pj & ~on_g
    jump_rows = T[:, 5] > 0.5
    pclass = np.exp(-nll)
    return {"blocks": int(len(labels)), "mean_nll": round(float(nll.mean()), 3),
            "argmax_in_class": round(float(masks[np.arange(len(p)), p.argmax(1)].mean()), 4),
            "first_jumps": int(first.sum()), "mean_P_class_first_jump": round(float(pclass[first].mean()), 4),
            "dodge_presses": int(dodge.sum()), "mean_P_class_dodge": round(float(pclass[dodge].mean()), 4),
            "mean_P_any_jump_row_at_dodge": round(float(p[dodge][:, jump_rows].sum(1).mean()), 4)}


def heldout() -> dict:
    out = {}
    pols = {"S3C-11900": START, "CTRL": bundle("CTRL"), "IL": bundle("IL")}
    for name, b in pols.items():
        out[name] = {"m03 (primary)": heldout_metrics(b, DEMOS, [2]),
                     "pilot0 (secondary)": heldout_metrics(b, PILOT0, [0]),
                     "m01-m02 (training, reference)": heldout_metrics(b, DEMOS, [0, 1])}
    (OUT / "heldout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ dodges in simulated matches
def _dodge_match(args) -> dict:
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
    run = {a_team: PolicyRunner(load_bundle(a_path, device="cpu"), seed=2 * s + (0 if a_blue else 1)),
           1 - a_team: PolicyRunner(load_bundle(b_path, device="cpu"), seed=2 * s + (1 if a_blue else 0))}
    hist = []  # per step for the candidate car: speed, v_to_ball, flip, touch, on_ground, up_z, has_jumped
    for _ in range(300 * 15):
        acts = {ag: np.array([run[env.state.cars[ag].team_num].act(obs[ag])]) for ag in env.agents}
        obs, _, term, _ = env.step(acts)
        st = env.state
        car = next(c for c in st.cars.values() if c.team_num == a_team)
        v = np.asarray(car.physics.linear_velocity, float)
        to_ball = np.asarray(st.ball.position, float) - np.asarray(car.physics.position, float)
        hist.append((np.linalg.norm(v), float(v @ to_ball / (np.linalg.norm(to_ball) + 1e-6)), bool(car.has_flipped),
                     car.ball_touches > 0, bool(car.on_ground), float(car.physics.up[2]), bool(car.has_jumped)))
        if any(term.values()):
            obs = env.reset()
            hist.append(None)  # reset marker
    env.close()
    dodges, jumps, recov = [], 0, []
    H = hist
    for k in range(1, len(H)):
        if H[k] is None or H[k - 1] is None:
            continue
        if H[k][6] and not H[k - 1][6]:
            jumps += 1
        if H[k][2] and not H[k - 1][2]:  # dodge started during step k
            w = [h for h in H[k:k + 16] if h is not None]  # ~1.07 s
            if len(w) < 8:
                continue
            dodges.append({"dspeed": w[7][0] - H[k - 1][0], "dv_ball": w[7][1] - H[k - 1][1],
                           "touch_1s": any(h[3] for h in w), "wheels_1s": any(h[4] and h[5] > 0.7 for h in w[1:])})
    k = 1
    while k < len(H):  # recoveries: landing after >= 0.5 s airborne (descriptive)
        if H[k] is not None and not H[k][4]:
            s0 = k
            while k < len(H) and H[k] is not None and not H[k][4]:
                k += 1
            if k < len(H) and H[k] is not None and k - s0 >= 8:
                w = [h for h in H[k:k + 15] if h is not None]
                recov.append(any(h[0] >= 1000 for h in w))
        k += 1
    return {"seed": seed, "a_blue": a_blue, "minutes": 5.0, "dodges": dodges, "jumps": jumps, "recoveries": recov}


def dodges() -> dict:
    path = OUT / "dodges.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    out = {}
    with ProcessPoolExecutor(12) as ex:
        for name, b in (("S3C-11900", START), ("CTRL", bundle("CTRL")), ("IL", bundle("IL"))):
            jobs = [(str(b), str(START), 17400 + i, blue) for i in range(20) for blue in (True, False)]
            rows = list(ex.map(_dodge_match, jobs))
            mins = sum(r["minutes"] for r in rows)
            dd = [d for r in rows for d in r["dodges"]]
            rec = [x for r in rows for x in r["recoveries"]]
            per_match = np.array([len(r["dodges"]) / r["minutes"] for r in rows])
            m, lo, hi = bootstrap_mean_ci(per_match)
            out[name] = {
                "matches": len(rows), "dodges": len(dd), "dodges_per_car_minute": round(m, 3),
                "dodges_per_car_minute_ci95": [round(lo, 3), round(hi, 3)],
                "matches_with_any_dodge": int(sum(1 for r in rows if r["dodges"])),
                "jumps_per_car_minute": round(sum(r["jumps"] for r in rows) / mins, 2),
                "dodge_mean_speed_change_0p5s": round(float(np.mean([d["dspeed"] for d in dd])), 1) if dd else None,
                "dodge_mean_v_to_ball_change_0p5s": round(float(np.mean([d["dv_ball"] for d in dd])), 1) if dd else None,
                "dodge_touch_within_1s": round(float(np.mean([d["touch_1s"] for d in dd])), 3) if dd else None,
                "dodge_wheels_within_1s": round(float(np.mean([d["wheels_1s"] for d in dd])), 3) if dd else None,
                "recoveries_descriptive": {"landings_after_0p5s_air": len(rec),
                                           "reach_1000uu_s_within_1s": round(float(np.mean(rec)), 3) if rec else None},
            }
            print(name, json.dumps(out[name]), flush=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ------------------------------------------------------------------ match strength
def matches() -> dict:
    il, ctrl = bundle("IL"), bundle("CTRL")
    reps = {"h2h": run_eval("h2h-IL-vs-CTRL", il, ctrl, 100, 17000)}
    for arm, b in (("CTRL", ctrl), ("IL", il)):
        reps[f"{arm}_vs_11900"] = run_eval(f"{arm}-vs-S3C11900", b, START, 100, 17200)
        reps[f"{arm}_vs_R0"] = run_eval(f"{arm}-vs-R0", b, R0, 40, 0)
        reps[f"{arm}_vs_M2REF"] = run_eval(f"{arm}-vs-M2REF", b, M2REF, 40, 0)
    return reps


def decide() -> dict:
    reps = matches()
    h = side_stats(reps["h2h"], "A")
    verdict = "IL better" if h["ci95"][0] > 0.5 else "IL worse" if h["ci95"][1] < 0.5 else "inconclusive"
    res = {"action_selection": "sampled for every policy", "h2h_IL_vs_CTRL": h, "h2h_verdict": verdict, "arms": {}}
    q = {}
    for arm in ("CTRL", "IL"):
        v = {k: side_stats(reps[f"{arm}_{k}"], "A") for k in ("vs_11900", "vs_R0", "vs_M2REF")}
        reg = v["vs_R0"]["score"] >= 0.45 and v["vs_M2REF"]["score"] >= 0.45
        q[arm] = v["vs_11900"]["ci95"][0] > 0.5 and reg
        res["arms"][arm] = {**v, "regression_checks_pass": reg, "qualifies_for_promotion": q[arm]}
    cand = ("IL" if verdict == "IL better" else "CTRL") if (q["CTRL"] and q["IL"]) else \
        ("CTRL" if q["CTRL"] else "IL" if q["IL"] else None)
    res["promotion_candidate"] = cand
    res["best_verified_after"] = f"{cand}-iter{FINAL}" if cand else "S3C-iter11900 (unchanged)"
    (OUT / "decision.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ preflight check
def preflight() -> dict:
    """Run on runs/il-preflight after its 20 iterations: training-set imitation loss before/after and logged stats."""
    import torch

    from botcore.deploy.live_report import feature_names
    from botcore.export import load_bundle
    from botcore.imitation import class_nll, load_demos

    run = ROOT / "runs" / "il-preflight"
    N = feature_names()
    obs, _, masks = load_demos(str(DEMOS), [0, 1], N.index("self.on_ground[0]"))
    idx = np.random.default_rng(0).choice(len(obs), 4096, replace=False)

    def loss(b):
        with torch.no_grad():
            return float(class_nll(load_bundle(b, device="cpu").actor(torch.as_tensor(obs[idx])),
                                   torch.as_tensor(masks[idx])).mean())

    from botcore.export import bundle_from_checkpoint
    last = sorted((run / "checkpoints").glob("iter_*"))[-1]
    b_after = run / "exports" / last.name
    if not b_after.exists():
        bundle_from_checkpoint(last, b_after)
    m = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    il = [x["imitation"] for x in m]
    res = {"iterations": len(m), "train_sample_loss_before": round(loss(START), 4),
           "train_sample_loss_after": round(loss(b_after), 4),
           "il_grad_over_ppo_grad_mean": round(float(np.mean([s["il_grad_norm"] / s["ppo_actor_grad_norm"] for s in il])), 3),
           "weighted_il_loss_first_last": [round(il[0]["weighted_loss"], 4), round(il[-1]["weighted_loss"], 4)],
           "cos_ppo_il_mean": round(float(np.mean([s["cos_ppo_il"] for s in il])), 3),
           "max_kl": round(max(x["ppo"]["approx_kl"] for x in m), 4),
           "max_clip": round(max(x["ppo"]["clip_fraction"] for x in m), 4),
           "all_finite": all(np.isfinite(list(s.values())).all() for s in il)}
    res["pass"] = (res["all_finite"] and res["train_sample_loss_after"] < res["train_sample_loss_before"]
                   and res["il_grad_over_ppo_grad_mean"] >= 0.10 and res["max_kl"] < 0.01 and res["max_clip"] < 0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "preflight":
        preflight()
    if what in ("heldout", "all"):
        heldout()
    if what in ("dodges", "all"):
        dodges()
    if what in ("matches", "decide", "all"):
        decide()
