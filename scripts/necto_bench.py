"""External benchmark (evaluation only): BotCore checkpoints vs Necto, the RLGym community bot (V1), in RocketSim.

    .\\.venv\\Scripts\\python.exe scripts\\necto_bench.py selftest   # adapter vs Necto's own obs/agent code
    .\\.venv\\Scripts\\python.exe scripts\\necto_bench.py smoke      # 1 colour-swapped pair + Necto mirror check
    .\\.venv\\Scripts\\python.exe scripts\\necto_bench.py run        # 20 pairs per checkpoint, same seeds
    .\\.venv\\Scripts\\python.exe scripts\\necto_bench.py report

Necto source: https://github.com/Rolv-Arild/Necto (CC BY-NC-SA 4.0), cloned to external/Necto; the shipped RLBot
model rlbot-support/Necto/necto-model.pt (TorchScript) is used unchanged. Necto is frozen and plays as shipped in
RLBot: argmax actions (beta = 1), no hard-coded kickoffs, one decision every 8 ticks.

Adapter (src/botcore/necto.py, shared with training; checked against Necto's code by `selftest`):
* Observation = Necto V1 (training/obs.py at a7462bb, identical layout to rlbot-support/Necto/necto_obs.py):
  ball + players (self first, then opponent) + 34 boost pads, 24 features each; orange is mirrored (x, y negated,
  team flags swapped); q = own row + previous 8-control action; kv positions/velocities relative to self.
  Boost and demo timers follow the training builder, updated once per 8-tick decision (the RLBot port rebuilds
  them every tick, which makes its demo timer cycle; training never saw that).
* State fields from rlgym 2: boost_amount / 100, on_ground, has_flip (same definition as rlgym v1), forward/up
  columns of the rotation matrix; boost pads re-ordered from RocketSim's arena order to Necto's list by position.
* Action = Necto's 5 heads -> 8 controls with Necto's own mapping (throttle = pitch, steer = yaw or roll when
  handbrake), applied through our RocketSimEngine with the same RLBot one-tick input delay as our policy.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NECTO_DIR = ROOT / "external" / "Necto" / "rlbot-support" / "Necto"
NECTO_MODEL = NECTO_DIR / "necto-model.pt"
OUT = ROOT / "runs" / "necto-bench"
REF = ROOT / "runs" / "reference"
CHECKPOINTS = {"S3C-11900": REF / "S3C-iter11900", "S4I-12900": REF / "S4I-iter12900"}
SEED_OFFSET, N_PAIRS = 18_000, 20
SMOKE_OFFSET = 18_900
TICK_SKIP = 8


from botcore.necto import (NectoObsV1, NectoPolicy, car_dict, necto_boost_locations,  # noqa: E402
                          necto_controls, pad_permutation)


# ------------------------------------------------------------------ environment with raw controls for Necto
def build_env_ext(seed: int):
    from rlgym.api import RLGym
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
    from rlgym.rocket_league.sim import RocketSimEngine

    from botcore.actions import LookupActionParser
    from botcore.obs.builder import PREV_ACTIONS_KEY
    from botcore.obs import SpecObsBuilder
    from botcore.rewards import LoggedReward
    from botcore.scenarios import ScenarioMutator

    class Parser(LookupActionParser):
        """Our table index for BotCore cars; an 8-control vector for the external car (repeated for 8 ticks)."""

        def parse_actions(self, actions, state, shared_info):
            prev = shared_info.setdefault(PREV_ACTIONS_KEY, {})
            ours = {a: v for a, v in actions.items() if np.asarray(v).size == 1}
            parsed = super().parse_actions(ours, state, shared_info)
            for a, v in actions.items():
                if a not in ours:
                    c = np.asarray(v, dtype=np.float32).reshape(8)
                    prev[a] = c
                    parsed[a] = np.repeat(c[None, :], self.action_repeat, axis=0)
            return parsed

    return RLGym(state_mutator=ScenarioMutator(1, {"kickoff": 1.0}, seed), obs_builder=SpecObsBuilder(),
                 action_parser=Parser(TICK_SKIP), reward_fn=LoggedReward({}, 0.0), termination_cond=GoalCondition(),
                 truncation_cond=NoTouchTimeoutCondition(timeout_seconds=1e9),
                 transition_engine=RocketSimEngine(rlbot_delay=True))


def play(args) -> dict:
    """One 300 s match (overtime to the next goal, capped at 120 s, as in botcore.evaluate). `ours` is a bundle path
    or "NECTO" (mirror check). Records goals, behaviour, kickoff first touches and per-goal context."""
    ours, ours_blue, seed, frames_out = args
    import torch

    from botcore import constants as C
    from botcore.config import EVAL_SEED_BASE
    from botcore.export import PolicyRunner, load_bundle

    torch.set_num_threads(1)
    s = EVAL_SEED_BASE + seed
    env = build_env_ext(s)
    obs = env.reset()
    perm = pad_permutation(env, necto_boost_locations())
    agents = list(env.agents)
    team_of = {a: env.state.cars[a].team_num for a in agents}
    our_team = C.BLUE_TEAM if ours_blue else C.ORANGE_TEAM
    pol = {}
    for a in agents:
        mine = team_of[a] == our_team
        if mine and ours != "NECTO":
            pol[a] = ("bc", PolicyRunner(load_bundle(ours, device="cpu"), device="cpu", deterministic=False,
                                         seed=2 * s + (0 if team_of[a] == C.BLUE_TEAM else 1)))
        else:
            pol[a] = ("necto", NectoPolicy(seed=2 * s + team_of[a]))
    for kind, p in pol.values():
        if kind == "necto":
            p.reset(len(agents))
    others = {a: [b for b in agents if b != a] for a in agents}

    goals = {0: 0, 1: 0}
    reg, ot_cap = 300 * 120, 120 * 120
    ticks, overtime = 0, False
    n_dec = {0: 0, 1: 0}
    acc = {t: dict(touches=0, speed=0.0, airborne=0, boost=0.0, ball_dist=0.0, dodges=0, jumps=0, supersonic=0)
           for t in (0, 1)}
    prev_flip = {a: False for a in agents}
    prev_jumped = {a: False for a in agents}
    kickoffs = []  # team that touched first after each kickoff (or -1 both / None no touch)
    k_first = None
    since_kickoff = 0
    last_touch = {0: None, 1: None}  # tick of each team's last touch
    touch_ctx = {}  # context at each team's last touch
    goal_log = []
    frames = []
    while True:
        st = env.state
        if frames_out:
            frames.append([round(ticks / 120, 2), [round(float(v)) for v in st.ball.position],
                           [[round(float(c.physics.position[0])), round(float(c.physics.position[1])),
                             round(float(c.physics.position[2])), round(float(c.physics.yaw), 3), int(c.team_num),
                             int(round(c.boost_amount)), int(bool(c.on_ground))] for c in st.cars.values()],
                           [goals[0], goals[1]]])
        actions = {}
        for a in agents:
            kind, p = pol[a]
            actions[a] = np.array([p.act(obs[a])]) if kind == "bc" else p.act(st, a, others[a], perm)
        obs, _, term, _ = env.step(actions)
        ticks += TICK_SKIP
        since_kickoff += TICK_SKIP
        st = env.state
        ball = np.asarray(st.ball.position, float)
        touched = []
        for a in agents:
            car, t = st.cars[a], team_of[a]
            d = acc[t]
            n_dec[t] += 1
            d["speed"] += float(np.linalg.norm(car.physics.linear_velocity))
            d["airborne"] += int(not car.on_ground)
            d["boost"] += float(car.boost_amount)
            d["ball_dist"] += float(np.linalg.norm(np.asarray(car.physics.position) - ball))
            d["supersonic"] += int(car.is_supersonic)
            if car.has_flipped and not prev_flip[a]:
                d["dodges"] += 1
            if car.has_jumped and not prev_jumped[a]:
                d["jumps"] += 1
            prev_flip[a], prev_jumped[a] = bool(car.has_flipped), bool(car.has_jumped)
            if car.ball_touches > 0:
                d["touches"] += 1
                touched.append(t)
                last_touch[t] = ticks
                opp = st.cars[others[a][0]]
                # defender goal-side: between the ball and its own goal in y (blue defends -y, orange +y)
                opp_y = float(np.asarray(opp.physics.position)[1])
                opp_goal_side = opp_y < ball[1] if opp.team_num == C.BLUE_TEAM else opp_y > ball[1]
                touch_ctx[t] = {"t_since_kickoff": since_kickoff / 120, "ball_z": float(ball[2]),
                                "toucher_airborne": bool(not car.on_ground), "toucher_flipping": bool(car.is_flipping),
                                "opp_dist": float(np.linalg.norm(np.asarray(opp.physics.position) - ball)),
                                "opp_goal_side": bool(opp_goal_side), "opp_airborne": bool(not opp.on_ground),
                                "opp_boost": float(opp.boost_amount),
                                "opp_speed": float(np.linalg.norm(opp.physics.linear_velocity)),
                                "opp_touched_last_2s": last_touch[1 - t] is not None and ticks - last_touch[1 - t] <= 240}
        if touched and k_first is None:
            k_first = touched[0] if len(set(touched)) == 1 else -1
        if any(term.values()):
            scorer = int(st.scoring_team)
            goals[scorer] += 1
            lt = max((t for t in (0, 1) if last_touch[t] is not None), key=lambda t: last_touch[t], default=None)
            ctx = dict(touch_ctx.get(scorer, {}))
            goal_log.append({"scorer": scorer, "t_since_kickoff": since_kickoff / 120, "last_toucher": lt,
                             "own_goal": lt is not None and lt != scorer, "kickoff_first_touch": k_first,
                             "scorer_last_touch": ctx})
            kickoffs.append(k_first)
            k_first, since_kickoff = None, 0
            last_touch = {0: None, 1: None}
            touch_ctx = {}
            if overtime:
                break
            obs = env.reset()
            for kind, p in pol.values():
                if kind == "necto":
                    p.reset(len(agents))
            prev_flip = {a: False for a in agents}
            prev_jumped = {a: False for a in agents}
        if not overtime and ticks >= reg:
            if goals[0] != goals[1]:
                break
            overtime = True
        if overtime and ticks >= reg + ot_cap:
            break
    if k_first is not None:
        kickoffs.append(k_first)
    env.close()
    minutes = ticks / 120 / 60
    beh = {}
    for t in (0, 1):
        d, n = acc[t], max(1, n_dec[t])
        beh[t] = {"touches_per_min": d["touches"] / minutes, "dodges_per_min": d["dodges"] / minutes,
                  "jumps_per_min": d["jumps"] / minutes, "mean_speed": d["speed"] / n, "airborne": d["airborne"] / n,
                  "mean_boost": d["boost"] / n, "ball_dist": d["ball_dist"] / n, "supersonic": d["supersonic"] / n}
    res = {"ours": str(ours), "ours_blue": ours_blue, "seed": seed, "our_team": our_team,
           "goals_for": goals[our_team], "goals_against": goals[1 - our_team], "seconds": ticks / 120,
           "overtime": overtime, "kickoffs": kickoffs, "goals": goal_log, "behaviour": {"ours": beh[our_team],
                                                                                       "necto": beh[1 - our_team]}}
    if frames_out:
        from botcore.viz import write_html

        write_html({"title": f"RocketSim: {Path(str(ours)).name} {'blue' if ours_blue else 'orange'} vs Necto, "
                             f"seed {seed}", "frames": frames}, Path(frames_out))
    return res


# ------------------------------------------------------------------ self-test against Necto's own code
def selftest() -> dict:
    """Compare the adapter with Necto's necto_obs.py / agent.py on real RocketSim states (both colours).
    Necto's modules are loaded from external/Necto by path (code read before use); their rlgym_compat imports are
    stubbed because only BLUE_TEAM/ORANGE_TEAM and type names are used."""
    from botcore import constants as C

    stubs = {"rlgym_compat": types.ModuleType("rlgym_compat"),
             "rlgym_compat.common_values": types.ModuleType("rlgym_compat.common_values"),
             "rlgym_compat.game_state": types.ModuleType("rlgym_compat.game_state")}
    stubs["rlgym_compat.common_values"].BLUE_TEAM, stubs["rlgym_compat.common_values"].ORANGE_TEAM = 0, 1
    stubs["rlgym_compat.game_state"].GameState = stubs["rlgym_compat.game_state"].PlayerData = object
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        mods = {}
        for name in ("necto_obs", "agent"):
            spec = importlib.util.spec_from_file_location(f"necto_ref_{name}", NECTO_DIR / f"{name}.py")
            mods[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mods[name])
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    class PO:
        def __init__(self, d):
            self.position, self.linear_velocity, self.angular_velocity = d["pos"], d["vel"], d["angvel"]
            self._f, self._u = d.get("fwd"), d.get("up")

        def forward(self):
            return self._f

        def up(self):
            return self._u

    class PD:
        def __init__(self, d, cid):
            self.car_data, self.team_num, self.car_id = PO(d), d["team"], cid
            self.boost_amount, self.on_ground, self.has_flip = d["boost"], d["on_ground"], d["has_flip"]

    class GS:
        pass

    from botcore.config import EVAL_SEED_BASE
    from botcore.export import PolicyRunner, load_bundle

    ref_agent = mods["agent"].Agent()
    pol = NectoPolicy()
    env = build_env_ext(EVAL_SEED_BASE + 18_950)
    obs = env.reset()
    perm = pad_permutation(env, necto_boost_locations())
    runner = PolicyRunner(load_bundle(CHECKPOINTS["S4I-12900"], device="cpu"), device="cpu", seed=0)
    agents = list(env.agents)
    rng = np.random.default_rng(0)
    max_obs_err, action_mismatch, n = 0.0, 0, 0
    for step in range(400):
        st = env.state
        if step % 10 == 0:  # compare on a fresh reference builder (first call: same timers as ours; demo col skipped)
            for a in agents:
                me, opp = a, [b for b in agents if b != a]
                players = [car_dict(st.cars[x]) for x in [me] + opp]
                ball = {"pos": np.asarray(st.ball.position), "vel": np.asarray(st.ball.linear_velocity),
                        "angvel": np.asarray(st.ball.angular_velocity)}
                active = (np.asarray(st.boost_pad_timers)[perm] == 0).astype(float)
                prev = rng.integers(-1, 2, 8).astype(float)
                gs = GS()
                gs.ball = PO(ball)
                gs.players = [PD(p, i) for i, p in enumerate(players)]
                gs.boost_pads = active
                ref = mods["necto_obs"].NectoObsBuilder()
                rq, rkv, rm = ref.build_obs(gs.players[0], gs, prev)
                mine = NectoObsV1(necto_boost_locations())
                mq, mkv, mm = mine.build(ball, players, active, prev)
                cols = [c for c in range(24) if c != 21]  # player demo timer: RLBot-port quirk, see docstring
                err = max(np.abs(rq[..., cols] - mq[..., cols]).max(), np.abs(rq[..., 24:] - mq[..., 24:]).max(), np.abs(rkv[..., cols] - mkv[..., cols]).max(),
                          np.abs(rkv[0, 3:, 21] - mkv[0, 3:, 21]).max(), np.abs(rm - mm).max())
                max_obs_err = max(max_obs_err, float(err))
                ref_ctrl, _ = ref_agent.act((mq, mkv, mm), 1)
                my_ctrl = necto_controls(pol.heads(mq, mkv, mm))
                action_mismatch += int(not np.allclose(ref_ctrl, my_ctrl))
                n += 1
        acts = {}
        for a in agents:
            acts[a] = (np.array([runner.act(obs[a])]) if st.cars[a].team_num == C.BLUE_TEAM
                       else pol.act(st, a, [b for b in agents if b != a], perm))
        obs, _, term, _ = env.step(acts)
        if any(term.values()):
            obs = env.reset()
            pol.reset(2)
    env.close()
    # control mapping: every head combination
    mapping_ok = True
    for h in np.ndindex(3, 3, 2, 2, 2):
        c = necto_controls(np.array(h))
        mapping_ok &= c[0] == c[2] and (c[3] == 0 or c[4] == 0) and c[7] == h[4]
    out = {"comparisons": n, "max_obs_abs_error_excl_player_demo_col": max_obs_err,
           "action_mismatches": action_mismatch, "mapping_ok": bool(mapping_ok),
           "pass": bool(max_obs_err < 1e-6 and action_mismatch == 0 and mapping_ok)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selftest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def mirror_check() -> dict:
    """Necto vs Necto, both argmax, from a symmetric kickoff: with correct colour handling the orange car's
    trajectory is the blue car's mirrored (x, y negated) until physics asymmetries or the first touch split them."""
    from botcore.config import EVAL_SEED_BASE

    env = build_env_ext(EVAL_SEED_BASE + SMOKE_OFFSET)
    env.reset()
    perm = pad_permutation(env, necto_boost_locations())
    agents = list(env.agents)
    pols = {a: NectoPolicy() for a in agents}
    blue = next(a for a in agents if env.state.cars[a].team_num == 0)
    orange = next(a for a in agents if a != blue)
    p0b, p0o = np.asarray(env.state.cars[blue].physics.position), np.asarray(env.state.cars[orange].physics.position)
    symmetric_spawn = bool(np.allclose(p0b[:2], -p0o[:2], atol=1))
    errs, first_touch = [], None
    for step in range(30):  # 2 s
        st = env.state
        acts = {a: pols[a].act(st, a, [b for b in agents if b != a], perm) for a in agents}
        if step < 10:
            errs.append(float(np.abs(acts[blue] - acts[orange]).max()))
        env.step(acts)
        st = env.state
        if first_touch is None and any(c.ball_touches > 0 for c in st.cars.values()):
            first_touch = step
        pb, po = np.asarray(st.cars[blue].physics.position), np.asarray(st.cars[orange].physics.position)
        if first_touch is None:
            errs.append(float(np.linalg.norm(pb[:2] + po[:2])))
    env.close()
    return {"symmetric_spawn": symmetric_spawn, "max_mirror_error_before_first_touch": max(errs),
            "first_touch_step": first_touch}


def smoke() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"mirror": mirror_check()}
    jobs = [(str(CHECKPOINTS["S4I-12900"]), True, SMOKE_OFFSET, str(OUT / "smoke-12900-blue.html")),
            (str(CHECKPOINTS["S4I-12900"]), False, SMOKE_OFFSET, str(OUT / "smoke-12900-orange.html"))]
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(play, jobs))
    out["matches"] = [{k: r[k] for k in ("ours_blue", "goals_for", "goals_against", "seconds", "behaviour")} |
                      {"kickoffs_ours_first": sum(k == r["our_team"] for k in r["kickoffs"]),
                       "kickoffs_necto_first": sum(k == 1 - r["our_team"] for k in r["kickoffs"]),
                       "kickoffs": len(r["kickoffs"])} for r in rows]
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(str(path), blue, SEED_OFFSET + i, None) for path in CHECKPOINTS.values()
            for i in range(N_PAIRS) for blue in (True, False)]
    todo = []
    for j in jobs:
        f = OUT / "matches" / f"{Path(j[0]).name}-{j[2]}-{'blue' if j[1] else 'orange'}.json"
        if not f.is_file():
            todo.append(j)
    (OUT / "matches").mkdir(exist_ok=True)
    print(f"{len(todo)} of {len(jobs)} matches to play", flush=True)
    with ProcessPoolExecutor(11) as ex:
        for j, r in zip(todo, ex.map(play, todo)):
            f = OUT / "matches" / f"{Path(j[0]).name}-{j[2]}-{'blue' if j[1] else 'orange'}.json"
            f.write_text(json.dumps(r), encoding="utf-8")
            print(f"{f.name}: {r['goals_for']}-{r['goals_against']}", flush=True)


def report() -> dict:
    from botcore.evaluate import bootstrap_ci

    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "matches").glob("*.json"))]
    seeds = [SEED_OFFSET + i for i in range(N_PAIRS)]
    out = {"opponent": "Necto V1 (RLBot model, argmax, no hard-coded kickoffs)", "seeds": f"{SEED_OFFSET}-"
           f"{SEED_OFFSET + N_PAIRS - 1} (+EVAL_SEED_BASE), colour-swapped", "our_inference": "sampled"}
    pair_scores = {}
    for name, path in CHECKPOINTS.items():
        rs = [r for r in rows if Path(r["ours"]).name == path.name]
        score = lambda r: 1.0 if r["goals_for"] > r["goals_against"] else 0.5 if r["goals_for"] == r["goals_against"] else 0.0
        ps = np.array([np.mean([score(r) for r in rs if r["seed"] == s]) for s in seeds])
        gd = np.array([np.mean([r["goals_for"] - r["goals_against"] for r in rs if r["seed"] == s]) for s in seeds])
        pair_scores[name] = (ps, gd)
        ko = [(k, r["our_team"]) for r in rs for k in r["kickoffs"]]
        ko_ours = sum(k == t for k, t in ko)
        ko_necto = sum(k == 1 - t for k, t in ko)
        conceded = [g for r in rs for g in r["goals"] if g["scorer"] != r["our_team"]]
        scored = [g for r in rs for g in r["goals"] if g["scorer"] == r["our_team"]]
        b = lambda who, k: float(np.mean([r["behaviour"][who][k] for r in rs]))
        touches_ours, touches_necto = b("ours", "touches_per_min"), b("necto", "touches_per_min")
        out[name] = {
            "matches": len(rs), "W-D-L": f"{sum(score(r) == 1 for r in rs)}-{sum(score(r) == .5 for r in rs)}-"
                                         f"{sum(score(r) == 0 for r in rs)}",
            "goals_for": int(sum(r["goals_for"] for r in rs)), "goals_against": int(sum(r["goals_against"] for r in rs)),
            "score": float(np.mean([score(r) for r in rs])), "score_ci95": bootstrap_ci(ps, unit_interval=True),
            "goal_diff_per_match": float(gd.mean()), "goal_diff_ci95": bootstrap_ci(gd, seed=1),
            "goal_margins_sorted": sorted(r["goals_for"] - r["goals_against"] for r in rs),
            "as_blue_score": float(np.mean([score(r) for r in rs if r["ours_blue"]])),
            "as_orange_score": float(np.mean([score(r) for r in rs if not r["ours_blue"]])),
            "kickoffs": len(ko), "kickoff_first_touch_ours": ko_ours, "kickoff_first_touch_necto": ko_necto,
            "kickoff_first_touch_share_ours": ko_ours / max(1, ko_ours + ko_necto),
            "touch_share_ours": touches_ours / (touches_ours + touches_necto),
            "behaviour_ours": {k: b("ours", k) for k in rows[0]["behaviour"]["ours"]},
            "behaviour_necto": {k: b("necto", k) for k in rows[0]["behaviour"]["necto"]},
            "conceded": goal_context(conceded), "scored": goal_context(scored),
            "own_goals_ours_per_match": float(np.mean([sum(g["own_goal"] and g["scorer"] != r["our_team"]
                                                           for g in r["goals"]) for r in rs])),
        }
        # per-pair kickoff share for the paired comparison
        out[name]["_ko_pairs"] = [
            (sum(k == r["our_team"] for r in rs if r["seed"] == s for k in r["kickoffs"]),
             sum(k == 1 - r["our_team"] for r in rs if r["seed"] == s for k in r["kickoffs"])) for s in seeds]
    a, b_ = "S4I-12900", "S3C-11900"
    d = pair_scores[a][0] - pair_scores[b_][0]
    dg = pair_scores[a][1] - pair_scores[b_][1]
    ka = np.array([x / max(1, x + y) for x, y in out[a].pop("_ko_pairs")])
    kb = np.array([x / max(1, x + y) for x, y in out[b_].pop("_ko_pairs")])
    out["paired_12900_minus_11900"] = {
        "score_diff": float(d.mean()), "score_diff_ci95": bootstrap_ci(d, seed=2),
        "goal_diff_diff": float(dg.mean()), "goal_diff_diff_ci95": bootstrap_ci(dg, seed=3),
        "kickoff_first_touch_share_diff": float((ka - kb).mean()),
        "kickoff_first_touch_share_diff_ci95": bootstrap_ci(ka - kb, seed=4)}
    (OUT / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def goal_context(goals: list) -> dict:
    """Descriptive context of goals at the scorer's last touch (heuristic categories, not a diagnosis)."""
    if not goals:
        return {"n": 0}
    ctx = [g["scorer_last_touch"] for g in goals if g["scorer_last_touch"]]
    t = np.array([g["t_since_kickoff"] for g in goals])
    f = lambda k: float(np.mean([c[k] for c in ctx])) if ctx else float("nan")
    return {"n": len(goals), "own_goals_by_other_side": int(sum(g["own_goal"] for g in goals)),
            "median_s_since_kickoff": float(np.median(t)), "within_5s_of_kickoff": float(np.mean(t <= 5)),
            "scorer_won_kickoff_first_touch": float(np.mean([g["kickoff_first_touch"] == g["scorer"] for g in goals])),
            "at_scoring_touch": {"ball_z_over_300": float(np.mean([c["ball_z"] > 300 for c in ctx])),
                                 "scorer_airborne": f("toucher_airborne"), "scorer_dodging": f("toucher_flipping"),
                                 "defender_goal_side": f("opp_goal_side"), "defender_airborne": f("opp_airborne"),
                                 "defender_mean_dist": f("opp_dist"), "defender_mean_boost": f("opp_boost"),
                                 "defender_mean_speed": f("opp_speed"),
                                 "defender_touched_within_2s_before": f("opp_touched_last_2s")}}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"selftest": selftest, "smoke": smoke, "run": run, "report": report}[mode]()
