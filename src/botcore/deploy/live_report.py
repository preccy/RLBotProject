"""Analyse a live RLBot session recorded with BOTCORE_LIVE_DIR (see scripts/live_check.py).

Everything here is measured from what the bot actually received and did in Rocket League:
packet cadence, full-callback latency, fallbacks, resets, logged warnings, kickoff behaviour, team-frame
orientation for each colour, the controller inputs the game reports, and the observation values the policy
saw, compared with a simulator reference sample of the same policy (a distribution-shift diagnostic, not a
parity proof).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from rlbot import flat

from botcore import constants as C
from botcore.actions import get_action_table
from botcore.deploy.recording import Recording, load_recording
from botcore.obs import LAYOUT
from botcore.obs import spec as S

_ACTIVE = {flat.MatchPhase.Kickoff, flat.MatchPhase.Active}


def _pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else None


def feature_names() -> List[str]:
    names = [""] * S.OBS_SIZE
    for block, feats, base in (("self", S.SELF_FEATURES, LAYOUT.self_slice.start),
                               ("ball", S.BALL_FEATURES, LAYOUT.ball_slice.start)):
        off = base
        for f in feats:
            for k in range(f.size):
                names[off + k] = f"{block}.{f.name}[{k}]"
            off += f.size
    for slot in range(S.N_CAR_SLOTS):
        off = LAYOUT.car_slot_slice(slot).start
        for f in S.CAR_FEATURES:
            for k in range(f.size):
                names[off + k] = f"car{slot}.{f.name}[{k}]"
            off += f.size
    for k in range(S.PAD_SIZE):
        names[LAYOUT.pads_slice.start + k] = f"pad[{k}]"
    return names


def sim_reference(bundle_dir: Path, n_steps: int = 4000, seed: int = 7) -> np.ndarray:
    """Observations the same policy produces in RocketSim (kickoff + open play, both colours)."""
    from botcore.config import Config
    from botcore.env import EnvSpec, build_env
    from botcore.export import PolicyRunner, load_bundle

    runner = PolicyRunner(load_bundle(bundle_dir), seed=seed)
    cfg = Config()
    cfg.env.scenario_weights = {"kickoff": 0.5, "open_play": 0.5}
    env = build_env(EnvSpec.from_config(cfg), seed)
    obs = env.reset()
    out = []
    for _ in range(n_steps):
        out.extend(obs.values())
        actions = {a: np.array([runner.act(o)]) for a, o in obs.items()}
        obs, _, term, trunc = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            obs = env.reset()
    return np.asarray(out, dtype=np.float32)


def _player(packet, player_id):
    for p in packet.players:
        if p.player_id == player_id:
            return p
    return None


def analyse_recording(rec: Recording, reference: np.ndarray | None) -> Dict:
    meta = rec.meta
    pid = meta.get("player_id")
    team = meta.get("team")
    out: Dict = {"meta": meta}

    # ---- packet cadence and phases
    frames = np.array([p.match_info.frame_num for p in rec.packets], dtype=np.int64)
    walls = np.array(rec.packet_wall)
    phases = [p.match_info.match_phase for p in rec.packets]
    deltas = np.diff(frames)
    active = np.array([ph in _ACTIVE for ph in phases])
    act_deltas = np.diff(frames)[active[1:] & active[:-1]] if len(frames) > 1 else np.array([])
    span_s = (walls[-1] - walls[0]) if len(walls) > 1 else 0.0
    out["packets"] = {
        "count": len(rec.packets),
        "wall_seconds": span_s,
        "packets_per_second": len(rec.packets) / span_s if span_s else None,
        "ticks_per_second": float((frames[-1] - frames[0]) / span_s) if span_s else None,
        "frame_delta_active_hist": {int(k): int(v) for k, v in sorted(Counter(act_deltas.tolist()).items())[:8]},
        "frames_going_backwards": int((deltas < 0).sum()),
    }
    transitions = []
    for i, ph in enumerate(phases):
        if i == 0 or ph != phases[i - 1]:
            transitions.append((int(frames[i]), str(ph)))
    out["phase_transitions"] = len(transitions)
    out["kickoffs"] = sum(1 for _, ph in transitions if ph.endswith("Kickoff"))
    out["goals_seen"] = sum(1 for _, ph in transitions if ph.endswith("GoalScored"))
    last = rec.packets[-1] if rec.packets else None
    out["final_score"] = None if last is None else [t.score for t in last.teams]

    # ---- callback timing (what rlgym-rlbot measures as its 6 ms window)
    if rec.timing:
        t = np.array([x[1] for x in rec.timing])
        backlog = np.array([x[2] for x in rec.timing])
        out["callback_ms"] = {"n": len(t), "p50": _pct(t, 50), "p99": _pct(t, 99), "p99.9": _pct(t, 99.9),
                              "max": float(t.max()), "over_6ms": int((t > 6).sum()),
                              "callbacks_with_backlog(>1 packet)": int((backlog > 1).sum()),
                              "fallback_ticks": int(sum(x[3] for x in rec.timing))}

    # ---- decisions: policy time, actions, observation values
    if rec.decision_obs:
        obs = np.stack(rec.decision_obs)
        ms = np.array(rec.decision_ms)
        out["decisions"] = {
            "count": len(obs), "policy_ms_p50": _pct(ms, 50), "policy_ms_p99": _pct(ms, 99),
            "policy_ms_max": float(ms.max()),
            "decisions_per_active_second": None,
            "non_finite_values": int((~np.isfinite(obs)).sum()),
            "obs_abs_max": float(np.abs(obs).max()),
            "top_actions": Counter(rec.decision_action).most_common(5),
        }
        if reference is not None and len(reference):
            names = feature_names()
            lo = np.quantile(reference, 0.001, axis=0)
            hi = np.quantile(reference, 0.999, axis=0)
            margin = 0.05 * np.maximum(hi - lo, 0.05)
            outside = ((obs < lo - margin) | (obs > hi + margin)).mean(axis=0)
            worst = np.argsort(-outside)[:8]
            out["obs_vs_sim_reference"] = {
                "reference_samples": int(len(reference)),
                "features_with_any_outside": int((outside > 0).sum()),
                "worst_features": [(names[i], round(float(outside[i]), 4), round(float(obs[:, i].min()), 3),
                                    round(float(obs[:, i].max()), 3), round(float(lo[i]), 3), round(float(hi[i]), 3))
                                   for i in worst if outside[i] > 0],
            }

        # ---- orientation: kickoff decisions (ball resting at centre)
        bp = obs[:, LAYOUT.ball_slice.start + S.BALL_IDX["pos"].start:LAYOUT.ball_slice.start + S.BALL_IDX["pos"].start + 3]
        bv = obs[:, LAYOUT.ball_slice.start + S.BALL_IDX["lin_vel"].start:LAYOUT.ball_slice.start + S.BALL_IDX["lin_vel"].start + 3]
        kick = (np.abs(bp[:, :2]).max(axis=1) < 1e-3) & (np.abs(bv).max(axis=1) < 1e-3)
        if kick.any():
            sp = obs[kick][:, LAYOUT.self_slice.start + S.SELF_IDX["pos"].start + 1] * S.POS_SCALE
            fwd = obs[kick][:, LAYOUT.self_slice.start + S.SELF_IDX["forward"].start:LAYOUT.self_slice.start + S.SELF_IDX["forward"].start + 2]
            rel = obs[kick][:, LAYOUT.ball_slice.start + S.BALL_IDX["rel_pos"].start:LAYOUT.ball_slice.start + S.BALL_IDX["rel_pos"].start + 2]
            facing = (fwd * rel).sum(1) / (np.linalg.norm(rel, axis=1) + 1e-9)
            out["orientation_at_kickoff"] = {
                "decisions": int(kick.sum()),
                "self_y_team_frame_max": float(sp.max()),
                "all_in_own_half_in_team_frame": bool((sp < 0).all()),
                "mean_cos_forward_to_ball": float(facing.mean()),
            }

    # ---- demolitions: from packets, and whether the policy's observation flagged them
    if pid is not None and rec.packets:
        demo_frames = set()
        demos = 0
        was = False
        for packet in rec.packets:
            me = _player(packet, pid)
            if me is None:
                continue
            now = me.demolished_timeout != -1
            if now:
                demo_frames.add(packet.match_info.frame_num)
            if now and not was:
                demos += 1
            was = now
        flagged = None
        if rec.decision_obs and demo_frames:
            idx = LAYOUT.self_slice.start + S.SELF_IDX["is_demoed"].start
            during = [o[idx] for o, t in zip(rec.decision_obs, rec.decision_tick) if t in demo_frames]
            flagged = {"decisions_while_demolished": len(during),
                       "fraction_with_is_demoed_flag": float(np.mean(during)) if during else None}
        out["demolitions_of_this_car"] = demos
        out["demolition_observation_check"] = flagged

    # ---- world-frame orientation and kickoff behaviour from packets
    if pid is not None and rec.packets:
        world_y = []
        kickoff_runs = []
        jump_held_first_tick = 0
        inputs = []
        table = get_action_table().controls
        for i, packet in enumerate(rec.packets):
            me = _player(packet, pid)
            if me is None:
                continue
            ph = packet.match_info.match_phase
            if ph == flat.MatchPhase.Kickoff and (i == 0 or rec.packets[i - 1].match_info.match_phase != flat.MatchPhase.Kickoff):
                world_y.append(me.physics.location.y)
                jump_held_first_tick += int(me.last_input.jump)
                start_frame = packet.match_info.frame_num
                ball0 = np.array([packet.balls[0].physics.location.x, packet.balls[0].physics.location.y])
                pos0 = np.array([me.physics.location.x, me.physics.location.y])
                d0 = np.linalg.norm(ball0 - pos0)
                speed_1s, d_1s = None, None
                for later in rec.packets[i:]:
                    if later.match_info.frame_num - start_frame >= 120:
                        m2 = _player(later, pid)
                        if m2 is not None:
                            v = m2.physics.velocity
                            speed_1s = float(np.hypot(v.x, v.y))
                            b = later.balls[0].physics.location
                            d_1s = float(np.hypot(b.x - m2.physics.location.x, b.y - m2.physics.location.y))
                        break
                kickoff_runs.append({"start_distance": float(d0), "speed_after_1s": speed_1s,
                                     "distance_after_1s": d_1s})
            if ph in _ACTIVE:
                li = me.last_input
                inputs.append([li.throttle, li.steer, li.pitch, li.yaw, li.roll, float(li.jump), float(li.boost),
                               float(li.handbrake)])
        if world_y:
            expected = "negative (blue half)" if team == C.BLUE_TEAM else "positive (orange half)"
            out["kickoff_world_y"] = {"values": [round(y) for y in world_y[:6]], "expected": expected,
                                      "consistent": bool(all((y < 0) == (team == C.BLUE_TEAM) for y in world_y))}
        out["kickoff_behaviour"] = {
            "kickoffs": len(kickoff_runs),
            "jump_held_on_first_kickoff_tick": jump_held_first_tick,
            "mean_speed_after_1s": float(np.mean([k["speed_after_1s"] for k in kickoff_runs if k["speed_after_1s"] is not None])) if kickoff_runs else None,
            "mean_distance_closed_in_1s": float(np.mean([k["start_distance"] - k["distance_after_1s"] for k in kickoff_runs if k["distance_after_1s"] is not None])) if kickoff_runs else None,
        }
        if inputs:
            arr = np.round(np.asarray(inputs, dtype=np.float32), 4)
            rows = {tuple(r) for r in table.tolist()}
            in_table = np.array([tuple(r) in rows for r in arr.tolist()])
            zero = (arr == 0).all(axis=1)
            out["applied_inputs"] = {"ticks": len(arr), "fraction_matching_action_table": float(in_table.mean()),
                                     "fraction_all_zero": float(zero.mean()),
                                     "examples_not_in_table": [list(map(float, r)) for r in arr[~in_table][:3]]}
    return out


def analyse_log(path: Path) -> Dict:
    counts: Counter = Counter()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\S+ \S+ (WARNING|ERROR|CRITICAL) (.+?): (.*)", line)
        if m:
            msg = re.sub(r"\d+(\.\d+)?", "#", m.group(3))[:90]
            counts[f"{m.group(1)}: {msg}"] += 1
    return dict(counts.most_common(12))


def live_report(directory: Path, bundle_dir: Path | None = None, log=print) -> Dict:
    directory = Path(directory)
    recordings = sorted(directory.glob("botcore-*.bcp"))
    if not recordings:
        raise FileNotFoundError(f"no botcore-*.bcp recordings in {directory}")
    reference = None
    if bundle_dir is not None:
        reference = sim_reference(Path(bundle_dir))
    report: Dict = {"directory": str(directory), "bots": {}}
    match_file = directory / "match.json"
    if match_file.is_file():
        report["match"] = json.loads(match_file.read_text(encoding="utf-8"))
    for path in recordings:
        stem = path.with_suffix("")
        entry = analyse_recording(load_recording(path), reference)
        summary = Path(f"{stem}.summary.json")
        entry["summary"] = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else "missing (bot did not retire cleanly)"
        logf = Path(f"{stem}.log")
        entry["log_warnings"] = analyse_log(logf) if logf.is_file() else "missing"
        report["bots"][path.stem] = entry
    out = directory / "live_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(json.dumps(report, indent=2, default=str))
    log(f"report written to {out}")
    return report
