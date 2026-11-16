"""External diagnostic: a fixed set of offline 1v1 matches, BotCore vs the Psyonix All-Star bot, in Rocket League.

    .\\.venv\\Scripts\\python.exe scripts\\allstar_series.py --label pre --bundle runs\\reference\\S1L-iter2263
    .\\.venv\\Scripts\\python.exe scripts\\allstar_series.py --label post --bundle <final export>
    .\\.venv\\Scripts\\python.exe scripts\\allstar_series.py --label pre --summarise-only

Fixed set: 4 standard 5-minute matches on Stadium_P, alternating colours (BotCore blue, orange, blue, orange),
BotCore sampling stochastically as in training, CPU inference. Each match runs through scripts/live_check.py
(diagnostics + live report); a match tied at the end of regulation is stopped instead of playing unlimited
overtime and counted as a draw. Rocket League saves each completed match's replay to
Documents/My Games/Rocket League/TAGame/Demos; an HTML top-down replay is written next to each match.

This is an external diagnostic, not a rank estimate: four matches against one fixed built-in bot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MATCHES = [("blue", "match_vs_allstar.toml"), ("orange", "match_allstar_orange.toml"),
           ("blue", "match_vs_allstar.toml"), ("orange", "match_allstar_orange.toml")]


def player_stats(packets, index: int) -> dict:
    from rlbot import flat

    active = [p for p in packets if p.match_info.match_phase == flat.MatchPhase.Active and len(p.players) > index]
    touches = set()
    for p in packets:
        if len(p.players) > index and p.players[index].latest_touch is not None:
            touches.add(round(p.players[index].latest_touch.game_seconds, 3))
    speed = [np.linalg.norm([p.players[index].physics.velocity.x, p.players[index].physics.velocity.y,
                             p.players[index].physics.velocity.z]) for p in active]
    final = [p for p in packets if len(p.players) > index][-1].players[index]
    si = final.score_info
    active_minutes = len(active) / 120.0 / 60.0
    return {
        "name": final.name, "team": final.team,
        "goals": si.goals, "own_goals": si.own_goals, "shots": si.shots, "saves": si.saves,
        "assists": si.assists, "demolitions": si.demolitions, "scoreboard_points": si.score,
        "touches": len(touches), "touches_per_active_min": round(len(touches) / active_minutes, 2) if active else None,
        "airborne_fraction": round(float(np.mean([p.players[index].air_state != flat.AirState.OnGround
                                                  for p in active])), 3) if active else None,
        "mean_speed": round(float(np.mean(speed)), 0) if speed else None,
        "supersonic_fraction": round(float(np.mean([p.players[index].is_supersonic for p in active])), 3)
        if active else None,
        "mean_boost": round(float(np.mean([p.players[index].boost for p in active])), 1) if active else None,
    }


def analyse_match(match_dir: Path) -> dict:
    from botcore.deploy.recording import load_recording

    match = json.loads((match_dir / "match.json").read_text(encoding="utf-8"))
    bcp = sorted(match_dir.glob("botcore-team*-index*.bcp"))
    if not bcp:
        return {"dir": str(match_dir), "error": "no BotCore recording"}
    rec = load_recording(bcp[0])
    team = int(rec.meta.get("team", 0))
    packets = [p for p in rec.packets if len(p.players) >= 2]
    last = packets[-1]
    bot_i = next(i for i, p in enumerate(last.players) if p.team == team)
    star_i = 1 - bot_i
    score = match["final_score"]
    mine, theirs = score[team], score[1 - team]
    ended = any("Ended" in ph["phase"] for ph in match["phases"])
    result = "W" if mine > theirs else "L" if mine < theirs else "D"
    report = json.loads((match_dir / "live_report.json").read_text(encoding="utf-8")) \
        if (match_dir / "live_report.json").is_file() else {}
    bot_report = next(iter(report.get("bots", {}).values()), {})
    summ = bot_report.get("summary", {})
    return {
        "dir": str(match_dir), "botcore_colour": "blue" if team == 0 else "orange",
        "score_botcore_allstar": [mine, theirs], "result": result,
        "ended_normally": ended, "stopped_at_regulation_tie": (not ended) and mine == theirs,
        "wall_seconds": round(match["wall_seconds"]),
        "botcore": player_stats(packets, bot_i), "allstar": player_stats(packets, star_i),
        "health": {k: summ.get(k) for k in ("fallback_ticks", "policy_faults", "env_resets", "unexpected_resets",
                                            "missed_kickoff_resets")}
        | {"callback_p99_ms": summ.get("latency_full_callback", {}).get("p99_ms"),
           "callbacks_over_6ms": summ.get("latency_full_callback", {}).get("missed_deadline")},
    }


def summarise(series_dir: Path, label: str) -> dict:
    rows = [analyse_match(d) for d in sorted(series_dir.glob("m*-*")) if (d / "match.json").is_file()]
    ok = [r for r in rows if "error" not in r]
    agg = {
        "label": label, "matches": len(ok),
        "W-D-L": f"{sum(r['result'] == 'W' for r in ok)}-{sum(r['result'] == 'D' for r in ok)}-"
                 f"{sum(r['result'] == 'L' for r in ok)}",
        "goals_botcore_allstar": [sum(r["score_botcore_allstar"][0] for r in ok),
                                  sum(r["score_botcore_allstar"][1] for r in ok)],
        "botcore_own_goals": sum(r["botcore"]["own_goals"] for r in ok),
        "allstar_own_goals": sum(r["allstar"]["own_goals"] for r in ok),
    }
    for who in ("botcore", "allstar"):
        for k in ("shots", "saves", "touches"):
            agg[f"{who}_{k}"] = sum(r[who][k] for r in ok)
        for k in ("touches_per_active_min", "airborne_fraction", "mean_speed", "supersonic_fraction", "mean_boost"):
            vals = [r[who][k] for r in ok if r[who][k] is not None]
            agg[f"{who}_{k}_mean"] = round(float(np.mean(vals)), 3) if vals else None
    out = {"aggregate": agg, "matches": rows,
           "note": "External diagnostic vs the built-in Psyonix All-Star bot; 4 matches; not a rank estimate."}
    (series_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [f"# All-Star series: {label}", "",
             "| # | BotCore colour | Score (BotCore-AllStar) | Result | BotCore shots/saves/own goals | "
             "All-Star shots/saves/own goals | BotCore touches/min | BotCore airborne | Health |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok, 1):
        b, s, h = r["botcore"], r["allstar"], r["health"]
        note = " (stopped: tied at regulation)" if r["stopped_at_regulation_tie"] else ""
        lines.append(f"| {i} | {r['botcore_colour']} | {r['score_botcore_allstar'][0]}-{r['score_botcore_allstar'][1]}"
                     f" | {r['result']}{note} | {b['shots']}/{b['saves']}/{b['own_goals']} | "
                     f"{s['shots']}/{s['saves']}/{s['own_goals']} | {b['touches_per_active_min']} | "
                     f"{b['airborne_fraction']} | fallbacks {h['fallback_ticks']}, p99 {h['callback_p99_ms'] or 0:.1f} ms,"
                     f" missed resets {h['missed_kickoff_resets']} |")
    lines += ["", f"Total: W-D-L {agg['W-D-L']}, goals {agg['goals_botcore_allstar'][0]}-"
                  f"{agg['goals_botcore_allstar'][1]}, BotCore own goals {agg['botcore_own_goals']}."]
    (series_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--bundle", help="export bundle BotCore plays with")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", help="match numbers (1-4) to (re)play; default all missing")
    args = ap.parse_args()
    series_dir = ROOT / "runs" / "allstar" / args.label
    series_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarise_only:
        if not args.bundle:
            ap.error("--bundle is required to play")
        from botcore.export import load_bundle

        bundle = load_bundle(Path(args.bundle))
        (series_dir / "bundle.json").write_text(json.dumps({"bundle": str(Path(args.bundle).resolve()),
                                                            "weights_sha256": bundle.manifest["weights_sha256"]},
                                                           indent=2), encoding="utf-8")
        for n, (colour, match_file) in enumerate(MATCHES, 1):
            match_dir = series_dir / f"m{n}-{colour}"
            if (args.only and n not in args.only) or (not args.only and (match_dir / "match.json").is_file()):
                continue
            print(f"=== match {n}: BotCore {colour} vs All-Star -> {match_dir}", flush=True)
            subprocess.call([sys.executable, str(ROOT / "scripts" / "live_check.py"), "--match",
                             str(ROOT / "bot" / match_file), "--out", str(match_dir), "--bundle", args.bundle,
                             "--max-minutes", "12", "--save-replay"])
            bcp = sorted(match_dir.glob("botcore-team*-index*.bcp"))
            if bcp:
                subprocess.call([sys.executable, "-m", "botcore", "replay-html", "--recording", str(bcp[0]),
                                 "--out", str(series_dir / f"m{n}-{colour}.html")])
            time.sleep(5)
    summarise(series_dir, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
