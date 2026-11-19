"""Live benchmark: recorded offline 1v1 matches, BotCore vs Beast from the East (official RLBot v5 botpack).

    .\\.venv\\Scripts\\python.exe scripts\\beast_series.py --label ctrl-13300 --bundle bot\\model [--only 1]
    .\\.venv\\Scripts\\python.exe scripts\\beast_series.py --label ctrl-13300 --summarise-only

Fixed set: 6 standard 5-minute matches on Stadium_P, alternating colours (BotCore blue, orange, ...). BotCore
samples its actions as deployed, with CPU inference. Each match runs through scripts/live_check.py (packet
recording, diagnostics and live report). A match tied at the end of regulation is stopped rather than playing
unlimited overtime, and counted as a draw. Beast is launched from the botpack the RLBot GUI downloaded to
%LOCALAPPDATA%/RLBot5/bots (bot/match_vs_beast.toml, bot/match_beast_orange.toml).

This is a live benchmark against one community bot, not a rank estimate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BEAST_TOML = Path.home() / "AppData/Local/RLBot5/bots/bob_build_x86_64-windows/Beast/beast.bot.toml"
MATCHES = [("blue", "match_vs_beast.toml"), ("orange", "match_beast_orange.toml")] * 3


def summarise(series_dir: Path, label: str) -> dict:
    from allstar_series import analyse_match

    rows = []
    for d in sorted(series_dir.glob("m*-*")):
        if not (d / "match.json").is_file():
            continue
        r = analyse_match(d)
        if "error" not in r:  # the shared analyser names the opponent "allstar"; rename for this series
            r["score_botcore_beast"] = r.pop("score_botcore_allstar")
            r["beast"] = r.pop("allstar")
        rows.append(r)
    ok = [r for r in rows if "error" not in r]
    agg = {"label": label, "matches": len(ok),
           "W-D-L": f"{sum(r['result'] == 'W' for r in ok)}-{sum(r['result'] == 'D' for r in ok)}-"
                    f"{sum(r['result'] == 'L' for r in ok)}",
           "goals_botcore_beast": [sum(r["score_botcore_beast"][0] for r in ok),
                                   sum(r["score_botcore_beast"][1] for r in ok)]}
    for who in ("botcore", "beast"):
        for k in ("shots", "saves", "own_goals", "touches"):
            agg[f"{who}_{k}"] = sum(r[who][k] for r in ok)
    out = {"aggregate": agg, "matches": rows,
           "note": "Live benchmark vs Beast from the East (RLBot v5 botpack); 6 offline matches; not a rank estimate."}
    (series_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [f"# Beast series: {label}", "",
             "| # | BotCore colour | Score (BotCore-Beast) | Result | BotCore shots/saves/own goals | "
             "Beast shots/saves/own goals | BotCore touches/min | Health |", "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok, 1):
        b, s, h = r["botcore"], r["beast"], r["health"]
        note = " (stopped: tied at regulation)" if r["stopped_at_regulation_tie"] else ""
        lines.append(f"| {i} | {r['botcore_colour']} | {r['score_botcore_beast'][0]}-{r['score_botcore_beast'][1]} | "
                     f"{r['result']}{note} | {b['shots']}/{b['saves']}/{b['own_goals']} | "
                     f"{s['shots']}/{s['saves']}/{s['own_goals']} | {b['touches_per_active_min']} | "
                     f"fallbacks {h['fallback_ticks']}, p99 {h['callback_p99_ms'] or 0:.1f} ms, "
                     f"missed resets {h['missed_kickoff_resets']} |")
    lines += ["", f"Total: W-D-L {agg['W-D-L']}, goals {agg['goals_botcore_beast'][0]}-{agg['goals_botcore_beast'][1]}."]
    (series_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--bundle", help="export bundle BotCore plays with")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", help="match numbers (1-6) to (re)play; default all missing")
    args = ap.parse_args()
    series_dir = ROOT / "runs" / "beast" / args.label
    series_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarise_only:
        if not args.bundle:
            ap.error("--bundle is required to play")
        if not BEAST_TOML.is_file():
            ap.error(f"Beast not found at {BEAST_TOML}; download the botpack in the RLBot v5 GUI")
        from botcore.export import load_bundle

        bundle = load_bundle(Path(args.bundle))
        (series_dir / "bundle.json").write_text(json.dumps({"bundle": str(Path(args.bundle).resolve()),
                                                            "weights_sha256": bundle.manifest["weights_sha256"],
                                                            "opponent": str(BEAST_TOML)}, indent=2), encoding="utf-8")
        for n, (colour, match_file) in enumerate(MATCHES, 1):
            match_dir = series_dir / f"m{n}-{colour}"
            if (args.only and n not in args.only) or (not args.only and (match_dir / "match.json").is_file()):
                continue
            print(f"=== match {n}: BotCore {colour} vs Beast -> {match_dir}", flush=True)
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
