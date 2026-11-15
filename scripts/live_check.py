"""Run one offline RLBot v5 match with BotCore live diagnostics, then analyse it.

Usage (PowerShell, project root):
    .\\.venv\\Scripts\\python.exe scripts\\live_check.py [--match bot\\match_live_selfplay.toml] [--max-minutes 10]

What it does:
1. sets BOTCORE_LIVE_DIR so every BotCore instance writes runs\\live\\<timestamp>\\botcore-team<T>-index<I>.{log,bcp,summary.json};
2. starts RLBotServer (from %LOCALAPPDATA%\\RLBot5\\bin) and the match; RLBot launches Rocket League via Steam if needed;
3. waits for the match to end (or the time limit), stops the match and shuts RLBotServer down
   (Rocket League itself is left running);
4. runs `python -m botcore live-report` on the folder.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=str(ROOT / "bot" / "match_live_selfplay.toml"))
    ap.add_argument("--max-minutes", type=float, default=10.0)
    ap.add_argument("--out")
    ap.add_argument("--allow-overtime", action="store_true",
                    help="keep playing overtime (unlimited in Rocket League); by default stop when regulation ends")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "runs" / "live" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["BOTCORE_LIVE_DIR"] = str(out.resolve())  # inherited by RLBotServer and the bot processes

    from rlbot import flat
    from rlbot.managers import MatchManager

    phases = []
    t0 = time.time()
    manager = MatchManager()
    try:
        print(f"starting match {args.match}; diagnostics -> {out}")
        manager.start_match(Path(args.match).resolve())
        print(f"first packet after {time.time() - t0:.0f}s")
        last_phase = None
        deadline = t0 + args.max_minutes * 60
        while time.time() < deadline:
            packet = manager.packet
            if packet is not None:
                phase = packet.match_info.match_phase
                if phase != last_phase:
                    phases.append({"t": round(time.time() - t0, 1), "frame": packet.match_info.frame_num,
                                   "phase": str(phase), "score": [t.score for t in packet.teams]})
                    print(phases[-1])
                    last_phase = phase
                if phase == flat.MatchPhase.Ended:
                    break
                if packet.match_info.is_overtime and not args.allow_overtime:
                    print("regulation over and tied: stopping instead of playing unlimited overtime")
                    break
            time.sleep(0.2)
        final = manager.packet
        result = {"phases": phases, "timed_out": time.time() >= deadline,
                  "final_score": None if final is None else [t.score for t in final.teams],
                  "wall_seconds": time.time() - t0, "match_file": str(Path(args.match).resolve())}
    finally:
        try:
            manager.stop_match()
            time.sleep(5)  # let the bots retire and write their summaries
        except Exception as exc:  # noqa: BLE001
            print(f"stop_match failed: {exc}")
        try:
            manager.shut_down()
        except Exception as exc:  # noqa: BLE001
            print(f"shut_down failed: {exc}")
    (out / "match.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "phases"}, indent=2))
    return subprocess.call([sys.executable, "-m", "botcore", "live-report", "--dir", str(out)])


if __name__ == "__main__":
    sys.exit(main())
