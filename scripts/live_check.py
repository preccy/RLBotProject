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


class EventScript:
    """Staged events via RLBot state setting, each started once play is Active:
    1. demolition: blue drives at 2300 uu/s into a stationary orange car (ball moved away);
    2. goal into the blue net, 3. after the kickoff reset, goal into the orange net.
    Everything after each setup is ordinary play by the bots; the live report checks what happened."""

    def __init__(self, manager, flat):
        self.m, self.flat = manager, flat
        self.stage = 0
        self.log = []
        self.next_at = 8.0  # seconds of active play before the first event
        self.done = False

    def _v(self, x=None, y=None, z=None):
        return self.flat.Vector3Partial(x=x, y=y, z=z)

    def _car(self, x, y, yaw, vy):
        f = self.flat
        return f.DesiredCarState(physics=f.DesiredPhysics(location=self._v(x, y, 17.0),
                                                          rotation=f.RotatorPartial(pitch=0.0, yaw=yaw, roll=0.0),
                                                          velocity=self._v(0.0, vy, 0.0),
                                                          angular_velocity=self._v(0.0, 0.0, 0.0)),
                                 boost_amount=50.0)

    def _ball(self, x, y, z, vy):
        f = self.flat
        return f.DesiredBallState(physics=f.DesiredPhysics(location=self._v(x, y, z), velocity=self._v(0.0, vy, 0.0),
                                                           angular_velocity=self._v(0.0, 0.0, 0.0)))

    def step(self, packet, active_seconds: float) -> None:
        if self.done or active_seconds < self.next_at:
            return
        idx = {p.team: i for i, p in enumerate(packet.players)}
        if self.stage == 0:
            self.m.set_game_state(balls={0: self._ball(3000.0, 3500.0, 93.0, 0.0)},
                                  cars={idx[0]: self._car(0.0, -1400.0, 1.5708, 2300.0),
                                        idx[1]: self._car(0.0, -700.0, 0.0, 0.0)})
            self.log.append({"event": "demolition setup", "frame": packet.match_info.frame_num})
            self.next_at = active_seconds + 6.0
        elif self.stage == 1:
            self.m.set_game_state(balls={0: self._ball(0.0, -4600.0, 300.0, -2500.0)})
            self.log.append({"event": "goal into blue net", "frame": packet.match_info.frame_num})
            self.next_at = active_seconds + 10.0
        elif self.stage == 2:
            self.m.set_game_state(balls={0: self._ball(0.0, 4600.0, 300.0, 2500.0)})
            self.log.append({"event": "goal into orange net", "frame": packet.match_info.frame_num})
            self.done = True
        print(self.log[-1])
        self.stage += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=str(ROOT / "bot" / "match_live_selfplay.toml"))
    ap.add_argument("--max-minutes", type=float, default=10.0)
    ap.add_argument("--out")
    ap.add_argument("--allow-overtime", action="store_true",
                    help="keep playing overtime (unlimited in Rocket League); by default stop when regulation ends")
    ap.add_argument("--events", action="store_true",
                    help="stage a demolition and goals with state setting (needs bot/match_live_events.toml)")
    ap.add_argument("--max-play-seconds", type=float, default=None,
                    help="stop after this much Active play time (after the scripted events, if any)")
    ap.add_argument("--bundle", help="BotCore export bundle to play (sets BOTCORE_BUNDLE; default bot/model)")
    ap.add_argument("--save-replay", action="store_true",
                    help="ask Rocket League to save the match replay (Documents/My Games/Rocket League/TAGame/Demos)")
    args = ap.parse_args()
    if args.events and args.match == str(ROOT / "bot" / "match_live_selfplay.toml"):
        args.match = str(ROOT / "bot" / "match_live_events.toml")

    out = Path(args.out) if args.out else ROOT / "runs" / "live" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["BOTCORE_LIVE_DIR"] = str(out.resolve())  # inherited by RLBotServer and the bot processes
    if args.bundle:
        os.environ["BOTCORE_BUNDLE"] = str(Path(args.bundle).resolve())

    from rlbot import flat
    from rlbot.managers import MatchManager

    phases = []
    t0 = time.time()
    manager = MatchManager()
    try:
        print(f"starting match {args.match}; diagnostics -> {out}")
        from rlbot.config import load_match_config

        match_config = load_match_config(Path(args.match).resolve())
        match_config.auto_save_replay = args.save_replay  # rlbot's toml loader does not read this field
        manager.start_match(match_config)
        print(f"first packet after {time.time() - t0:.0f}s")
        last_phase = None
        deadline = t0 + args.max_minutes * 60
        script = EventScript(manager, flat) if args.events else None
        active_seconds = 0.0
        last_t = time.time()
        while time.time() < deadline:
            packet = manager.packet
            now = time.time()
            if packet is not None:
                phase = packet.match_info.match_phase
                if phase == flat.MatchPhase.Active:
                    active_seconds += now - last_t
                    if script is not None:
                        script.step(packet, active_seconds)
                if (args.max_play_seconds and active_seconds >= args.max_play_seconds
                        and (script is None or script.done)):
                    print(f"stopping after {active_seconds:.0f}s of active play")
                    break
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
            last_t = now
            time.sleep(0.05)
        final = manager.packet
        result = {"phases": phases, "timed_out": time.time() >= deadline,
                  "final_score": None if final is None else [t.score for t in final.teams],
                  "wall_seconds": time.time() - t0, "match_file": str(Path(args.match).resolve()),
                  "scripted_events": None if script is None else script.log}
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
