"""Start an offline RLBot v5 match from a match.toml (alternative to the RLBot GUI).

Usage (PowerShell, project root, venv active):
    python scripts\\start_match.py bot\\match_vs_allstar.toml

Uses the RLBotServer installed by the RLBot v5 installer (%LOCALAPPDATA%\\RLBot5\\bin). RLBot launches Rocket
League itself if it is not running. NOT verified by the automated tests (needs the game).
"""

import sys
from pathlib import Path
from time import sleep

from rlbot import flat
from rlbot.managers import MatchManager


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    config = Path(sys.argv[1]).resolve()
    with MatchManager() as manager:
        manager.start_match(config)
        print("match started; Ctrl+C here stops waiting (the match keeps running in Rocket League)")
        try:
            while manager.packet is None or manager.packet.match_info.match_phase != flat.MatchPhase.Ended:
                sleep(1.0)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
