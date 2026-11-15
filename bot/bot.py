"""RLBot v5 entry point for BotCore.

RLBot starts this with the project's virtual environment (see bot.toml). It loads the export bundle in
./model (installed with `python -m botcore export --checkpoint <run or checkpoint> --to-bot`).

Environment variables (optional):
  BOTCORE_DEVICE=cpu|cuda:0     inference device (default cpu; see STATUS.md for measured latency)
  BOTCORE_DETERMINISTIC=1       argmax instead of sampling (default: sample, as in training)
  BOTCORE_BUNDLE=<dir>          use a different bundle directory
"""

import logging
import os
from pathlib import Path

from botcore.deploy.runner import BotCoreBot

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bundle = os.environ.get("BOTCORE_BUNDLE") or str(Path(__file__).resolve().parent / "model")
    BotCoreBot(
        bundle,
        device=os.environ.get("BOTCORE_DEVICE", "cpu"),
        deterministic=os.environ.get("BOTCORE_DETERMINISTIC", "0") == "1",
    ).run()
