"""Small opponent registry.

Each entry records how an opponent can be used:

* ``sim``            - an export bundle with the current obs/action contract; can drive cars in RocketSim
                       (training as a frozen opponent and headless evaluation).
* ``rlbot-only``     - a bot that exists only as an RLBot program (e.g. Necto/Nexto executables from the bot
                       pack, Psyonix bots). Usable in real-game checks only.
* ``not-integrated`` - known about but no working path yet.

Only ``sim`` entries are ever sampled for training or headless evaluation, and every ``sim`` entry is
contract-checked when it is loaded (botcore.export.load_bundle).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from botcore.checkpoint import atomic_write_text

COMPAT_VALUES = ("sim", "rlbot-only", "not-integrated")
REGISTRY_FILE = "registry.json"


@dataclass
class OpponentEntry:
    id: str
    compat: str
    path: Optional[str] = None  # bundle directory (sim) or bot.toml (rlbot-only), relative to the registry
    note: str = ""
    added_iteration: Optional[int] = None


class OpponentRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / REGISTRY_FILE
        self.entries: List[OpponentEntry] = []
        if self.file.is_file():
            data = json.loads(self.file.read_text(encoding="utf-8"))
            self.entries = [OpponentEntry(**e) for e in data.get("opponents", [])]

    def save(self) -> None:
        atomic_write_text(self.file, json.dumps({"opponents": [asdict(e) for e in self.entries]}, indent=2))

    def add(self, entry: OpponentEntry) -> None:
        if entry.compat not in COMPAT_VALUES:
            raise ValueError(f"compat must be one of {COMPAT_VALUES}")
        if any(e.id == entry.id for e in self.entries):
            raise ValueError(f"opponent id {entry.id!r} already registered")
        self.entries.append(entry)
        self.save()

    def sim_entries(self) -> List[OpponentEntry]:
        return [e for e in self.entries if e.compat == "sim"]

    def resolve(self, entry: OpponentEntry) -> Path:
        if entry.path is None:
            raise ValueError(f"opponent {entry.id} has no path")
        return (self.root / entry.path).resolve()

    def sample(self, rng: np.random.Generator) -> OpponentEntry:
        """Uniform over simulator-compatible snapshots (a tunable hypothesis, see docs/DESIGN.md)."""
        pool = self.sim_entries()
        if not pool:
            raise LookupError("no simulator-compatible opponents registered")
        return pool[int(rng.integers(len(pool)))]
