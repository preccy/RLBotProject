"""Replay recorder identification (docs/EXPERIMENT_HL_IMITATION.md, "Retained-side pilot").

The recorder is the player named in the replay title, "<date> <name> Ranked Duel [Win|Loss]". The title is a file
name, so characters that are not file-name safe are stripped from <name>. Names are therefore compared by their
ASCII letters and digits only, case-insensitively.

A match is ambiguous (skipped entirely) unless:
* the title parses;
* its name key is non-empty and equals the key of exactly one of the two players;
* the two players' keys differ;
* a result suffix, if present and starting with W or L, agrees with the final score from the recorder's side.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TITLE = re.compile(r"^\S+ (.+?) Ranked Duel(?: (\S+))?\s*$")


def key(name: str) -> str:
    return "".join(ch for ch in name if ch.isascii() and ch.isalnum()).casefold()


def metadata(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    m = z["metadata"]
    return m.item() if m.dtype == object else json.loads(str(m))


def identify(md: dict):
    """(recorder_is_orange or None, reason)."""
    g = md["game"]
    m = TITLE.match(g["replay_name"])
    if not m:
        return None, "title does not parse"
    k = key(m.group(1))
    keys = [key(p["name"]) for p in md["players"]]
    if not k:
        return None, "empty title name"
    if len(md["players"]) != 2 or keys[0] == keys[1]:
        return None, "players not distinguishable"
    hits = [p for p, pk in zip(md["players"], keys) if pk == k]
    if len(hits) != 1:
        return None, "title name matches no player"
    orange = bool(hits[0]["is_orange"])
    res = (m.group(2) or "").upper()
    if res[:1] in ("W", "L"):
        mine, theirs = (g["team_1_score"], g["team_0_score"]) if orange else (g["team_0_score"], g["team_1_score"])
        if (res[0] == "W") != (mine > theirs):
            return None, "result suffix contradicts score"
    return orange, "ok"


def identify_selection(root: Path):
    sel = json.loads((root / "external" / "hlrlrd" / "selection.json").read_text())
    out = []
    for m in sel["matches"]:
        orange, why = identify(metadata(root / "external" / "hlrlrd" / m["path"]))
        out.append({"index": m["index"], "split": m["split"], "path": m["path"], "recorder_orange": orange,
                    "retained_team": None if orange is None else int(not orange), "status": why})
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    rows = identify_selection(root)
    for split in ("train", "holdout"):
        r = [x for x in rows if x["split"] == split]
        amb = [(x["index"], x["status"]) for x in r if x["recorder_orange"] is None]
        print(f"{split}: {len(r)} matches, {len(r) - len(amb)} identified, ambiguous {amb}")
    train = [x for x in rows if x["split"] == "train"]
    print("training positions 7-11:", [(i, x["index"], x["recorder_orange"]) for i, x in enumerate(train) if 7 <= i < 12])
