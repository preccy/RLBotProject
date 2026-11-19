"""Select and download a small, varied 1v1 subset of Rolv-Arild/hlrlrd-parsed (High-Level RL Replay Dataset, parsed
into rlgym-tools ReplayFrames). Only the selected files are downloaded; candidates are screened by reading just the
`metadata` member at the end of each .npz with an HTTP range request.

    .\\.venv\\Scripts\\python.exe scripts\\hl_replays\\select_matches.py [--n 100] [--holdout 15]

Selection (fixed in docs/EXPERIMENT_HL_IMITATION.md):
* exactly 2 players, replay name contains "Ranked Duel", match_type Online;
* near-full length: num_frames >= 8,500 (a full 5-minute duel is ~9,000+ replay frames; forfeits are shorter);
* no player (platform online_id) in more than one selected match, so train and held-out matches share no players;
* at most 12 matches per map;
* candidates visited in a seeded random order over the whole dataset.
Held-out matches: a seeded random `--holdout` of the selected matches, assigned before any conversion.
Output: external/hlrlrd/selection.json and the files under external/hlrlrd/data/.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "external" / "hlrlrd"
REPO = "Rolv-Arild/hlrlrd-parsed"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
MIN_FRAMES = 8500
MAP_CAP = 12


def list_files() -> list[str]:
    cache = OUT / "file_list.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    dirs = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"))
    files = []
    for d in dirs:
        listing = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{d['path']}"))
        files += [x["path"] for x in listing if x["path"].endswith(".npz")]
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(files))
    return files


def read_metadata(path: str) -> dict | None:
    """Fetch the last 16 KiB of the .npz and decode the `metadata.npy` member (a unicode JSON string)."""
    req = urllib.request.Request(BASE + path, headers={"Range": "bytes=-16384"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Range"].split("/")[1])
        tail = r.read()
    base = total - len(tail)
    e = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[e + 12:e + 20])
    cd = tail[cd_off - base:cd_off - base + cd_size]
    i = 0
    while i < len(cd):
        method = struct.unpack("<H", cd[i + 10:i + 12])[0]
        csize = struct.unpack("<I", cd[i + 20:i + 24])[0]
        nlen, elen, clen = struct.unpack("<HHH", cd[i + 28:i + 34])
        off = struct.unpack("<I", cd[i + 42:i + 46])[0]
        name = cd[i + 46:i + 46 + nlen].decode()
        i += 46 + nlen + elen + clen
        if name != "metadata.npy" or off < base:
            continue
        lh = tail[off - base:]
        ln, le = struct.unpack("<HH", lh[26:30])
        raw = lh[30 + ln + le:30 + ln + le + csize]
        data = zlib.decompress(raw, -15) if method == 8 else raw
        hlen = struct.unpack("<H", data[8:10])[0]  # npy v1 header
        arr = np.frombuffer(data[10 + hlen:], dtype="<U" + str((len(data) - 10 - hlen) // 4))
        return json.loads(str(arr[0]))
    return None


def eligible(meta: dict) -> bool:
    g = meta.get("game", {})
    return (len(meta.get("players", [])) == 2 and "Ranked Duel" in (g.get("replay_name") or "")
            and g.get("match_type") == "Online" and (g.get("num_frames") or 0) >= MIN_FRAMES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()
    files = list_files()
    print(f"{len(files)} files listed", flush=True)
    order = files[:]
    random.Random(args.seed).shuffle(order)
    chosen, players, maps, screened, n_eligible = [], set(), {}, 0, 0
    with ThreadPoolExecutor(8) as ex:
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            for path, meta in zip(batch, ex.map(lambda p: (lambda: read_metadata(p))() , batch)):
                screened += 1
                if meta is None or not eligible(meta):
                    continue
                n_eligible += 1
                ids = {p["online_id"] for p in meta["players"]}
                mp = meta["game"].get("map_name")
                if ids & players or maps.get(mp, 0) >= MAP_CAP:
                    continue
                players |= ids
                maps[mp] = maps.get(mp, 0) + 1
                g = meta["game"]
                chosen.append({"path": path, "replay_id": g.get("id"), "map": mp, "date": g.get("date"),
                               "num_frames": g.get("num_frames"),
                               "score": [g.get("team_0_score") or 0, g.get("team_1_score") or 0],
                               "players": [{"online_id": p["online_id"], "kind": p.get("online_id_kind"),
                                            "is_orange": p["is_orange"], "unique_id": p["unique_id"]}
                                           for p in meta["players"]]})
                if len(chosen) >= args.n:
                    break
            print(f"screened {screened}, eligible {n_eligible}, chosen {len(chosen)}", flush=True)
            if len(chosen) >= args.n:
                break
    rng = random.Random(args.seed + 1)
    held = set(rng.sample(range(len(chosen)), args.holdout))
    for i, c in enumerate(chosen):
        c["index"], c["split"] = i, "holdout" if i in held else "train"
    for c in chosen:
        out = OUT / c["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.is_file():
            urllib.request.urlretrieve(BASE + c["path"], out)
        c["bytes"] = out.stat().st_size
    sel = {"source": REPO, "seed": args.seed, "screened": screened, "eligible_seen": n_eligible,
           "criteria": {"players": 2, "replay_name_contains": "Ranked Duel", "match_type": "Online",
                        "min_num_frames": MIN_FRAMES, "map_cap": MAP_CAP, "distinct_players": True},
           "maps": maps, "matches": chosen}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"selected {len(chosen)} ({args.holdout} held out) from {screened} screened; maps {maps}; "
          f"downloaded {sum(c['bytes'] for c in chosen) / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
