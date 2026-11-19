"""Retained-side (non-recorder) HL demonstration sets (docs/EXPERIMENT_HL_IMITATION.md, "Retained-side pilot").

    .\\.venv\\Scripts\\python.exe scripts\\hl_replays\\build_hl_demos.py

Writes runs/demos/hl1.npz (84 identified training matches), runs/demos/hl1-holdout.npz (15 held-out matches, used
only for the final held-out imitation fit) and runs/demos/hl1-counts.json. The format is the build_demos.py one read by
botcore.imitation.load_demos; `match` is the selection index.
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hlconv  # noqa: E402
import recorder  # noqa: E402
from hlconv import ROOT  # noqa: E402

KEYS = ("obs", "label", "prev", "frame", "tick", "approx_dir", "dodge_lost", "press_inflight", "press_offset", "dodge")


def one(job):
    index, path, retained_team = job
    d, st, fr, A = hlconv.extract(ROOT / "external" / "hlrlrd" / path, scheme="nearest", seed=index, keep_frames=True)
    c = int(np.where(A["team"] == retained_team)[0][0])
    sel = d["car"] == c
    out = {k: d[k][sel] for k in KEYS}
    use = ~out["press_inflight"] & ~out["dodge_lost"]
    cnt = {"index": index, "retained_team": "blue" if retained_team == 0 else "orange",
           "blocks_postkickoff_both_cars": st["blocks_postkickoff"], "blocks_after_exclusions": int(sel.sum()),
           "blocks_usable_by_loader": int(use.sum()), "usable_dodge_blocks": int((use & out["dodge"]).sum()),
           "dodges_in_grid": st["dodges_in_grid_kept_by_car"][c][0],
           "dodges_in_grid_kept": st["dodges_in_grid_kept_by_car"][c][1]}
    print(json.dumps(cnt), flush=True)
    return index, out, cnt


def main() -> int:
    rows = recorder.identify_selection(ROOT)
    res = {}
    for split, name in (("train", "hl1"), ("holdout", "hl1-holdout")):
        jobs = [(r["index"], r["path"], r["retained_team"]) for r in rows
                if r["split"] == split and r["recorder_orange"] is not None]
        with Pool(10) as pool:
            parts = pool.map(one, jobs, chunksize=1)
        parts.sort(key=lambda x: x[0])
        merged = {k: np.concatenate([p[1][k] for p in parts]) for k in KEYS}
        merged["match"] = np.concatenate([np.full(len(p[1]["label"]), p[0], np.int32) for p in parts])
        merged["obs"] = merged["obs"].astype(np.float32)
        merged["label"] = merged["label"].astype(np.int32)
        merged["prev"] = merged["prev"].astype(np.int32)
        np.savez_compressed(ROOT / "runs" / "demos" / f"{name}.npz", **merged)
        cnts = [p[2] for p in parts]
        res[split] = {"matches": len(cnts), "players": len(cnts),
                      "skipped_ambiguous": [r["index"] for r in rows if r["split"] == split and r["recorder_orange"] is None],
                      **{k: int(sum(c[k] for c in cnts)) for k in ("blocks_after_exclusions", "blocks_usable_by_loader",
                                                                     "usable_dodge_blocks", "dodges_in_grid", "dodges_in_grid_kept")},
                      "retained_blue": sum(c["retained_team"] == "blue" for c in cnts), "per_match": cnts}
        print(split, {k: v for k, v in res[split].items() if k != "per_match"}, flush=True)
    (ROOT / "runs" / "demos" / "hl1-counts.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
