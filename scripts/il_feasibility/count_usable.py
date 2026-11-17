"""Count usable demonstration content in a build_demos.py dataset, per match.

    .\\.venv\\Scripts\\python.exe scripts\\il_feasibility\\count_usable.py runs\\demos\\s1.npz

Usable = not flagged press_inflight / dodge_lost. Contiguity = consecutive decisions 8 frames apart.
* movement blocks: on the ground, no jump in the label, speed > 500 uu/s.
* recoveries: >= 4 consecutive airborne blocks (>= 0.27 s) without a fresh press by the human in the first of them
  (i.e. not an ordinary own jump), then landing wheels-down (self.up z > 0.7) and >= 4 more usable blocks.
* complete jump-release-dodge (strict): first-jump label on the ground -> >= 1 released block -> fresh airborne jump label
  (dodge press), all blocks usable and contiguous. Approximate-direction dodges counted separately.
  Relaxed: the same pattern traced back from each usable dodge press, allowing the first-jump label itself to be
  flagged in-flight (that one label is dropped from the loss; the release and dodge labels are usable).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from botcore.deploy.live_report import feature_names  # noqa: E402
from il_map import T  # noqa: E402

N = feature_names()


def runs_of_contiguous(frame):
    breaks = np.where(np.diff(frame) != 8)[0] + 1
    return np.split(np.arange(len(frame)), breaks)


def count(d, m):
    sel = d["match"] == m
    o, lab, prev, fr = d["obs"][sel], d["label"][sel], d["prev"][sel], d["frame"][sel]
    bad = d["press_inflight"][sel] | d["dodge_lost"][sel]
    approx = d["approx_dir"][sel]
    og = o[:, N.index("self.on_ground[0]")] > 0.5
    upz = o[:, N.index("self.up[2]")]
    speed = np.linalg.norm(o[:, [N.index(f"self.lin_vel[{i}]") for i in range(3)]], axis=1) * 2300
    jump, pj = T[lab, 5] > 0.5, T[prev, 5] > 0.5
    onset = jump & ~pj
    movement = int(np.sum(og & ~jump & (speed > 500) & ~bad))
    rec = jrd = jrd_approx = 0
    for idx in runs_of_contiguous(fr):
        # recoveries
        k = 0
        while k < len(idx):
            if not og[idx[k]]:
                s = k
                while k < len(idx) and not og[idx[k]]:
                    k += 1
                own_jump = onset[idx[max(s - 2, 0):s + 1]].any()
                if k - s >= 4 and not own_jump and k + 4 <= len(idx) and upz[idx[k]] > 0.7 and not bad[idx[s:k + 4]].any():
                    rec += 1
            else:
                k += 1
        # complete jump -> release -> dodge
        for a in range(len(idx)):
            i = idx[a]
            if not (onset[i] and og[i]) or bad[i]:
                continue
            for b in range(a + 1, min(a + 20, len(idx))):  # flip window 1.25 s ~ 19 blocks
                j = idx[b]
                if bad[j] or og[j]:
                    break
                if onset[j]:
                    released = (~jump[idx[a + 1:b]]).any()
                    if released:
                        jrd += 1
                        jrd_approx += bool(approx[j])
                    break
    # relaxed: every usable dodge press (fresh airborne jump label), traced back to the last ground block
    hj = o[:, N.index("self.has_jumped[0]")] > 0.5
    relaxed = no_ground_start = released_missing = from_ground_flagged = 0
    for idx in runs_of_contiguous(fr):
        for b in range(len(idx)):
            j = idx[b]
            if not (onset[j] and not og[j]) or bad[j]:
                continue
            a = b - 1
            while a >= 0 and not og[idx[a]] and b - a < 20:
                a -= 1
            if a < 0 or not og[idx[a]]:
                no_ground_start += 1  # dodge after a long airborne stretch / wall / not traceable inside the window
                continue
            mid = idx[a:b]
            if not jump[mid].any():
                no_ground_start += 1  # left the ground without a jump label (e.g. drove off a ramp / bump)
                continue
            if (~jump[idx[a + 1:b]]).sum() == 0:
                released_missing += 1
                continue
            relaxed += 1
            from_ground_flagged += bool(bad[mid].any())
    return dict(blocks=int(sel.sum()), usable=int((~bad).sum()), movement_blocks=movement, recoveries=rec,
                jump_release_dodge_strict=jrd, jump_release_dodge_relaxed=relaxed,
                relaxed_with_flagged_first_jump=from_ground_flagged, of_which_approx_direction=jrd_approx,
                dodge_presses_not_from_ground_jump=no_ground_start, release_missing=released_missing)


if __name__ == "__main__":
    d = np.load(sys.argv[1])
    recs = d["recordings"]
    tot = {}
    for m in np.unique(d["match"]):
        c = count(d, m)
        print(f"match {m} ({Path(str(recs[m])).parent.name}): {c}")
        for k, v in c.items():
            tot[k] = tot.get(k, 0) + v
    print("total:", tot)
