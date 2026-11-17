"""Build a supervised demonstration set (human car) from BotCore live recordings (.bcp). No training.

    .\\.venv\\Scripts\\python.exe scripts\\il_feasibility\\build_demos.py <out.npz> <recording.bcp> [...]   [--example]

Timing convention (matches training, where RocketSimEngine(rlbot_delay=True) runs the first tick of every decision
with the previous controls): the observation is the state at packet frame f; the target label is the table row for
the human inputs that produced frames f+2..f+9; `prev_action` is the row for frames f-6..f+1 (the action already
committed when the observation is taken). `last_input` at packet k is the input that produced packet k's state.

Labels are strictly local: each row depends only on the 8 inputs of its own block, so the previous-action feature
can never contain a later press. Per block: majority value per channel; jump = held on >= 4 ticks or a press starts
in the first 4 ticks; if a press starts in the block, pitch/yaw/roll come from the press tick (dodge direction).

Excluded: kickoff phase and the first 1.0 s after it, non-Active phases, blocks with missing packets (f..f+9 must
be consecutive), demolished human car. Flagged (kept): `approx_dir` (air jump with yaw, mapped onto roll: exact only
if yaw+roll stays in -1..1), `dodge_lost` (blocks around a recorded dodge whose press/release/press pattern the
local rule does not reproduce).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from botcore.deploy.live_report import feature_names  # noqa: E402
from botcore.deploy.packets import CompatObsPipeline  # noqa: E402
from botcore.deploy.recording import load_recording  # noqa: E402
from il_map import T, ctl, nearest_row  # noqa: E402

NAMES = feature_names()
PREV = slice(NAMES.index("self.prev_action[0]"), NAMES.index("self.prev_action[0]") + 8)
SKIP_AFTER_KICKOFF = 120


class JumpMachine:
    """Causal jump labelling over consecutive 8-tick windows. A block's jump bit depends only on its own window and
    earlier ones: a press starting in the first 4 ticks of a window is assigned to that block (<= 3 ticks early); a
    later press waits for the next block (<= 4 ticks late); between two presses at least one released block is kept,
    delaying the second press if needed, so press -> release -> press (dodge) patterns survive."""

    def __init__(self):
        self.prev_bit = False
        self.pending = None  # (stick pitch/yaw/roll, on_ground) of an unassigned press
        self.last_tick_jump = False

    def step(self, u: np.ndarray, ground: np.ndarray):
        j = u[:, 5] > 0.5
        starts = [k for k in range(8) if j[k] and (not j[k - 1] if k > 0 else not self.last_tick_jump)]
        self.last_tick_jump = bool(j[-1])
        press = None
        if self.prev_bit:
            # keep holding only if still held from the start of the window; a new press after a release waits
            bit = bool(j[0] and j.sum() >= 4 and not any(k > 0 for k in starts))
            if starts:
                self.pending = (u[starts[-1], 2:5], bool(ground[starts[-1]]), starts[-1] - 8)
        else:
            if self.pending is not None:
                bit, press = True, self.pending
                self.pending = None
            elif starts and starts[0] < 4:
                bit, press = True, (u[starts[0], 2:5], bool(ground[starts[0]]), starts[0])
            elif starts:
                bit = False
                self.pending = (u[starts[0], 2:5], bool(ground[starts[0]]), starts[0] - 8)
            else:
                bit = False
        self.prev_bit = bit
        return bit, press


def block_row(u: np.ndarray, ground: np.ndarray, bit: bool, press) -> tuple[int, bool]:
    """Table row for one 8-tick window given the machine's jump decision. Returns (row, approx_dir)."""
    tgt = np.array([Counter(np.round(u[:, j], 3)).most_common(1)[0][0] for j in range(8)], dtype=np.float32)
    tgt[5] = float(bit)
    g = float(np.mean(ground))
    approx, dodge = False, False
    if press is not None:
        stick, on_ground, _ = press
        tgt[2:5] = stick
        if not on_ground:
            g, dodge = 0.0, True
            approx = bool(abs(stick[1] + stick[2]) > 1)  # yaw+roll = +-2: table can only give +-1
    if bit and not dodge:
        tgt[3] = 0.0  # table jump rows have no yaw; outside a dodge press, yaw is dropped rather than turned into roll
    return nearest_row(tgt, g, dodge=dodge), approx


def extract(path: Path, example: bool = False):
    rec = load_recording(path)
    pipe = CompatObsPipeline(rec.field)
    pk = [p for p in rec.packets if len(p.players) >= 2 and p.balls]
    hum = next(x for x in pk[0].players if x.team == 0).player_id
    F, PH, U, G, DEM, OBS, DODGE = [], [], [], [], [], [], []
    for p in pk:
        pipe.update(p)  # causal: the observation at frame f is built before packet f+1 is seen
        h = next(x for x in p.players if x.player_id == hum)
        F.append(p.match_info.frame_num); PH.append(str(p.match_info.match_phase).split(".")[-1])
        U.append(ctl(h.last_input)); G.append(str(h.air_state).endswith("OnGround"))
        DEM.append(h.demolished_timeout > 0); DODGE.append(h.has_dodged)
        OBS.append(pipe.obs(hum, prev_action=None))  # prev_action slice filled in below
    F, U, G, DEM, DODGE = np.array(F), np.array(U), np.array(G), np.array(DEM), np.array(DODGE)
    OBS = np.array(OBS, dtype=np.float32)
    # valid Active stretches, kickoff excluded
    valid_from = np.zeros(len(F), bool)
    k = 0
    while k < len(F):
        if PH[k] == "Kickoff":
            while k < len(F) and PH[k] == "Kickoff":
                k += 1
            s = k + SKIP_AFTER_KICKOFF
            while k < len(F) and PH[k] == "Active":
                if k >= s:
                    valid_from[k] = True
                k += 1
        else:
            k += 1
    # contiguous usable stretches: valid phase, consecutive frames, human not demolished
    good = valid_from & ~DEM
    good[1:] &= np.diff(F) == 1
    rows, prevs, obs, frames, approx, inflight, offs = [], [], [], [], [], [], []
    k = 0
    while k < len(F):
        if not good[k]:
            k += 1
            continue
        z = k
        while z < len(F) and good[z]:
            z += 1
        # windows of the stretch [k, z): window w covers U[k + 8w : k + 8w + 8]; the decision for window w (w >= 1)
        # is taken at obs frame index k + 8w - 2 (target = frames f+2..f+9) and its prev_action is window w-1
        machine, labels = JumpMachine(), []
        w = 0
        while k + 8 * w + 8 <= z:
            a = k + 8 * w
            bit, press = machine.step(U[a:a + 8], G[a - 1:a + 7] if a > 0 else G[a:a + 8])
            r, ap = block_row(U[a:a + 8], G[a - 1:a + 7] if a > 0 else G[a:a + 8], bit, press)
            labels.append((r, ap, press))
            w += 1
        for w in range(1, len(labels)):
            i = k + 8 * w - 2
            r, ap, press = labels[w]
            o = OBS[i].copy(); o[PREV] = T[labels[w - 1][0]]
            rows.append(r); prevs.append(labels[w - 1][0]); obs.append(o); frames.append(F[i]); approx.append(ap)
            # the press began before (or at) the observation frame: obs already shows it in flight
            inflight.append(press is not None and press[2] < -1)
            offs.append(np.nan if press is None else press[2])
        k = z
    rows, frames = np.array(rows), np.array(frames)
    # dodge preservation: each recorded dodge (has_dodged rising edge) inside the grid needs ... 1,0,1 ... before it
    lost = np.zeros(len(rows), bool)
    edges = [F[k] for k in range(1, len(F)) if DODGE[k] and not DODGE[k - 1]]
    jb = T[rows, 5] > 0.5
    n_dodge = n_kept = 0
    onset = jb & ~(T[np.array(prevs), 5] > 0.5) if len(rows) else jb
    for e in edges:
        b = np.searchsorted(frames, e - 2) - 1  # block whose target window (f+2..f+9) contains the dodge press
        if b < 1 or not (frames[b] + 2 <= e <= frames[b] + 9) or frames[b] - frames[b - 1] != 8:
            continue  # dodge outside the usable grid (kickoff, gap, demolition, replay)
        n_dodge += 1
        kept = bool(onset[max(b - 1, 0):b + 2].any())  # a fresh jump label within one block of the human's press
        n_kept += kept
        if not kept:
            lost[max(b - 1, 0):b + 2] = True
    out = dict(obs=np.array(obs), label=rows, prev=np.array(prevs), frame=frames, approx_dir=np.array(approx),
               dodge_lost=lost, press_inflight=np.array(inflight), press_offset=np.array(offs, dtype=float))
    stats = dict(packets=len(F), blocks=len(rows), dodges_in_grid=n_dodge, dodges_pattern_kept=n_kept)
    if example:
        timing_checks(out, OBS, F, U, G, edges)
    return out, stats


def timing_checks(d, OBS, F, U, G, edges):
    o, lab = d["obs"], d["label"]
    on_g, hj = o[:, NAMES.index("self.on_ground[0]")], o[:, NAMES.index("self.has_jumped[0]")]
    hf, fa = o[:, NAMES.index("self.has_flipped[0]")], o[:, NAMES.index("self.flip_available[0]")]
    jump_lab, prev_j = T[lab, 5] > 0.5, d["prev"] >= 0
    pj = T[d["prev"], 5] > 0.5
    onset = jump_lab & ~pj
    first = onset & (on_g > 0.5)
    dodge = onset & (on_g < 0.5)
    print(f"[leak checks] first-jump labels (onset, on ground in obs): {first.sum()}; obs already shows has_jumped "
          f"{np.mean(hj[first] > 0.5):.0%}")
    print(f"              dodge labels (onset, airborne in obs): {dodge.sum()}; obs has_flipped already {np.mean(hf[dodge] > 0.5):.0%}, "
          f"flip_available {np.mean(fa[dodge] > 0.5):.0%}")
    po = d["press_offset"][~np.isnan(d["press_offset"])]
    print(f"              press timing, label start minus human press tick: {dict(zip(*np.unique(-po.astype(int), return_counts=True)))} "
          f"(positive = label later than the human)")
    infl = d["press_inflight"]
    print(f"              first-jump labels already in flight in obs (flagged press_inflight): {np.mean(infl[first]):.0%}; "
          f"not flagged but obs shows has_jumped: {np.mean((hj[first] > 0.5) & ~infl[first]):.0%}")
    print(f"              prev_action jump bit at jump onsets: {np.mean(pj[onset]):.0%} (must be 0 by construction)")
    print(f"              label == prev_action (repeat-last baseline for any imitation model): {np.mean(lab == d['prev']):.0%}")
    # one concrete sequence: the first recorded dodge that lies on the decision grid
    fr = d["frame"]
    for e in edges:
        b = np.searchsorted(fr, e - 2) - 1
        if 3 <= b < len(fr) - 2 and fr[b + 1] - fr[b - 3] == 32 and on_g[b - 3] > 0.5:
            print(f"\n[example] human dodge press registered at frame {e} (has_dodged rising edge)")
            print(" obs frame | on_gnd has_jmp flip_av has_flp | prev_action (thr,str,pit,yaw,rol,J,B,hb) | label row (target frames f+2..f+9) | human jump/pitch/roll on target ticks")
            for x in range(b - 3, b + 2):
                f = fr[x]
                k = int(np.where(F == f)[0][0])
                print(f" {f:9d} |   {on_g[x]:.0f}      {hj[x]:.0f}      {fa[x]:.0f}       {hf[x]:.0f}    | {T[d['prev'][x]].astype(int)} | "
                      f"{T[lab[x]].astype(int)} | J {''.join('1' if v else '.' for v in U[k+2:k+10,5]>0.5)} "
                      f"P {''.join({-1:'-',0:'0',1:'+'}[int(v)] for v in U[k+2:k+10,2])} R {''.join({-1:'-',0:'0',1:'+'}[int(v)] for v in U[k+2:k+10,4])}")
            break


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path, recs = Path(args[0]), [Path(a) for a in args[1:]]
    parts = []
    for m, r in enumerate(recs):
        d, st = extract(r, example="--example" in sys.argv and m == 0)
        d["match"] = np.full(len(d["label"]), m)
        parts.append(d)
        print(f"{r}: {st}; approx_dir blocks {int(d['approx_dir'].sum())}, dodge_lost blocks {int(d['dodge_lost'].sum())}")
    all_ = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(out_path, **all_, recordings=np.array([str(r) for r in recs]))
    print(f"wrote {out_path}: {len(all_['label'])} labelled blocks from {len(recs)} recording(s)")
