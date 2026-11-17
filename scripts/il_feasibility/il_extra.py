import sys, numpy as np
from pathlib import Path
sys.argv = [sys.argv[0], str(Path(__file__).parent)]
sys.path.insert(0, str(Path(__file__).parent))
import il_sim as S
from il_map import T, map_period
frames = [p.match_info.frame_num for p in S.P]; ph = [str(p.match_info.match_phase).split(".")[-1] for p in S.P]
ks = [i for i in range(1, len(S.P)) if ph[i] == "Kickoff" and ph[i-1] != "Kickoff"]
starts = [frames[i-1] for i in ks]; snaps = S.snapshots(starts)
res = {v: [] for v in ("RAW", "EVENT", "EVDIR", "NAIVE")}; taps = []
for f in starts:
    rs = S.series(f, 300); U = rs["Uh"][1:]; Ub = rs["Ub"][1:]; g = rs["hground"][1:]
    ev = T[np.repeat(map_period(U, g, "event"), 8)[:len(U)]]; evd = ev.copy(); aj = (ev[:, 5] > .5) & ~g; evd[aj, 2:5] = U[aj, 2:5]
    na = T[np.repeat(map_period(U, g, "naive"), 8)[:len(U)]]
    touch = np.where(np.linalg.norm(rs["ball"][1:] - rs["ball"][0], axis=1) > 5)[0]; tt = int(touch[0]) if len(touch) else 299
    rec_min = np.linalg.norm(rs["hpos"][1:tt+1] - rs["ball"][1:tt+1], axis=1).min()
    for name, Uh in (("RAW", U), ("EVENT", ev), ("EVDIR", evd), ("NAIVE", na)):
        sim = S.simulate(snaps[f], Uh, Ub)
        d = np.linalg.norm(sim["pos"][:tt] - rs["ball"][1:tt+1], axis=1)  # vs the ball's recorded (still resting) position
        res[name].append((d.min(), rec_min))
    # steering pulses on the ground before the first jump
    s = U[:, 1]; j0 = np.where(U[:, 5] > .5)[0]; s = s[: j0[0] if len(j0) else len(s)]
    k = 0
    while k < len(s):
        if s[k] != 0:
            e = k
            while e < len(s) and s[e] == s[k]: e += 1
            taps.append(e - k); k = e
        else: k += 1
for name, v in res.items():
    v = np.array(v)
    print(f"{name:5s} closest approach to the resting ball before the recorded first touch: median {np.median(v[:,0]):.0f} uu "
          f"(recorded {np.median(v[:,1]):.0f}); within 200 uu in {np.sum(v[:,0] < 200)}/20, within 300 uu in {np.sum(v[:,0] < 300)}/20")
taps = np.array(taps)
print(f"kickoff ground steering pulses before the first jump: {len(taps)}; length median {np.median(taps):.0f} ticks; "
      f"shorter than 8 ticks {np.mean(taps < 8):.0%}; 1-3 ticks {np.mean(taps <= 3):.0%}")
