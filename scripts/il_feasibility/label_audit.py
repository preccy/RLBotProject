import sys, numpy as np, pickle
from pathlib import Path
R = Path(r"C:\Users\TomPr\RLBotProject"); sys.path.insert(0, str(R / "src"))
from botcore.deploy.recording import load_recording
from rlbot import flat
rec = load_recording(R / "runs/live/human-argmax-11900/botcore-team1-index0.bcp")
P = [p for p in rec.packets if len(p.players) >= 2 and p.balls]
H = [next(x for x in p.players if x.team == 0) for p in P]
fr = np.array([p.match_info.frame_num for p in P]); ph = [str(p.match_info.match_phase).split(".")[-1] for p in P]
C = ["throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"]
U = np.array([[float(getattr(h.last_input, c)) for c in C] for h in H])
print("packets", len(P), "| frame deltas:", dict(zip(*np.unique(np.diff(fr), return_counts=True))))
print("human is_bot:", H[0].is_bot, " fields on PlayerInfo.last_input:", [a for a in dir(H[0].last_input) if not a.startswith('_') and a not in ('pack','unpack')])
for j, c in enumerate(C[:5]):
    v = U[:, j]; vals = np.unique(np.round(v, 3))
    print(f"  {c:9s} distinct values {len(vals):4d}  fraction at -1/0/+1: {np.mean(np.isin(np.round(v,3),[-1,0,1])):.1%}  sample {vals[:6]}")
act = np.array([x in ("Active", "Kickoff") for x in ph])
chg = np.any(np.diff(U, axis=0) != 0, axis=1)
print(f"control vector changes on {chg[act[1:]].mean():.1%} of consecutive in-play packets")
# timing: does each packet carry a fresh value? look at runs of identical vectors and min run
runs = []; r = 1
for k in range(1, len(U)):
    if np.array_equal(U[k], U[k-1]): r += 1
    else: runs.append(r); r = 1
print("run length of identical control vectors (ticks): min", min(runs), "median", int(np.median(runs)), "share of 1-tick runs", f"{np.mean(np.array(runs)==1):.1%}")
# alignment: jump press -> has_jumped / air_state change
air = [str(h.air_state).split(".")[-1] for h in H]
lags = []
for k in range(1, len(P) - 20):
    if U[k, 5] == 1 and U[k-1, 5] == 0 and air[k-1] == "OnGround" and act[k]:
        for L in range(0, 15):
            if H[k+L].has_jumped or air[k+L] != "OnGround": lags.append(L); break
print("jump press (ground) -> has_jumped/air_state change, lag in packets:", dict(zip(*np.unique(lags, return_counts=True))))
lags = []
for k in range(1, len(P) - 20):
    if U[k, 5] == 1 and U[k-1, 5] == 0 and air[k-1] in ("Jumping", "InAir") and not H[k-1].has_dodged and H[k-1].has_jumped and act[k]:
        for L in range(0, 15):
            if H[k+L].has_dodged: lags.append(L); break
print("2nd jump press (air) -> has_dodged, lag in packets:", dict(zip(*np.unique(lags, return_counts=True))))
lags = []
b = np.array([h.boost for h in H])
for k in range(1, len(P) - 20):
    if U[k, 6] == 1 and U[k-1, 6] == 0 and b[k] > 5 and act[k]:
        for L in range(0, 15):
            if b[k+L] < b[k-1] - 0.01: lags.append(L); break
print("boost press -> boost amount decreasing, lag in packets:", dict(zip(*np.unique(lags, return_counts=True))))
# dodge direction vs inputs at press
rows = []
for k in range(1, len(P) - 5):
    if H[k].has_dodged and not H[k-1].has_dodged:
        d = H[k+2].dodge_dir; rows.append((d.x, d.y, U[k-2:k+1, 2].tolist(), U[k-2:k+1, 3].tolist(), U[k-2:k+1, 4].tolist()))
print("dodges seen:", len(rows), " examples (dodge_dir x,y | pitch, yaw, roll at the 3 packets up to has_dodged):")
for r in rows[:6]: print("  ", np.round(r[0], 2), np.round(r[1], 2), "|", r[2], r[3], r[4])
pickle.dump({"fr": fr, "U": U, "phase": ph}, open(sys.argv[1] + "/human_inputs.pkl", "wb"))
