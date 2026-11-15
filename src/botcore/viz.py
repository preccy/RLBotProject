"""Dependency-free match viewer: a self-contained HTML file with a top-down animation.

Sources: a simulated match between two saved policies (RocketSim) or a live packet recording
(BOTCORE_LIVE_DIR). Frames are sampled at the policy's decision rate (15 Hz). Opened in any browser; nothing
here is used by training or benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from botcore import constants as C


def _car_frame(pos, yaw, team, boost, on_ground, extra=None) -> list:
    return [round(float(pos[0])), round(float(pos[1])), round(float(pos[2])), round(float(yaw), 3), int(team),
            int(round(boost)), int(bool(on_ground))]


def frames_from_simulation(policy: Path, opponent: Path, seconds: float, seed: int,
                           scenario: str = "kickoff", deterministic: bool = False) -> Dict:
    from botcore.env import EnvSpec, build_env
    from botcore.export import PolicyRunner, load_bundle

    blue_b, orange_b = load_bundle(policy), load_bundle(opponent)
    runners = {C.BLUE_TEAM: PolicyRunner(blue_b, deterministic=deterministic, seed=seed),
               C.ORANGE_TEAM: PolicyRunner(orange_b, deterministic=deterministic, seed=seed + 1)}
    spec = EnvSpec(team_size=1, action_repeat=blue_b.action_repeat, rlbot_delay=blue_b.rlbot_delay,
                   no_touch_timeout_seconds=1e9, episode_timeout_seconds=1e9,
                   scenario_weights={scenario: 1.0}, reward_weights={})
    env = build_env(spec, seed)
    obs = env.reset()
    frames: List[list] = []
    score = [0, 0]
    for step in range(int(seconds * C.TICKS_PER_SECOND / blue_b.action_repeat)):
        st = env.state
        cars = [_car_frame(c.physics.position, c.physics.yaw, c.team_num, c.boost_amount, c.on_ground)
                for c in st.cars.values()]
        frames.append([round(step * blue_b.action_repeat / C.TICKS_PER_SECOND, 2),
                       [round(float(v)) for v in st.ball.position], cars, list(score)])
        actions = {a: np.array([runners[st.cars[a].team_num].act(obs[a])]) for a in env.agents}
        obs, _, term, _ = env.step(actions)
        if any(term.values()):
            score[env.state.scoring_team] += 1
            obs = env.reset()
    env.close()
    return {"title": f"RocketSim: {policy.name} (blue) vs {opponent.name} (orange), seed {seed}",
            "frames": frames}


def frames_from_recording(recording: Path, every: int = 8) -> Dict:
    from botcore.deploy.recording import load_recording

    rec = load_recording(recording)
    frames = []
    last = None
    for packet in rec.packets:
        f = packet.match_info.frame_num
        if last is not None and f - last < every:
            continue
        last = f
        if not packet.balls:
            continue
        b = packet.balls[0].physics.location
        cars = []
        for p in packet.players:
            loc, rot = p.physics.location, p.physics.rotation
            cars.append(_car_frame((loc.x, loc.y, loc.z), rot.yaw, p.team, p.boost,
                                   str(p.air_state).endswith("OnGround")))
        frames.append([round(f / C.TICKS_PER_SECOND, 2), [round(b.x), round(b.y), round(b.z)], cars,
                       [t.score for t in packet.teams] or [0, 0], str(packet.match_info.match_phase).split(".")[-1]])
    return {"title": f"Live Rocket League recording: {recording.name}", "frames": frames}


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BotCore Replay</title>
<style>
:root{--bg:#f4f5f7;--fg:#1d2330;--muted:#5d6676;--pitch:#2f6b3a;--line:#e8f0e8;--blue:#2a6df4;--orange:#f08c1a;--ball:#f4f4f4}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ef;--muted:#9aa3b2;--pitch:#1f4a28;--line:#b9c7b9}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:12px 16px}
h1{font-size:16px;margin:0 0 8px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}
canvas{width:100%;height:auto;display:block;border-radius:8px}
input[type=range]{flex:1;min-width:160px}
.muted{color:var(--muted)}
button{font:inherit;padding:4px 12px}
</style></head><body><main>
<h1 id="title"></h1>
<div class="bar"><button id="play">Pause</button>
<label>speed <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
<input id="scrub" type="range" min="0" value="0"><span id="info" class="muted"></span></div>
<canvas id="c" width="820" height="1000"></canvas>
<p class="muted">Top-down view from behind the blue goal (blue defends the bottom goal; white square = car front). Car size grows with height; a ring marks a car off the ground; the ball's shadow shows its height. 15 frames per game second.</p>
</main>
<script>
const DATA = __DATA__;
const F = DATA.frames, cv = document.getElementById('c'), g = cv.getContext('2d');
document.getElementById('title').textContent = DATA.title;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const W = 4096, L = 5120, GOAL_W = 893, GOAL_D = 880, S = cv.width / (2 * W + 400);
// Rocket League is left-handed: facing from the blue goal towards orange, +x is to the LEFT.
const X = x => cv.width / 2 - x * S, Y = y => cv.height / 2 - y * S;
const scrub = document.getElementById('scrub'); scrub.max = F.length - 1;
let i = 0, playing = true, acc = 0, last = performance.now();
function field(){
  g.fillStyle = css('--bg'); g.fillRect(0, 0, cv.width, cv.height);
  g.fillStyle = css('--pitch'); g.strokeStyle = css('--line'); g.lineWidth = 2;
  const c = 1152;
  g.beginPath(); g.moveTo(X(-W + c), Y(L)); g.lineTo(X(W - c), Y(L)); g.lineTo(X(W), Y(L - c)); g.lineTo(X(W), Y(-L + c));
  g.lineTo(X(W - c), Y(-L)); g.lineTo(X(-W + c), Y(-L)); g.lineTo(X(-W), Y(-L + c)); g.lineTo(X(-W), Y(L - c)); g.closePath();
  g.fill(); g.stroke();
  g.beginPath(); g.moveTo(X(-W), Y(0)); g.lineTo(X(W), Y(0)); g.stroke();
  g.beginPath(); g.arc(X(0), Y(0), 1000 * S, 0, 7); g.stroke();
  for (const [s, col] of [[1, '--orange'], [-1, '--blue']]) { g.fillStyle = css(col); g.globalAlpha = .55;
    g.fillRect(X(-GOAL_W), s > 0 ? Y(L + GOAL_D) : Y(-L), 2 * GOAL_W * S, GOAL_D * S); g.globalAlpha = 1; }
}
function draw(){
  const f = F[i]; field();
  const [t, ball, cars, score, phase] = f;
  g.fillStyle = 'rgba(0,0,0,.35)'; g.beginPath(); g.arc(X(ball[0]), Y(ball[1]), 92 * S, 0, 7); g.fill();
  for (const c of cars) {
    const [x, y, z, yaw, team, boost, ground] = c, k = 1 + Math.min(z, 2000) / 1200;
    g.save(); g.translate(X(x), Y(y)); g.rotate(yaw + Math.PI); g.fillStyle = css(team ? '--orange' : '--blue');
    g.fillRect(-60 * S * k, -42 * S * k, 120 * S * k, 84 * S * k);
    g.fillStyle = '#fff'; g.fillRect(40 * S * k, -10 * S * k, 20 * S * k, 20 * S * k); g.restore();
    if (!ground) { g.strokeStyle = '#fff'; g.beginPath(); g.arc(X(x), Y(y), 95 * S * k, 0, 7); g.stroke(); }
  }
  const kb = 1 + Math.min(ball[2], 2000) / 1000;
  g.fillStyle = css('--ball'); g.strokeStyle = '#222';
  g.beginPath(); g.arc(X(ball[0]), Y(ball[1] + ball[2] * .25), 92 * S * kb, 0, 7); g.fill(); g.stroke();
  document.getElementById('info').textContent =
    `t ${t.toFixed(1)}s  score ${score[0]}-${score[1]}${phase ? '  ' + phase : ''}  ball z ${ball[2]}`;
  scrub.value = i;
}
function tick(now){
  const dt = (now - last) / 1000; last = now;
  if (playing) { acc += dt * parseFloat(document.getElementById('speed').value);
    const step = F.length > 1 ? Math.max(F[1][0] - F[0][0], 1 / 15) : 1;
    while (acc >= step && i < F.length - 1) { acc -= step; i++; } }
  draw(); requestAnimationFrame(tick);
}
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; };
scrub.oninput = () => { i = +scrub.value; acc = 0; };
requestAnimationFrame(tick);
</script></body></html>
"""


def write_html(data: Dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    return out
