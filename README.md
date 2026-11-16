# BotCore: offline Rocket League bot (RocketSim + RLGym v2 + PPO + RLBot v5)

Offline-only. Training runs headless in RocketSim; the trained policy plays local matches through the
supported RLBot v5 interface. There is no online, ranked or matchmaking integration.

* What is verified and what is not, plus measured numbers: [STATUS.md](STATUS.md)
* Why this stack, sources and versions: [docs/DECISIONS.md](docs/DECISIONS.md)
* Observation contract, feature by feature: [docs/OBSERVATION_SPEC.md](docs/OBSERVATION_SPEC.md)
* Actions, timing, rewards, curriculum, self-play, evaluation, deployment limits, next experiments:
  [docs/DESIGN.md](docs/DESIGN.md)

## Commands

Use **PowerShell** in the project folder `C:\Users\TomPr\RLBotProject`. Activate the venv once per terminal:

```powershell
cd C:\Users\TomPr\RLBotProject
.\.venv\Scripts\Activate.ps1
```

(If PowerShell refuses to run the activation script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or
skip activation and replace `python` below with `.\.venv\Scripts\python.exe`.)

| Task | Command | Time |
|---|---|---|
| Diagnose the environment | `python -m botcore doctor` | ~10 s |
| Run the test suite | `python -m pytest` (add `-m "not slow"` to skip multi-process tests) | ~1 min |
| End-to-end plumbing check | `python -m botcore smoke-test` | ~20 s |
| Check a config | `python -m botcore validate --config configs\default.toml` | instant |
| Throughput benchmark | `python -m botcore benchmark --workers 4 6 8 12` | ~2 min |
| MLP vs entity model benchmark | `python -m botcore benchmark-models` | ~1 min |
| **Train** (full speed) | `python -m botcore train --config configs\default.toml --name s1-a` | until stopped |
| Train, leaving CPU free | `python -m botcore train --config configs\modest.toml --name s1-m` | until stopped |
| Stop training cleanly | `Ctrl+C`, or create `runs\<name>\STOP` | checkpoints first |
| Resume | `python -m botcore resume --run runs\s1-a` | |
| Resume with changed rewards/curriculum | `python -m botcore resume --run runs\s1-a --config configs\my_stage2.toml --allow-config-change` | |
| Export latest checkpoint | `python -m botcore export --checkpoint runs\s1-a` | seconds |
| Export and install into the RLBot bot | `python -m botcore export --checkpoint runs\s1-a --to-bot` | seconds |
| Compare two policies (headless) | `python -m botcore evaluate --a runs\s1-a --b runs\s1-a\checkpoints\iter_000100 --pairs 50` | minutes |
| Deployment latency CPU vs GPU | `python -m botcore latency --policy bot\model` | ~30 s |
| Replay a match in the browser | `python -m botcore replay-html --policy runs\s1-a --opponent runseference\R0-random-init --seconds 120` (or `--recording <file.bcp>` for a live match); open the HTML from `runsiz\` | ~20 s |
| Watch a simulated match in RLViser | `python -m botcore watch --policy bot\model --seconds 120` (needs `rlviser.exe`, below) | |
| Live game check with diagnostics | `python scripts\live_check.py` (BotCore vs BotCore, stops at overtime), then read `runs\live\<time>\live_report.json` | ~8 min |
| Goal attribution (own goals) | `python scripts\goal_attribution.py <bundle> <opponent bundle>` | ~1 min |
| Promote a candidate | `python -m botcore export --checkpoint runs\s1-a --mark-candidate`, evaluate it vs the current best with `--pairs 100`, then `python -m botcore promote --run runs\s1-a --reports <report.json> ...` | |

Training prints one line per iteration and appends full metrics (PPO stats, reward breakdown raw and weighted,
throughput, CPU/RAM/VRAM) to `runs\<name>\metrics.jsonl`. Checkpoints go to `runs\<name>\checkpoints\` (written
atomically, last 5 kept plus anything a pointer refers to). `runs\` is not tracked by Git.

**Long training is your decision to start.** Nothing in this repository starts an unattended run on its own.
A sensible first long run is `train --config configs\default.toml` for a few hours, then evaluate it against an
early checkpoint and watch it (see "Judging progress" in docs/DESIGN.md).

## Playing it in Rocket League (RLBot v5)

1. Export a trained checkpoint into the bot folder:
   `python -m botcore export --checkpoint runs\<name> --to-bot` (fills `bot\model\`).
2. Set Rocket League's frame rate cap to **120 or 240** (not 360, not uncapped), per rlgym-rlbot's timing guidance.
   VSync caps FPS too, so leave it off or make sure it allows at least 120.
3. Either:
   * **RLBot GUI** (`%LOCALAPPDATA%\RLBot5\bin\rlbotgui.exe`): add the folder `C:\Users\TomPr\RLBotProject\bot`
     (it contains `bot.toml`), then put BotCore on a team and start an offline match; or
   * **Script**: `python scripts\start_match.py bot\match_vs_allstar.toml` (BotCore vs Psyonix All-Star) or
     `python scripts\start_match.py bot\match_human_vs_bot.toml` (you vs BotCore).
4. The bot logs to the RLBot console: the bundle it loaded, then every 30 s its latency (p50/p99/max, number of
   decisions over 6 ms) and any fallback/fault counts. If the bundle fails to load it says so (CRITICAL) and
   sends neutral inputs rather than silently playing something else.

Optional environment variables for the bot: `BOTCORE_DEVICE=cuda:0` (default `cpu`), `BOTCORE_DETERMINISTIC=1`
(argmax instead of sampling), `BOTCORE_LIVE_DIR=<folder>` (per-bot log, packet/decision/timing recording and summary;
analyse with `python -m botcore live-report --dir <folder>`).

## Visualisers

**HTML replay (no extra software):** `python -m botcore replay-html ...` (table above) writes a self-contained page
with a top-down view, play/pause, speed and a scrub bar. Works for simulated matches and live recordings.

**RLViser (3D):** the installed `rlviser-py 0.6.13` speaks the protocol of RLViser **0.9.1 or older**; 0.9.2+ needs
rlviser-py 0.6.14+, which would force numpy 2 and break `rlgym-rocket-league 2.0.1`. Download
https://github.com/VirxEC/rlviser/releases/download/v0.9.1/rlviser.exe (SHA-256
`8155ff8b7883d57e7674987a912624ae9f0f3afd8386b1f860b939481ea1720b`) into this folder, then
`python -m botcore watch --policy <bundle|run|checkpoint> [--opponent <...>] [--scenario kickoff|shot|save|...]`.
`--dry-run` runs the same loop without opening a window. Rendering is never used during training.

## Layout

```
src/botcore/      obs/ (contract + builder)  actions/  rewards/  scenarios/  models/  collect.py  trainer.py
                  checkpoint.py  export.py  evaluate.py  opponents.py  deploy/ (RLBot runner)  cli.py
configs/          default (12 workers), modest (4 workers), entity (attention model), smoke
bot/              bot.toml, bot.py, match files; model/ is generated by `export --to-bot` (not in Git)
tests/            focused tests for the costly-to-miss bugs
docs/             DECISIONS.md, DESIGN.md, OBSERVATION_SPEC.md (generated)
```

## Reinstalling on a new machine

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m botcore doctor
```
