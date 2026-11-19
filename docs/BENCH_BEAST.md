# Live benchmark: CTRL-13300 vs Beast from the East (offline, RLBot v5)

Run 2026-11-19. Six recorded offline 1v1 matches in Rocket League. This is a live benchmark against one community
bot, not a rank estimate and not a promotion check.

## Setup
* **BotCore:** the deployed `bot/model`, CTRL-13300 (`0e3fd5323e0a`). It sampled its actions as deployed, with CPU
  inference.
* **Opponent:** Beast from the East v2.3.0 by Eastvillage (`eastvillage/beast/v2.3.0`). It is the prebuilt
  `beastbot.exe` from the official RLBot v5 botpack, as downloaded by the RLBot GUI to
  `%LOCALAPPDATA%/RLBot5/bots/bob_build_x86_64-windows/Beast` (botpack build hash `2a57e958e8f7ddd6`).
* **Matches:** standard 5-minute matches on Stadium_P. Colours alternate blue, orange, ... (3 each). A match tied at
  the end of regulation would have been stopped and counted a draw; none was.
* **Tools:** `scripts/beast_series.py` runs each match through `scripts/live_check.py` (packet recording, live
  report) with `bot/match_vs_beast.toml` and `bot/match_beast_orange.toml`. It also writes an HTML top-down replay
  per match. `scripts/possession_analysis.py` analyses the recordings.
* **Outputs:** `runs/beast/ctrl-13300/` (`summary.md`, `possession.json`, `losses_detail.json`, `m*-*.html`).
* **No in-game replays:** the match configs set `skip_replays = true`, as in the All-Star series, so Rocket League
  saved no `.replay` files. The HTML replays' time `t` equals the "seconds elapsed" used below.

## Results
| # | BotCore colour | Score (BotCore-Beast) | BotCore shots | Beast shots | Beast saves |
|---|---|---|---|---|---|
| 1 | blue | 11-1 | 15 | 1 | 1 |
| 2 | orange | 7-0 | 12 | 0 | 0 |
| 3 | blue | 3-0 | 8 | 0 | 0 |
| 4 | orange | 10-0 | 12 | 0 | 1 |
| 5 | blue | 7-0 | 12 | 0 | 1 |
| 6 | orange | 6-0 | 7 | 1 | 1 |

**Totals:** 6-0-0, goals **44-1**, shots 66-2. Neither side scored an own goal.

* Touches per active minute: BotCore 49-64; Beast 8-12.
* Runner clean in all six matches: 0 fallback ticks, 0 policy faults, 0 unexpected or missed kickoff resets,
  callback p99 at most 1.7 ms.
* The rlgym-rlbot "missed step" warnings appeared at the same rate as in the clean All-Star series (about 5 per
  match).

## Possession occasions
Definitions were fixed in `scripts/possession_analysis.py` before the Beast recordings were inspected:
* **Open play:** Active phase, and at least 4 s after the kickoff went live.
* **13300 wins the ball:** a 13300 touch after a Beast touch, or after the ball had been untouched for at least 2 s.
* **Room to act:** Beast at least 1,000 uu from the ball, and at least 0.5 s away at its closing speed.
* **Shot followed:** Rocket League's own shot counter rose, or 13300 scored, within 5 s.

Each match had 47-62 open-play ball wins, of which 21-28 had room to act.

### The first two occasions per match
In the table, "y" is the ball's position toward Beast's goal: 0 is midfield and +5120 is Beast's goal line.
Pressure gives Beast's distance from the ball, its closing speed (negative means moving away) and its estimated time
to the ball.

| Match | Clock (t) | Won from | Pressure | Ball y | 13300's first touch | Next to ball | Shot within 5 s? | How the possession ended |
|---|---|---|---|---|---|---|---|---|
| 1 blue | 4:50 (20.7) | loose ball | 2,287 uu, moving away | -47 | forward, 1,445 uu/s | 13300 after 1.8 s | no | contested loss near Beast's goal line after 7.0 s |
| 1 blue | 4:20 (59.3) | Beast touch | 4,274 uu, closing 1,116, 3.8 s | 3,294 | forward, 1,239 uu/s | 13300 after 1.3 s | no | **goal** after 8.9 s, 10 touches (1-1) |
| 2 orange | 4:51 (19.1) | Beast touch | 3,220 uu, closing 1,730, 1.9 s | -2,151 | forward, 1,846 uu/s | 13300 after 2.3 s | **yes** | shot after 5.1 s |
| 2 orange | 4:49 (21.4) | loose ball | 1,848 uu, moving away | -4 | forward, 1,783 uu/s | 13300 after 0.5 s | **yes, goal** | shot, then goal |
| 3 blue | 4:55 (15.0) | loose ball | 4,427 uu, closing 565, 7.8 s | 3,444 | forward, 1,366 uu/s | 13300 after 1.0 s | no | contested loss after 1.8 s |
| 3 blue | 4:49 (20.7) | Beast touch | 1,488 uu, closing 1,156, 1.3 s | 1,911 | soft/sideways, 893 uu/s | 13300 after 0.6 s | no | Beast collected cleanly after 11.1 s |
| 4 orange | 4:53 (16.3) | loose ball | 1,706 uu, closing 1,208, 1.4 s | 3,895 | backwards, 1,699 uu/s | 13300 after 0.9 s | no | **goal** after 14.1 s |
| 4 orange | 4:46 (23.8) | loose ball | 1,558 uu, moving away, not goal-side | 2,191 | forward, 1,836 uu/s | 13300 after 2.5 s | **yes, goal** | goal |
| 5 blue | 4:55 (14.8) | loose ball | 5,468 uu, moving away | 354 | forward, 1,503 uu/s | 13300 after 4.6 s | **yes** | **goal** after 8.8 s |
| 5 blue | 4:44 (35.5) | Beast touch | 3,563 uu, closing 1,716, 2.1 s | 3,008 | forward, 1,336 uu/s | 13300 after 1.2 s | **yes** | shot after 4.7 s |
| 6 orange | 4:55 (15.7) | loose ball | 2,764 uu, closing 1,686, 1.6 s | 1,007 | projected on target, 1,682 uu/s | **Beast** after 0.85 s | no | Beast collected cleanly (13300 >500 uu away) |
| 6 orange | 4:49 (21.4) | Beast touch | 2,944 uu, closing 1,630, 1.8 s | -1,569 | soft/sideways, 1,905 uu/s | 13300 after 4.5 s | no | **goal** after 11.1 s |

* Beast was goal-side in 11 of the 12 occasions.
* 13300 reached the ball next in 11 of 12.
* A shot followed within 5 s in 5 of 12.

### All 148 occasions with room (context for the first-two sample)
How the possession ended:

| Ending | Count |
|---|---|
| 13300 goal | 41 |
| 13300 shot | 24 |
| Lost in a contested touch (13300 within 500 uu when Beast touched) | **61** |
| Beast collected cleanly | 19 |
| Play stopped | 3 |

* **Where losses happened:** the ball was rarely lost early. Only 10 of the 80 losses came within 1.5 s, and 68 of
  the 80 happened in Beast's defensive third (44 within 1,500 uu of its goal line). Beast was goal-side at 84% of
  all occasions.
* **The contested losses (61):**
  * the ball was low (median height 105 uu; only 1 above 300 uu);
  * 13300 had zero boost in 43 of them and was airborne (jump or dodge) in 27;
  * Beast's touch sent the ball back toward 13300's half in 30;
  * two seconds later the ball was in 13300's half in only 8 of these.
* **Boost:** 13300 spent 65-77% of active time on zero boost (Beast 20-48%). Its median boost at the moment it won
  the ball was 0.

## Conclusions
* **Competitive practice: no.** Beast lost every match, 44-1 in total, managed 2 shots in 30 minutes, and touched the
  ball about one-sixth as often as 13300. When Beast won the ball back, it rarely got it out of its own half. Beast
  was about as uncompetitive for 13300 as the Psyonix All-Star (40-2 in 4). It does give one narrow kind of
  practice: it sits goal-side and contests the ball near its own goal.
* **What most often stops 13300 turning possession into an attack:** the attack usually exists. 13300 carries the
  ball into Beast's third. It is most often stopped at the end of that carry: the ball is lost in a low 50/50 near
  Beast's goal, before a shot. 13300 typically arrives with no boost and is often airborne at the moment it loses
  the ball. Losing the ball to an early bad first touch is rare.
* **Limits:** six matches against one bot, so this is descriptive only. "Contested", "room" and the first-touch
  labels are proximity and velocity heuristics fixed before inspection. Touches are contact events, and a dribble
  produces many.

## Worth watching (HTML replays in `runs/beast/ctrl-13300/`, time `t`)
* `m3-blue.html` t 131-141 (clock 3:08 -> 2:58): 13300 wins a loose ball and carries it into Beast's third on zero
  boost, then loses an airborne 50/50 near Beast's goal. This is the typical stall.
* `m6-orange.html` t 278-288 (1:19 -> 1:09): the same pattern, ending at Beast's goal line.
* `m6-orange.html` t 15.7 (4:55): a first touch heading on target that Beast collects 0.85 s later.
* `m1-blue.html` t 59-68 (4:20 -> 4:14): the counter-example. From a Beast touch, 13300 dribbles in 10 touches and
  equalises.
