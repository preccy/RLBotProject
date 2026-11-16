"""Drive the real BotCoreBot (rlgym-rlbot runner) offline with recorded or synthetic packets.

Mirrors the body of rlgym-rlbot 0.3.0's `RLGymBot._run` loop (one packet per iteration, as when the bot keeps
up with the game) with the socket replaced by a stub, so the exact runner code that plays in Rocket League can
be tested without the game: kickoff/overtime resets, fallbacks, action timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from rlbot import flat


@dataclass
class OfflineRun:
    sent: List[flat.ControllerState] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)


def attach(bot, field_info: flat.FieldInfo, team: int, index: int, player_id: int,
           opponents: Sequence[tuple] = ()) -> None:
    """Initialise a BotCoreBot as RLBotServer would (match config, field info, controllable team info).

    `opponents`: (team, player_id) of the other cars in the match configuration.
    """
    bot._game_interface.send_msg = lambda msg: None  # no server
    players = [flat.PlayerConfiguration(variety=flat.CustomBot(name="BotCore"), team=team, player_id=player_id)]
    players += [flat.PlayerConfiguration(variety=flat.CustomBot(name="Other"), team=t, player_id=pid)
                for t, pid in opponents]
    bot._handle_match_config(flat.MatchConfiguration(player_configurations=players))
    bot._handle_field_info(field_info)
    bot._handle_controllable_team_info(flat.ControllableTeamInfo(
        team=team, controllables=[flat.ControllableInfo(index=index, identifier=player_id)]))
    if not bot._initialized_bot:
        raise RuntimeError("BotCoreBot did not initialise")


def drive(bot, packets: Sequence[flat.GamePacket]) -> OfflineRun:
    run = OfflineRun()
    for packet in packets:
        bot._handle_packet(packet)
        bot._process_unused_packets()
        bot._last_packet = bot._unused_packets[-1]
        bot._unused_packets.clear()
        tick = bot._last_packet.match_info.frame_num
        if tick in bot._tick_action_map:
            bot._last_sent_action_tick = tick
            bot._last_sent_action = bot._tick_action_map[tick]
        run.sent.append(bot._last_sent_action)
        run.frames.append(tick)
    return run
