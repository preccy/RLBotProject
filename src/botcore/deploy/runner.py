"""RLBot v5 runner: an exported bundle driven through RLGym's rlgym-rlbot wrapper.

rlgym-rlbot 0.3.0 (RLGymBot) does the timing work: it rebuilds an RLGym-compatible GameState from every
GamePacket (rlgym-compat), steps the same ObsBuilder/ActionParser objects used in training on the same
tick schedule (StepOffset -1 relative to ACTION_END matches RocketSimEngine(rlbot_delay=True)), resets
them at kickoffs, and submits one ControllerState per physics tick. This module adds:

* bundle loading with a hard contract check, and a logged zero-action fallback if loading or inference
  fails (never a silent substitute policy);
* reconstructed boost pad timers and flip-window expiry (botcore.deploy.state_fixes);
* full-callback latency instrumentation with a 6 ms deadline counter, reset counting;
* optional live diagnostics (BOTCORE_LIVE_DIR): a log file, a packet + decision + timing recording and a
  summary JSON per bot instance, analysed with `python -m botcore live-report`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from rlbot import flat
from rlbot.utils.logging import DEFAULT_LOGGER
from rlgym_rlbot import RLGymBot, RLGymBotConfig
from rlgym_rlbot.rlgym_bot import StepOffset, StepOffsetRelativeTo

from botcore.actions import LookupActionParser, get_action_table
from botcore.deploy.state_fixes import LatencyStats, PadTimerTracker, flip_expired_ids
from botcore.export import Bundle, PolicyRunner, load_bundle
from botcore.obs import FLIP_EXPIRED_KEY, PAD_TIMERS_KEY, SpecObsBuilder

DEFAULT_AGENT_ID = "local/botcore"
REPORT_EVERY_SECONDS = 30.0
LIVE_DIR_ENV = "BOTCORE_LIVE_DIR"
_ACTIVE_PHASES = (flat.MatchPhase.Countdown, flat.MatchPhase.Kickoff, flat.MatchPhase.Active)


def zero_action_index() -> int:
    rows = np.nonzero((get_action_table().controls == 0).all(axis=1))[0]
    if len(rows) != 1:
        raise RuntimeError("action table has no unique all-zero row")
    return int(rows[0])


def step_offset_for(bundle: Optional[Bundle]) -> StepOffset:
    """rlgym-rlbot's documented setting for RocketSimEngine(rlbot_delay=True) is offset -1 at action end."""
    delay = True if bundle is None else bundle.rlbot_delay
    return StepOffset(offset=-1 if delay else 0, relative_to=StepOffsetRelativeTo.ACTION_END)


class BotCoreBot(RLGymBot):
    def __init__(self, bundle_dir: str | Path, device: str = "cpu", deterministic: bool = False,
                 live_dir: Optional[str | Path] = None):
        self.zero_index = zero_action_index()
        self.bundle: Optional[Bundle] = None
        self.policy: Optional[PolicyRunner] = None
        self.load_error: Optional[str] = None
        try:
            self.bundle = load_bundle(bundle_dir, device=device)
            self.policy = PolicyRunner(self.bundle, device=device, deterministic=deterministic)
        except Exception:  # noqa: BLE001 - reported below and in every status report
            self.load_error = traceback.format_exc()
        action_repeat = self.bundle.action_repeat if self.bundle else 8
        super().__init__(
            obs_builder=SpecObsBuilder(),
            action_parser=LookupActionParser(action_repeat),
            default_agent_id=DEFAULT_AGENT_ID,
            config=RLGymBotConfig(step_offset=step_offset_for(self.bundle)),
        )
        self.device = device
        self.deterministic = deterministic
        self.pad_tracker: Optional[PadTimerTracker] = None
        self.latency = LatencyStats(deadline_ms=6.0)
        self.policy_faults = 0
        self.fallback_ticks = 0
        self.decisions = 0
        self.env_resets = 0
        self.unexpected_resets = 0
        self.overtime_kickoff_resets = 0
        self._last_report = time.perf_counter()
        self.live_dir = Path(live_dir or os.environ.get(LIVE_DIR_ENV, "")) if (live_dir or os.environ.get(LIVE_DIR_ENV)) else None
        self.recorder = None
        self._log_handler: Optional[logging.Handler] = None

    # ------------------------------------------------------------------ lifecycle
    def _start_live_diagnostics(self) -> None:
        from botcore.deploy.recording import PacketRecorder

        self.live_dir.mkdir(parents=True, exist_ok=True)
        stem = self.live_dir / f"botcore-team{self.team}-index{self.index}"
        self._log_handler = logging.FileHandler(f"{stem}.log", encoding="utf-8")
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        for logger in {self.logger, DEFAULT_LOGGER}:
            logger.addHandler(self._log_handler)
        self.recorder = PacketRecorder(f"{stem}.bcp")
        self.recorder.field_info(self.field_info)
        self.recorder.meta({"team": self.team, "index": self.index, "player_id": self.player_id, "name": self.name,
                            "bundle": None if self.bundle is None else self.bundle.manifest["weights_sha256"],
                            "device": self.device, "deterministic": self.deterministic})
        self.logger.info("live diagnostics -> %s.{log,bcp,summary.json}", stem)

    def initialize(self):
        if self.live_dir is not None:
            try:
                self._start_live_diagnostics()
            except Exception:  # noqa: BLE001
                self.logger.error("could not start live diagnostics:\n%s", traceback.format_exc())
        if self.load_error:
            self.logger.critical("BotCore could not load its policy bundle; it will send ZERO actions only.\n%s",
                                 self.load_error)
        else:
            m = self.bundle.manifest
            self.logger.info("BotCore loaded %s (weights %s, %s, action_repeat=%d, device=%s, %s), team %d index %d",
                             self.bundle.path, self.bundle.identity, m["contract"]["obs_spec_version"],
                             self.bundle.action_repeat, self.device, "argmax" if self.deterministic else "sampled",
                             self.team, self.index)
        try:
            self.pad_tracker = PadTimerTracker(self.field_info.boost_pads)
        except Exception:  # noqa: BLE001
            self.logger.error("Boost pad layout not supported; pad timers will come from the packet as-is.\n%s",
                              traceback.format_exc())
        n_players = len(self.match_config.player_configurations)
        if n_players > 6:
            self.logger.critical("%d players: the obs-v1 contract supports at most 3v3", n_players)

    def summary(self) -> dict:
        return {"latency_full_callback": self.latency.summary(), "decisions": self.decisions,
                "policy_faults": self.policy_faults, "fallback_ticks": self.fallback_ticks,
                "env_resets": self.env_resets, "unexpected_resets": self.unexpected_resets,
                "overtime_kickoff_resets": self.overtime_kickoff_resets,
                "packets_recorded": None if self.recorder is None else self.recorder.count,
                "load_error": self.load_error is not None}

    def retire(self):
        summary = self.summary()
        self.logger.info("BotCore retiring: %s", summary)
        if self.recorder is not None:
            self.recorder.close()
            stem = self.recorder.path.with_suffix("")
            Path(f"{stem}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if self._log_handler is not None:
            self._log_handler.close()

    # ------------------------------------------------------------------ packet hooks
    def _handle_packet(self, packet: flat.GamePacket):
        try:
            if self.recorder is not None:
                self.recorder.packet(packet)
            if self.pad_tracker is not None:
                self.pad_tracker.update(packet)
                self.shared_info[PAD_TIMERS_KEY] = self.pad_tracker.timers()
            self.shared_info[FLIP_EXPIRED_KEY] = flip_expired_ids(packet)
        except Exception:  # noqa: BLE001
            self.logger.error("state reconstruction failed for this packet:\n%s", traceback.format_exc())
        super()._handle_packet(packet)

    def _env_reset(self, packet, game_state, start_tick, set_state=False):
        self.env_resets += 1
        self.logger.info("env reset #%d at tick %d (phase %s)", self.env_resets, start_tick,
                         packet.match_info.match_phase)
        return super()._env_reset(packet, game_state, start_tick, set_state)

    def _unexpected_env_reset(self):
        self.unexpected_resets += 1
        self.logger.warning("on-the-fly env reset #%d", self.unexpected_resets)
        return super()._unexpected_env_reset()

    def _process_unused_packets(self):
        t0 = time.perf_counter()
        last = self._last_packet
        if (last is not None and last.match_info.match_phase in (flat.MatchPhase.Kickoff, flat.MatchPhase.Active)
                and any(p.match_info.match_phase == flat.MatchPhase.Countdown for p in self._unused_packets)):
            # rlgym-rlbot 0.3.0 only starts its kickoff reset when the countdown follows GoalScored/Replay. An
            # overtime kickoff (Active -> Countdown, found in live test 1) would otherwise skip the reset, keep the
            # stale previous action and re-time the last action through the countdown. Present the previous packet
            # as GoalScored so the normal countdown/reset path runs, exactly as in training.
            self.overtime_kickoff_resets += 1
            self.logger.info("countdown after open play (overtime kickoff): forcing the kickoff reset path")
            self._last_packet = flat.GamePacket(
                players=last.players, boost_pads=last.boost_pads, balls=last.balls, teams=last.teams,
                match_info=flat.MatchInfo(frame_num=last.match_info.frame_num,
                                          seconds_elapsed=last.match_info.seconds_elapsed,
                                          match_phase=flat.MatchPhase.GoalScored))
        super()._process_unused_packets()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.latency.add(elapsed_ms)
        latest = self._unused_packets[-1]
        tick = latest.match_info.frame_num
        fallback = False
        if latest.match_info.match_phase in _ACTIVE_PHASES and tick not in self._tick_action_map:
            # rlgym-rlbot would repeat the previous action; a failure here (e.g. an exception while building the
            # observation) is safer handled with a legal neutral input.
            self._tick_action_map[tick] = flat.ControllerState()
            self.fallback_ticks += 1
            fallback = True
            self.logger.warning("no action produced for tick %d; sending a zero ControllerState", tick)
        if self.recorder is not None:
            self.recorder.timing(tick, elapsed_ms, len(self._unused_packets), fallback)
        now = time.perf_counter()
        if now - self._last_report > REPORT_EVERY_SECONDS:
            self._last_report = now
            self.logger.info("BotCore status: %s", self.summary())

    # ------------------------------------------------------------------ policy
    def get_action(self, obs, game_state, packet):
        t0 = time.perf_counter()
        if self.policy is None:
            action = self.zero_index
        else:
            try:
                action = self.policy.act(obs)
            except Exception:  # noqa: BLE001
                self.policy_faults += 1
                action = self.zero_index
                if self.policy_faults <= 5 or self.policy_faults % 100 == 0:
                    self.logger.error("policy inference failed (%d so far); zero action used:\n%s",
                                      self.policy_faults, traceback.format_exc())
        self.decisions += 1
        if self.recorder is not None:
            frame = packet.match_info.frame_num if packet is not None else -1
            self.recorder.decision(frame, game_state.tick_count, action, (time.perf_counter() - t0) * 1000.0, obs)
        return np.array([action])


def main(bundle_dir: Optional[str] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    bundle_dir = bundle_dir or os.environ.get("BOTCORE_BUNDLE")
    if bundle_dir is None:
        raise SystemExit("pass the bundle directory or set BOTCORE_BUNDLE")
    BotCoreBot(bundle_dir, device=os.environ.get("BOTCORE_DEVICE", "cpu"),
               deterministic=os.environ.get("BOTCORE_DETERMINISTIC", "0") == "1").run()
