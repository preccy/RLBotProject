"""RLBot v5 runner: an exported bundle driven through RLGym's rlgym-rlbot wrapper.

rlgym-rlbot 0.3.0 (RLGymBot) does the timing work: it rebuilds an RLGym-compatible GameState from every
GamePacket (rlgym-compat), steps the same ObsBuilder/ActionParser objects used in training on the same
tick schedule (StepOffset -1 relative to ACTION_END matches RocketSimEngine(rlbot_delay=True)), resets
them at kickoffs, and submits one ControllerState per physics tick. This module adds:

* bundle loading with a hard contract check, and a logged zero-action fallback if loading or inference
  fails (never a silent substitute policy);
* reconstructed boost pad timers and flip-window expiry (botcore.deploy.state_fixes);
* full-callback latency instrumentation with a 6 ms deadline counter.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from rlbot import flat
from rlgym_rlbot import RLGymBot, RLGymBotConfig
from rlgym_rlbot.rlgym_bot import StepOffset, StepOffsetRelativeTo

from botcore.actions import LookupActionParser, get_action_table
from botcore.deploy.state_fixes import LatencyStats, PadTimerTracker, flip_expired_ids
from botcore.export import Bundle, PolicyRunner, load_bundle
from botcore.obs import FLIP_EXPIRED_KEY, PAD_TIMERS_KEY, SpecObsBuilder

DEFAULT_AGENT_ID = "local/botcore"
REPORT_EVERY_SECONDS = 30.0
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
    def __init__(self, bundle_dir: str | Path, device: str = "cpu", deterministic: bool = False):
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
        self._last_report = time.perf_counter()
        self.recorder = None
        record_path = os.environ.get("BOTCORE_RECORD")
        if record_path:
            from botcore.deploy.recording import PacketRecorder

            self.recorder = PacketRecorder(record_path)

    # ------------------------------------------------------------------ lifecycle
    def initialize(self):
        if self.load_error:
            self.logger.critical("BotCore could not load its policy bundle; it will send ZERO actions only.\n%s",
                                 self.load_error)
        else:
            m = self.bundle.manifest
            self.logger.info("BotCore loaded %s (weights %s, %s, action_repeat=%d, device=%s, %s)", self.bundle.path,
                             self.bundle.identity, m["contract"]["obs_spec_version"], self.bundle.action_repeat,
                             self.device, "argmax" if self.deterministic else "sampled")
        if self.recorder is not None:
            self.recorder.field_info(self.field_info)
            self.logger.info("recording packets to %s", self.recorder.path)
        try:
            self.pad_tracker = PadTimerTracker(self.field_info.boost_pads)
        except Exception:  # noqa: BLE001
            self.logger.error("Boost pad layout not supported; pad timers will come from the packet as-is.\n%s",
                              traceback.format_exc())
        n_players = len(self.match_config.player_configurations)
        if n_players > 6:
            self.logger.critical("%d players: the obs-v1 contract supports at most 3v3", n_players)

    def retire(self):
        self.logger.info("BotCore retiring. latency=%s faults=%d fallback_ticks=%d", self.latency.summary(),
                         self.policy_faults, self.fallback_ticks)
        if self.recorder is not None:
            self.recorder.close()
            self.logger.info("recorded %d packets to %s", self.recorder.count, self.recorder.path)

    # ------------------------------------------------------------------ packet hooks
    def _handle_packet(self, packet: flat.GamePacket):
        try:
            if self.recorder is not None and self._initialized_bot:
                self.recorder.packet(packet)
            if self.pad_tracker is not None:
                self.pad_tracker.update(packet)
                self.shared_info[PAD_TIMERS_KEY] = self.pad_tracker.timers()
            self.shared_info[FLIP_EXPIRED_KEY] = flip_expired_ids(packet)
        except Exception:  # noqa: BLE001
            self.logger.error("state reconstruction failed for this packet:\n%s", traceback.format_exc())
        super()._handle_packet(packet)

    def _process_unused_packets(self):
        t0 = time.perf_counter()
        super()._process_unused_packets()
        self.latency.add((time.perf_counter() - t0) * 1000.0)
        latest = self._unused_packets[-1]
        tick = latest.match_info.frame_num
        if latest.match_info.match_phase in _ACTIVE_PHASES and tick not in self._tick_action_map:
            # rlgym-rlbot would repeat the previous action; a failure here (e.g. an exception while building the
            # observation) is safer handled with a legal neutral input.
            self._tick_action_map[tick] = flat.ControllerState()
            self.fallback_ticks += 1
            self.logger.warning("no action produced for tick %d; sending a zero ControllerState", tick)
        now = time.perf_counter()
        if now - self._last_report > REPORT_EVERY_SECONDS:
            self._last_report = now
            self.logger.info("BotCore status: latency=%s faults=%d fallback_ticks=%d", self.latency.summary(),
                             self.policy_faults, self.fallback_ticks)

    # ------------------------------------------------------------------ policy
    def get_action(self, obs, game_state, packet):
        if self.policy is None:
            return np.array([self.zero_index])
        try:
            return np.array([self.policy.act(obs)])
        except Exception:  # noqa: BLE001
            self.policy_faults += 1
            if self.policy_faults <= 5 or self.policy_faults % 100 == 0:
                self.logger.error("policy inference failed (%d so far); zero action used:\n%s", self.policy_faults,
                                  traceback.format_exc())
            return np.array([self.zero_index])


def main(bundle_dir: Optional[str] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    here = Path(__file__).resolve()
    bundle_dir = bundle_dir or os.environ.get("BOTCORE_BUNDLE")
    if bundle_dir is None:
        raise SystemExit("pass the bundle directory or set BOTCORE_BUNDLE")
    BotCoreBot(bundle_dir, device=os.environ.get("BOTCORE_DEVICE", "cpu"),
               deterministic=os.environ.get("BOTCORE_DETERMINISTIC", "0") == "1").run()
