"""Deployment-side state reconstruction for quantities the RLBot packet reports differently from RocketSim.

These functions only read `rlbot.flat` packets and field info; they are unit-tested with synthetic packets
(tests/test_parity.py) because a live game is not available in automated tests.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from botcore import constants as C

TICKS_PER_SECOND = C.TICKS_PER_SECOND


class FieldMismatchError(RuntimeError):
    """The map does not have the standard soccar boost pad layout the policy was trained on."""


def pad_order_mapping(field_pads: Iterable, reference=None) -> np.ndarray:
    """Map RLBot field_info pad index -> BOOST_LOCATIONS index (same 2 uu xy matching as rlgym-compat)."""
    from rlgym.rocket_league.common_values import BOOST_LOCATIONS

    ref = np.asarray(reference if reference is not None else BOOST_LOCATIONS, dtype=np.float64)
    pads = list(field_pads)
    if len(pads) != C.NUM_BOOST_PADS:
        raise FieldMismatchError(f"map has {len(pads)} boost pads; the policy was trained on the standard "
                                 f"{C.NUM_BOOST_PADS}-pad soccar layout")
    mapping = np.empty(len(pads), dtype=np.int64)
    for i, pad in enumerate(pads):
        loc = np.array([pad.location.x, pad.location.y])
        close = np.isclose(ref[:, :2], loc, atol=2.0).all(axis=1)
        if not close.any():
            raise FieldMismatchError(f"boost pad at {loc} does not match the standard soccar layout")
        mapping[i] = int(np.argmax(close))
        big = mapping[i] in C.BIG_PAD_INDICES
        if bool(pad.is_full_boost) != big:
            raise FieldMismatchError(f"boost pad at {loc} size mismatch (packet full={pad.is_full_boost})")
    if len(set(mapping.tolist())) != len(mapping):
        raise FieldMismatchError("two RLBot pads mapped onto the same standard pad")
    return mapping


class PadTimerTracker:
    """Seconds until each pad is available, in BOOST_LOCATIONS order, reconstructed from `is_active` edges.

    The meaning of `BoostPadState.timer` changed in the RLBot schema documentation (2026-10-04: "seconds
    remaining", previously "seconds since pickup"). Edges of `is_active` plus the fixed 4 s / 10 s cooldowns
    are unambiguous, so they are used whenever a pickup is observed. Only a pad that is already inactive when
    the tracker first sees it falls back to the packet timer (clipped to the cooldown).
    """

    def __init__(self, field_pads: Iterable):
        pads = list(field_pads)
        self.mapping = pad_order_mapping(pads)
        self.cooldown = np.array([C.BIG_PAD_RECHARGE_SECONDS if p.is_full_boost else C.SMALL_PAD_RECHARGE_SECONDS
                                  for p in pads], dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.expiry = np.zeros(len(self.mapping), dtype=np.float64)  # game time (s) the pad becomes active
        self.was_active: Optional[np.ndarray] = None
        self.last_frame: Optional[int] = None
        self.now = 0.0

    def update(self, packet) -> None:
        frame = int(packet.match_info.frame_num)
        if self.last_frame is not None and frame < self.last_frame:
            self.reset()  # new match / frame counter restarted
        self.last_frame = frame
        self.now = frame / TICKS_PER_SECOND
        states = list(packet.boost_pads)
        if len(states) != len(self.mapping):
            return  # malformed or different map; keep previous estimate
        active = np.array([bool(s.is_active) for s in states])
        for i, state in enumerate(states):
            if active[i]:
                self.expiry[i] = self.now
            elif self.was_active is None or not np.isfinite(self.expiry[i]):
                self.expiry[i] = self.now + float(np.clip(state.timer, 0.0, self.cooldown[i]))
            elif self.was_active[i]:
                self.expiry[i] = self.now + self.cooldown[i]  # picked up since the previous packet
            else:
                # Still inactive past the predicted expiry: it reactivated and was collected again between two
                # packets (e.g. a car parked on the pad), so a new cooldown started at the old expiry.
                while self.expiry[i] <= self.now + 1e-9:
                    self.expiry[i] += self.cooldown[i]
        self.was_active = active

    def timers(self) -> np.ndarray:
        """Remaining seconds per pad in BOOST_LOCATIONS order."""
        remaining = np.clip(self.expiry - self.now, 0.0, self.cooldown)
        out = np.zeros(C.NUM_BOOST_PADS, dtype=np.float32)
        out[self.mapping] = remaining
        return out


def flip_expired_ids(packet, agent_id_of=lambda player: player.player_id) -> Set:
    """Players that jumped, are in free flight, and whose dodge window the game reports as expired."""
    from rlbot.flat import AirState

    out = set()
    for player in packet.players:
        if (player.has_jumped and player.dodge_timeout == -1 and player.air_state == AirState.InAir
                and not player.has_dodged and not player.has_double_jumped):
            out.add(agent_id_of(player))
    return out


class LatencyStats:
    """Rolling statistics of per-decision processing time, with a deadline-miss counter."""

    def __init__(self, deadline_ms: float = 6.0, window: int = 2000):
        self.deadline_ms = deadline_ms
        self.window = window
        self.samples: List[float] = []
        self.total = 0
        self.missed = 0

    def add(self, ms: float) -> None:
        self.total += 1
        if ms > self.deadline_ms:
            self.missed += 1
        self.samples.append(ms)
        if len(self.samples) > self.window:
            del self.samples[: len(self.samples) - self.window]

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"count": 0}
        arr = np.asarray(self.samples)
        return {"count": self.total, "p50_ms": float(np.percentile(arr, 50)), "p99_ms": float(np.percentile(arr, 99)),
                "max_ms": float(arr.max()), "missed_deadline": self.missed, "deadline_ms": self.deadline_ms}
