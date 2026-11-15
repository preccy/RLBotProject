"""Versioned discrete action table and RLGym v2 action parser.

The table is RLGym's established 90-entry lookup table
(`rlgym.rocket_league.action_parsers.LookupTableAction.make_lookup_table`, rlgym-rocket-league
2.0.1, Apache-2.0, originally by Rolv-Arild Braaten for Necto). It is generated from the installed
package and checked against a pinned SHA-256, so an upstream change can never silently remap what
action index 37 means for an already-trained policy.

Columns: throttle, steer, pitch, yaw, roll, jump, boost, handbrake.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import ActionParser
from rlgym.rocket_league.action_parsers import LookupTableAction

from botcore import constants as C
from botcore.obs.builder import PREV_ACTIONS_KEY

ACTION_TABLE_ID = "rlgym-lookup-90-v1"
# sha256 of the table as little-endian float32, C order. Pinned on 2026-11-14 from rlgym-rocket-league 2.0.1.
ACTION_TABLE_SHA256 = "86baa15c48c42c497f3ea0fe62efeb49e4a8241cb3191957822e453cd2d0b655"


class ActionTableMismatchError(RuntimeError):
    """The installed RLGym lookup table differs from the one this project was built against."""


def table_sha256(table: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(table, dtype="<f4").tobytes()).hexdigest()


def _generate_table() -> np.ndarray:
    return np.asarray(LookupTableAction.make_lookup_table(), dtype=np.float32)


@dataclass(frozen=True)
class ActionTable:
    table_id: str
    sha256: str
    controls: np.ndarray  # (n_actions, 8) float32, read-only

    @property
    def n_actions(self) -> int:
        return int(self.controls.shape[0])


_TABLE: ActionTable | None = None


def get_action_table() -> ActionTable:
    """Return the verified action table (cached)."""
    global _TABLE
    if _TABLE is None:
        controls = _generate_table()
        digest = table_sha256(controls)
        if digest != ACTION_TABLE_SHA256:
            raise ActionTableMismatchError(
                f"Action table {ACTION_TABLE_ID} was pinned with sha256 {ACTION_TABLE_SHA256} but the installed "
                f"rlgym-rocket-league generates {digest} ({controls.shape[0]} actions). Policies trained with "
                "the pinned table would be mapped to different controls. Reinstall the pinned RLGym version "
                "(see requirements-lock.txt) or deliberately introduce a new table id."
            )
        controls.setflags(write=False)
        _TABLE = ActionTable(ACTION_TABLE_ID, digest, controls)
    return _TABLE


class LookupActionParser(ActionParser[Any, np.ndarray, np.ndarray, Any, Tuple[str, int]]):
    """Maps a discrete action index to `action_repeat` ticks of identical controls.

    It also records the controls each agent just committed to in ``shared_info['prev_actions']`` so the
    observation builder can expose the previous action. The record is cleared on every reset, which is
    what makes "previous action" well defined at episode boundaries in both the simulator and RLBot.
    """

    def __init__(self, action_repeat: int = 8):
        super().__init__()
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        self.table = get_action_table()
        self.action_repeat = int(action_repeat)
        self._repeated = np.repeat(self.table.controls[:, None, :], self.action_repeat, axis=1)
        self._repeated.setflags(write=False)

    def get_action_space(self, agent) -> Tuple[str, int]:
        return "discrete", self.table.n_actions

    def reset(self, agents: List[Any], initial_state, shared_info: Dict[str, Any]) -> None:
        shared_info[PREV_ACTIONS_KEY] = {}

    def parse_actions(self, actions: Dict[Any, np.ndarray], state, shared_info: Dict[str, Any]) -> Dict[Any, np.ndarray]:
        prev = shared_info.setdefault(PREV_ACTIONS_KEY, {})
        parsed: Dict[Any, np.ndarray] = {}
        n = self.table.n_actions
        for agent, action in actions.items():
            flat = np.asarray(action).reshape(-1)
            if flat.shape[0] != 1:
                raise ValueError(f"expected exactly one action index per agent, got shape {np.shape(action)}")
            idx = int(flat[0])
            if not 0 <= idx < n:
                raise ValueError(f"action index {idx} outside the {n}-entry table {self.table.table_id}")
            prev[agent] = self.table.controls[idx]
            parsed[agent] = self._repeated[idx]
        return parsed


def audit_table(controls: np.ndarray) -> dict:
    """Summarise what the table can and cannot express (used by tests and docs)."""
    rows = {tuple(r) for r in controls.tolist()}
    ground = controls[(controls[:, C.JUMP] == 0) & (controls[:, C.PITCH] == 0) & (controls[:, C.ROLL] == 0)]
    jump = controls[controls[:, C.JUMP] == 1]
    return {
        "n_actions": int(controls.shape[0]),
        "n_unique": len(rows),
        "analog_values": sorted({float(v) for v in np.unique(controls[:, :5])}),
        "n_jump_actions": int(jump.shape[0]),
        "jump_with_yaw": bool(np.any(jump[:, C.YAW] != 0)),
        "jump_with_steer": bool(np.any(jump[:, C.STEER] != 0)),
        "boost_without_throttle": bool(np.any((controls[:, C.BOOST] == 1) & (controls[:, C.THROTTLE] != 1))),
        "ground_handbrake_actions": int(np.sum(ground[:, C.HANDBRAKE] == 1)),
        "air_roll_and_yaw_together": bool(np.any((controls[:, C.ROLL] != 0) & (controls[:, C.YAW] != 0))),
        "throttle_differs_from_boost_in_air": bool(
            np.any((controls[:, C.PITCH] != 0) & (controls[:, C.THROTTLE] != controls[:, C.BOOST]))
        ),
    }
