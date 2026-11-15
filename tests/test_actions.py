"""Action table identity, parser mapping and jump press/release semantics."""

from __future__ import annotations

import numpy as np
import pytest

from botcore import constants as C
from botcore.actions import (
    ACTION_TABLE_ID,
    ACTION_TABLE_SHA256,
    LookupActionParser,
    audit_table,
    get_action_table,
    table_sha256,
)
from botcore.obs import PREV_ACTIONS_KEY
from tests.helpers import make_env


def _index(controls, **want) -> int:
    """Index of the unique table row matching the given control values (unspecified controls must be 0)."""
    target = np.zeros(C.NUM_CONTROLS, dtype=np.float32)
    for name, value in want.items():
        target[C.CONTROL_NAMES.index(name)] = value
    matches = np.nonzero((controls == target).all(axis=1))[0]
    assert len(matches) == 1, f"{want} matched {len(matches)} rows"
    return int(matches[0])


def test_table_identity_is_pinned():
    table = get_action_table()
    assert table.table_id == ACTION_TABLE_ID
    assert table.sha256 == ACTION_TABLE_SHA256 == table_sha256(table.controls)
    assert table.controls.shape == (90, 8)
    assert not table.controls.flags.writeable


def test_table_values_are_legal_controls():
    controls = get_action_table().controls
    assert np.isin(controls[:, :5], (-1.0, 0.0, 1.0)).all()
    assert np.isin(controls[:, 5:], (0.0, 1.0)).all()
    assert len({tuple(r) for r in controls.tolist()}) == 90, "duplicate actions waste policy capacity"


def test_table_audit_documents_known_omissions():
    audit = audit_table(get_action_table().controls)
    assert audit["analog_values"] == [-1.0, 0.0, 1.0]  # no partial steering/throttle
    assert audit["jump_with_yaw"] is False  # no stalls / yaw-directed dodges
    assert audit["jump_with_steer"] is False
    assert audit["boost_without_throttle"] is False
    assert audit["throttle_differs_from_boost_in_air"] is False  # air throttle is tied to boost


def test_parser_maps_index_to_repeated_controls_and_records_previous_action():
    parser = LookupActionParser(action_repeat=8)
    shared = {}
    parser.reset(["a", "b"], None, shared)
    assert shared[PREV_ACTIONS_KEY] == {}
    out = parser.parse_actions({"a": np.array([5]), "b": np.array([[89]])}, None, shared)
    table = parser.table.controls
    assert out["a"].shape == (8, 8) and out["b"].shape == (8, 8)
    assert (out["a"] == table[5]).all() and (out["b"] == table[89]).all()
    np.testing.assert_array_equal(shared[PREV_ACTIONS_KEY]["a"], table[5])
    parser.reset(["a", "b"], None, shared)
    assert shared[PREV_ACTIONS_KEY] == {}


@pytest.mark.parametrize("repeat", [1, 4, 8, 12])
def test_action_repeat_is_configurable(repeat):
    parser = LookupActionParser(action_repeat=repeat)
    out = parser.parse_actions({"a": np.array([10])}, None, {})
    assert out["a"].shape == (repeat, 8)


def test_out_of_range_index_is_rejected():
    parser = LookupActionParser()
    with pytest.raises(ValueError, match="outside the 90-entry table"):
        parser.parse_actions({"a": np.array([90])}, None, {})
    with pytest.raises(ValueError, match="exactly one action index"):
        parser.parse_actions({"a": np.array([1, 2])}, None, {})


def test_env_advances_action_repeat_ticks_per_decision():
    env = make_env(1, seed=0, action_repeat=8)
    env.reset()
    t0 = env.state.tick_count
    env.step({a: np.array([0]) for a in env.agents})
    assert env.state.tick_count - t0 == 8


def test_jump_needs_a_release_before_the_second_press():
    """Holding jump across two decisions must not double jump; press-release-press must."""
    controls = get_action_table().controls
    idle = _index(controls)
    jump = _index(controls, jump=1)

    def run(sequence):
        env = make_env(1, seed=0, scenario_weights={"kickoff": 1.0})
        env.reset()
        for idx in sequence:
            env.step({"blue-0": np.array([idx]), "orange-0": np.array([idle])})
        return env.state.cars["blue-0"]

    held = run([jump, jump, jump])
    assert held.has_jumped and not held.has_double_jumped and not held.has_flipped
    pressed_twice = run([jump, idle, jump])
    assert pressed_twice.has_jumped and pressed_twice.has_double_jumped


def test_dodge_is_expressible_and_consumes_the_flip():
    controls = get_action_table().controls
    idle = _index(controls)
    jump = _index(controls, jump=1)
    front_flip = _index(controls, pitch=-1, jump=1, handbrake=1)
    env = make_env(1, seed=0, scenario_weights={"kickoff": 1.0})
    env.reset()
    for idx in (jump, idle, front_flip):
        env.step({"blue-0": np.array([idx]), "orange-0": np.array([idle])})
    car = env.state.cars["blue-0"]
    assert car.has_flipped and not car.has_flip
