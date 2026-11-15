"""Observation contract: layout, team inversion, presence masks, finiteness."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from botcore import constants as C
from botcore.obs import LAYOUT, OBS_SIZE, UnsupportedPlayerCountError, build_obs
from botcore.obs import spec as S
from tests.helpers import counterpart, make_env, rotated_swapped, stepped_state


def test_layout_blocks_are_contiguous_and_complete():
    assert LAYOUT.self_slice.start == 0
    assert LAYOUT.ball_slice.start == LAYOUT.self_slice.stop
    assert LAYOUT.cars_slice.start == LAYOUT.ball_slice.stop
    assert LAYOUT.pads_slice.start == LAYOUT.cars_slice.stop
    assert LAYOUT.size == OBS_SIZE == S.SELF_SIZE + S.BALL_SIZE + S.N_CAR_SLOTS * S.CAR_SIZE + S.PAD_SIZE
    for feats, idx, size in ((S.SELF_FEATURES, S.SELF_IDX, S.SELF_SIZE), (S.BALL_FEATURES, S.BALL_IDX, S.BALL_SIZE),
                             (S.CAR_FEATURES, S.CAR_IDX, S.CAR_SIZE)):
        covered = np.zeros(size, dtype=int)
        for f in feats:
            covered[idx[f.name]] += 1
            # every feature documents all the fields the contract requires
            assert all([f.units, f.scale, f.frame, f.deploy, f.missing, f.reset]), f.name
        assert (covered == 1).all()


def test_spec_hash_is_stable_and_sensitive():
    assert S.spec_hash() == S.OBS_SPEC_HASH
    altered = S.spec_dict()
    altered["scales"]["pos"] = 2300.0
    import hashlib, json
    other = hashlib.sha256(json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert other != S.OBS_SPEC_HASH


@pytest.mark.parametrize("team_size", [1, 2, 3])
def test_every_index_is_written(team_size):
    _, state = stepped_state(team_size, seed=3)
    for agent in state.cars:
        obs = build_obs(state, agent, np.ones(8, dtype=np.float32), fill=np.nan)
        assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
        assert np.isfinite(obs).all(), "an index was never written or is not finite"


@pytest.mark.parametrize("team_size", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_team_inversion_parity(team_size, seed):
    """Blue's observation of a situation equals orange's observation of the rotated, colour-swapped situation."""
    _, state = stepped_state(team_size, seed=seed)
    # make pad timers asymmetric so the pad permutation is actually exercised
    state.boost_pad_timers = np.linspace(0.0, 9.0, C.NUM_BOOST_PADS).astype(np.float32)
    state._inverted_boost_pad_timers = None
    mirrored = rotated_swapped(state)
    rng = np.random.default_rng(seed)
    for agent in state.cars:
        prev = rng.uniform(-1, 1, size=8).astype(np.float32)
        a = build_obs(state, agent, prev)
        b = build_obs(mirrored, counterpart(agent), prev)
        np.testing.assert_allclose(a, b, atol=1e-6, err_msg=f"{agent} vs {counterpart(agent)}")


def test_orange_view_is_a_rotation_not_a_reflection():
    """Angular velocity must transform like an ordinary vector under the 180 degree rotation."""
    _, state = stepped_state(1, seed=5)
    blue, orange = "blue-0", "orange-0"
    car = state.cars[orange]
    car.physics.angular_velocity = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    car._inverted_physics = None
    obs = build_obs(state, orange)
    ang = obs[LAYOUT.self_slice][S.SELF_IDX["ang_vel"]] * S.CAR_ANG_VEL_SCALE
    np.testing.assert_allclose(ang, [-1.0, -2.0, 3.0], atol=1e-6)
    # local-frame quantities are invariant under the team rotation
    blue_view = build_obs(state, blue)
    local = S.SELF_IDX["ang_vel_local"]
    world_ang = np.asarray(state.cars[blue].physics.angular_velocity)
    rot = np.asarray(state.cars[blue].physics.rotation_mtx)
    np.testing.assert_allclose(blue_view[LAYOUT.self_slice][local] * S.CAR_ANG_VEL_SCALE, rot.T @ world_ang, atol=1e-5)


def test_presence_masks_1v1_and_2v2():
    present = S.CAR_IDX["present"]
    mate = S.CAR_IDX["is_teammate"]
    _, state = stepped_state(1, seed=1)
    obs = build_obs(state, "blue-0")
    slots = obs[LAYOUT.cars_slice].reshape(S.N_CAR_SLOTS, S.CAR_SIZE)
    assert slots[:, present].ravel().tolist() == [0, 0, 1, 0, 0]
    assert slots[2, mate].item() == 0.0
    for empty in (0, 1, 3, 4):
        assert not slots[empty].any(), "absent slots must be all zeros"

    _, state = stepped_state(2, seed=1)
    obs = build_obs(state, "orange-1")
    slots = obs[LAYOUT.cars_slice].reshape(S.N_CAR_SLOTS, S.CAR_SIZE)
    assert slots[:, present].ravel().tolist() == [1, 0, 1, 1, 0]
    assert slots[:, mate].ravel().tolist() == [1, 0, 0, 0, 0]


def test_no_nan_for_coincident_positions_and_empty_entity_set():
    _, state = stepped_state(1, seed=2)
    me = state.cars["blue-0"]
    # ball exactly on the car, opponent exactly on the car
    state.ball.position = np.array(me.physics.position, dtype=np.float32)
    state._inverted_ball = None
    other = state.cars["orange-0"]
    other.physics.position = np.array(me.physics.position, dtype=np.float32)
    other._inverted_physics = None
    for agent in state.cars:
        assert np.isfinite(build_obs(state, agent)).all()
    # no other cars at all: every slot is masked
    solo = copy.deepcopy(state)
    del solo.cars["orange-0"]
    obs = build_obs(solo, "blue-0")
    assert np.isfinite(obs).all()
    assert not obs[LAYOUT.cars_slice].any()


def test_demolished_cars_are_flagged_and_zeroed():
    _, state = stepped_state(2, seed=4)
    state.cars["orange-0"].demo_respawn_timer = 2.0
    obs = build_obs(state, "blue-0")
    slot = obs[LAYOUT.car_slot_slice(S.N_TEAMMATE_SLOTS)]
    assert slot[S.CAR_IDX["present"]] == 1 and slot[S.CAR_IDX["is_demoed"]] == 1
    for key in ("pos", "forward", "up", "lin_vel", "ang_vel", "rel_pos", "rel_vel", "rel_pos_local", "distance",
                "ball_rel_pos", "ball_distance"):
        assert not slot[S.CAR_IDX[key]].any(), key
    # demolished self: flags set, kinematics and relative ball features zero, still finite
    state.cars["blue-0"].demo_respawn_timer = 1.5
    obs = build_obs(state, "blue-0")
    me = obs[LAYOUT.self_slice]
    assert me[S.SELF_IDX["is_demoed"]] == 1 and me[S.SELF_IDX["demo_timer"]] == pytest.approx(0.5)
    assert not me[S.SELF_IDX["pos"]].any() and not obs[LAYOUT.ball_slice][S.BALL_IDX["rel_pos"]].any()
    assert np.isfinite(obs).all()


def test_too_many_players_is_a_clear_error():
    _, state = stepped_state(3, seed=0)
    extra = copy.deepcopy(state.cars["orange-0"])
    state.cars["orange-3"] = extra
    with pytest.raises(UnsupportedPlayerCountError, match="at most 3 opponents"):
        build_obs(state, "blue-0")


def test_wrong_pad_count_is_rejected():
    _, state = stepped_state(1, seed=0)
    with pytest.raises(ValueError, match="34 boost pads"):
        build_obs(state, "blue-0", pad_timers=np.zeros(10))


def test_previous_action_is_exposed_and_reset_at_episode_start():
    env = make_env(1, seed=7)
    obs = env.reset()
    prev_idx = S.SELF_IDX["prev_action"]
    for o in obs.values():
        assert not o[LAYOUT.self_slice][prev_idx].any(), "previous action must be zero right after a reset"
    table = env.action_parser.table.controls
    actions = {"blue-0": np.array([40]), "orange-0": np.array([77])}
    obs, *_ = env.step(actions)
    np.testing.assert_array_equal(obs["blue-0"][LAYOUT.self_slice][prev_idx], table[40])
    np.testing.assert_array_equal(obs["orange-0"][LAYOUT.self_slice][prev_idx], table[77])
    obs = env.reset()
    for o in obs.values():
        assert not o[LAYOUT.self_slice][prev_idx].any(), "previous action leaked across an episode boundary"
