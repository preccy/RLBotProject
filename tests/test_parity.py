"""Simulator vs deployment observation parity on synthetic RLBot packets, and the deployment state fixes."""

from __future__ import annotations

import numpy as np
import pytest
from rlbot import flat

from botcore import constants as C
from botcore.deploy.packets import CompatObsPipeline, field_info, state_to_packet
from botcore.deploy.state_fixes import FieldMismatchError, PadTimerTracker, flip_expired_ids, pad_order_mapping
from botcore.obs import LAYOUT, PREV_ACTIONS_KEY, build_obs
from botcore.obs import spec as S
from tests.helpers import make_env

PAD_ORDER = [int(i) for i in np.random.default_rng(42).permutation(C.NUM_BOOST_PADS)]  # RLBot order differs
PAD_TOL = 8 / 120 / S.PAD_TIMER_SCALE + 1e-6  # pickups are only observed at packets 8 ticks apart in this test
ON_GROUND = LAYOUT.self_slice.start + S.SELF_IDX["on_ground"].start


def _jumping_on_ground_indices(state, agent):
    """Indices of on_ground flags (self and car slots) for cars currently holding a jump."""
    out = []
    me = state.cars[agent]
    if me.is_jumping:
        out.append(ON_GROUND)
    mates = opps = 0
    for other_id, other in state.cars.items():
        if other_id == agent:
            continue
        if other.team_num == me.team_num:
            slot, mates = mates, mates + 1
        else:
            slot, opps = S.N_TEAMMATE_SLOTS + opps, opps + 1
        if other.is_jumping:
            out.append(LAYOUT.car_slot_slice(slot).start + S.CAR_IDX["on_ground"].start)
    return out


def _diff_report(a, b, ok):
    return "; ".join(f"{i}: sim={a[i]:.5f} deploy={b[i]:.5f}" for i in np.nonzero(~ok)[0][:10])


@pytest.mark.parametrize("team_size", [1, 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_observation_parity_over_a_rollout(team_size, seed):
    env = make_env(team_size, seed=seed, scenario_weights={"kickoff": 0.3, "open_play": 0.3, "aerial": 0.2,
                                                           "recovery": 0.2})
    env.reset()
    agents = list(env.agents)
    player_ids = {a: 100 + i for i, a in enumerate(agents)}
    rng = np.random.default_rng(seed)
    frame = 1000

    def fresh_pipeline():
        pipe = CompatObsPipeline(field_info(PAD_ORDER))
        pipe.update(state_to_packet(env.state, frame - 8, player_ids, PAD_ORDER))
        return pipe

    pipe = fresh_pipeline()
    compared = 0
    for step in range(300):
        obs, _, term, trunc = env.step({a: np.array([rng.integers(90)]) for a in agents})
        frame += 8
        if any(term.values()) or any(trunc.values()):
            env.reset()
            frame += 600
            pipe = fresh_pipeline()
            continue
        pipe.update(state_to_packet(env.state, frame, player_ids, PAD_ORDER))
        prev = env.shared_info[PREV_ACTIONS_KEY]
        for agent in agents:
            sim = obs[agent]
            dep = pipe.obs(player_ids[agent], prev[agent])
            tol = np.full(S.OBS_SIZE, 2e-4, dtype=np.float32)
            tol[LAYOUT.pads_slice] = PAD_TOL
            ok = np.abs(sim - dep) <= tol
            # Documented limitation: rlgym-compat infers on_ground during AirState.Jumping with a 6-tick heuristic.
            for idx in _jumping_on_ground_indices(env.state, agent):
                ok[idx] = True
            assert ok.all(), f"step {step} {agent}: {_diff_report(sim, dep, ok)}"
            compared += 1
    assert compared > 200


def test_rotation_round_trip_through_rlbot_euler_angles():
    env = make_env(1, seed=3, scenario_weights={"kickoff": 1e-9, "recovery": 1.0})
    for _ in range(20):
        env.reset()
        pipe = CompatObsPipeline(field_info())
        pipe.update(state_to_packet(env.state, 100, {"blue-0": 0, "orange-0": 1}))
        for agent, pid in (("blue-0", 0), ("orange-0", 1)):
            np.testing.assert_allclose(pipe.state.cars[pid].physics.rotation_mtx,
                                       env.state.cars[agent].physics.rotation_mtx, atol=1e-5)


def test_pad_mapping_handles_rlbot_order_and_rejects_other_maps():
    field = field_info(PAD_ORDER)
    assert pad_order_mapping(field.boost_pads).tolist() == PAD_ORDER
    with pytest.raises(FieldMismatchError):
        pad_order_mapping(field.boost_pads[:-1])
    moved = list(field.boost_pads)
    moved[0] = flat.BoostPad(location=flat.Vector3(x=1.0, y=2.0, z=70.0), is_full_boost=False)
    with pytest.raises(FieldMismatchError):
        pad_order_mapping(moved)


def _pad_packet(frame, active, timers):
    return flat.GamePacket(
        players=[], balls=[],
        boost_pads=[flat.BoostPadState(is_active=bool(a), timer=float(t)) for a, t in zip(active, timers)],
        match_info=flat.MatchInfo(frame_num=frame, match_phase=flat.MatchPhase.Active),
    )


@pytest.mark.parametrize("timer_semantics", ["remaining", "elapsed", "garbage"])
def test_pad_tracker_uses_pickup_edges_not_the_timer_field(timer_semantics):
    tracker = PadTimerTracker(field_info().boost_pads)
    n = C.NUM_BOOST_PADS
    active = np.ones(n, dtype=bool)
    tracker.update(_pad_packet(1200, active, np.zeros(n)))
    assert not tracker.timers().any()
    big, small = C.BIG_PAD_INDICES[0], 0
    active[[big, small]] = False
    for frame in range(1201, 1201 + 240):  # two seconds after both pickups
        elapsed = (frame - 1201) / 120
        field_timer = {"remaining": 10 - elapsed, "elapsed": elapsed, "garbage": 123.0}[timer_semantics]
        tracker.update(_pad_packet(frame, active, np.full(n, field_timer)))
    t = tracker.timers()
    assert t[big] == pytest.approx(10 - 239 / 120, abs=1e-5)
    assert t[small] == pytest.approx(4 - 239 / 120, abs=1e-5)
    assert np.count_nonzero(t) == 2
    active[[big, small]] = True
    tracker.update(_pad_packet(1441, active, np.zeros(n)))
    assert not tracker.timers().any()


def test_pad_tracker_resets_when_the_frame_counter_restarts():
    tracker = PadTimerTracker(field_info().boost_pads)
    n = C.NUM_BOOST_PADS
    active = np.ones(n, dtype=bool)
    tracker.update(_pad_packet(5000, active, np.zeros(n)))
    active[0] = False
    tracker.update(_pad_packet(5001, active, np.full(n, 3.0)))
    tracker.update(_pad_packet(10, np.ones(n, dtype=bool), np.zeros(n)))  # new match
    assert not tracker.timers().any()


def test_flip_expiry_is_detected_from_dodge_timeout():
    def player(**kw):
        base = dict(air_state=flat.AirState.InAir, has_jumped=True, dodge_timeout=-1.0, player_id=7)
        base.update(kw)
        return flat.PlayerInfo(**base)

    packet = flat.GamePacket(players=[player(), player(player_id=8, dodge_timeout=0.4),
                                      player(player_id=9, has_jumped=False),
                                      player(player_id=10, air_state=flat.AirState.OnGround),
                                      player(player_id=11, has_dodged=True),
                                      player(player_id=12, air_state=flat.AirState.Jumping)])
    assert flip_expired_ids(packet) == {7}


def test_stale_compat_air_time_would_report_a_flip_without_the_fix():
    env = make_env(1, seed=0, scenario_weights={"kickoff": 1.0})
    env.reset()
    state = env.state
    car = state.cars["blue-0"]
    car.on_ground = False
    car.has_jumped = True
    car.air_time_since_jump = 1.24  # last value rlgym-compat saw before dodge_timeout became -1
    idx = LAYOUT.self_slice.start + S.SELF_IDX["flip_available"].start
    assert build_obs(state, "blue-0")[idx] == 1.0
    assert build_obs(state, "blue-0", flip_expired={"blue-0"})[idx] == 0.0


def test_packet_recording_round_trip_and_replay(tmp_path):
    import torch

    from botcore.config import Config
    from botcore.deploy.recording import PacketRecorder, read_recording, replay
    from botcore.export import write_bundle
    from botcore.models import arch_from_config, build_actor

    env = make_env(1, seed=4)
    env.reset()
    recorder = PacketRecorder(tmp_path / "match.bcp")
    recorder.field_info(field_info(PAD_ORDER))
    for i in range(60):
        env.step({a: np.array([i % 90]) for a in env.agents})
        recorder.packet(state_to_packet(env.state, 200 + 8 * i, {"blue-0": 3, "orange-0": 9}, PAD_ORDER))
    recorder.close()
    field, packets = read_recording(tmp_path / "match.bcp")
    packets = list(packets)
    assert len(packets) == 60 and len(field.boost_pads) == C.NUM_BOOST_PADS
    assert packets[-1].match_info.frame_num == 200 + 8 * 59 and packets[0].players[1].player_id == 9

    cfg = Config()
    cfg.model.actor_layers = (32,)
    arch = arch_from_config(cfg.model)
    bundle = write_bundle(tmp_path / "b", build_actor(arch, S.OBS_SIZE, 90), arch, 8, True, {})
    report = replay(tmp_path / "match.bcp", bundle, log=lambda *_: None)
    assert report["decisions"] == 60 and report["non_finite_obs"] == 0
