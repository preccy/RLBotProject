import numpy as np
import pytest

from botcore.actions.table import LookupActionParser
from botcore.collect import EXTERNAL, LEARNER, Collector, external_envs, role_map, run_rollout
from botcore.config import Config, ConfigError, config_from_dict, learner_agent_slots, n_necto_envs, validate
from botcore.necto import necto_dir
from botcore.trainer import config_diff

AGENTS = ["blue-0", "orange-0"]


def test_parser_accepts_raw_controls_only_when_enabled():
    raw = np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    strict = LookupActionParser(8)
    with pytest.raises(ValueError):
        strict.parse_actions({"orange-0": raw}, None, {})
    loose = LookupActionParser(8, allow_raw_controls=True)
    shared = {}
    out = loose.parse_actions({"orange-0": raw, "blue-0": np.array([3])}, None, shared)
    assert out["orange-0"].shape == (8, 8) and np.all(out["orange-0"] == raw)
    assert np.all(out["blue-0"][0] == loose.table.controls[3])
    assert np.all(shared["prev_actions"]["orange-0"] == raw)
    with pytest.raises(ValueError):  # learner indices are still range-checked
        loose.parse_actions({"blue-0": np.array([999])}, None, {})


def test_external_envs_spread_alternate_and_leave_one_learner_car():
    ext = external_envs(78, 12)
    assert len(ext) == 12 and list(ext.values())[:4] == ["orange", "blue", "orange", "blue"]
    roles = role_map(78, AGENTS, 0, ext)
    assert (roles == EXTERNAL).sum() == 12 and (roles == LEARNER).sum() == 144
    for i, team in ext.items():
        assert roles[i, AGENTS.index(f"{team}-0")] == EXTERNAL
    with pytest.raises(ValueError):
        role_map(78, AGENTS, 3, ext)


def test_config_necto_fraction_keeps_learner_steps_and_old_checkpoints_resume():
    cfg = config_from_dict({"collect": {"n_workers": 13, "envs_per_worker": 6},
                            "self_play": {"necto_fraction": 0.15}})
    validate(cfg)
    assert n_necto_envs(cfg) == 12 and learner_agent_slots(cfg) == 144  # = 72 self-play envs x 2
    with pytest.raises(ConfigError):
        validate(config_from_dict({"self_play": {"necto_fraction": 0.15, "frozen_opponent_fraction": 0.1}}))
    old = Config().to_dict()
    for k in ("necto_fraction", "necto_dir"):
        old["self_play"].pop(k)
    assert config_diff(old, Config().to_dict()) == []
    assert any("necto_fraction" in d for d in config_diff(old, cfg.to_dict()))


@pytest.mark.skipif(not (necto_dir() / "necto-model.pt").is_file(), reason="Necto not downloaded (external/Necto)")
def test_collector_runs_necto_in_worker_and_keeps_it_out_of_learner_data():
    from botcore.env import EnvSpec

    spec = EnvSpec(team_size=1, action_repeat=8, rlbot_delay=True, no_touch_timeout_seconds=30.0,
                   episode_timeout_seconds=300.0, scenario_weights={"kickoff": 1.0}, reward_weights={})
    ext = {1: "orange"}
    rng = np.random.default_rng(0)
    with Collector(spec, n_workers=1, envs_per_worker=2, seed=123, external=ext) as col:
        obs = col.reset()
        roles = role_map(2, col.agents, 0, ext)
        act = lambda o: (rng.integers(0, 90, len(o)), np.zeros(len(o), np.float32))
        n = 60
        trajs, obs, timing = run_rollout(col, obs, roles, n, act, None, pipeline=False)
        _, episodes = col.pop_stats()
    assert timing["learner_steps"] == 3 * n  # env 0: two learner cars; env 1: blue only
    assert not any(t.env_index == 1 and t.agent_id.startswith("orange") for t in trajs)
    assert sum(len(t.obs) for t in trajs) == 3 * n
    assert all(e["necto_team"] in (None, "orange") for e in episodes)
