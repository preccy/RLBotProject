"""Checkpoint atomicity/retention, export manifests and compatibility rejection, config validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from botcore import checkpoint as ck
from botcore.config import Config, ConfigError, config_from_dict, learner_agent_slots, load_config
from botcore.export import IncompatibleExportError, PolicyRunner, load_bundle, write_bundle
from botcore.models import arch_from_config, build_actor
from botcore.obs import OBS_SIZE

ROOT = Path(__file__).resolve().parents[1]


def _bundle(tmp_path: Path, name="b", model_kind="mlp", seed=0) -> Path:
    cfg = Config()
    cfg.model.kind = model_kind
    cfg.model.actor_layers = (32, 32)
    cfg.model.embed_dim, cfg.model.n_heads, cfg.model.head_layers = 32, 4, (32,)
    arch = arch_from_config(cfg.model)
    torch.manual_seed(seed)
    actor = build_actor(arch, OBS_SIZE, 90)
    return write_bundle(tmp_path / name, actor, arch, 8, True, {"test": True})


# ------------------------------------------------------------------------------------ checkpoints
def test_failed_save_leaves_nothing_behind(tmp_path):
    def broken(tmp):
        (tmp / "partial.bin").write_bytes(b"x")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        ck.save_checkpoint(tmp_path, 5, broken, {"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_existing_checkpoint_is_never_overwritten(tmp_path):
    ck.save_checkpoint(tmp_path, 1, lambda tmp: None, {"x": 1})
    with pytest.raises(ck.CheckpointError, match="refusing to overwrite"):
        ck.save_checkpoint(tmp_path, 1, lambda tmp: None, {"x": 2})
    assert ck.read_trainer_state(tmp_path / "iter_000001")["x"] == 1


def test_incomplete_checkpoint_is_rejected(tmp_path):
    path = ck.save_checkpoint(tmp_path, 3, lambda tmp: None, {})
    (path / ck.COMPLETE_MARKER).unlink()
    with pytest.raises(ck.CheckpointError, match="incomplete"):
        ck.read_trainer_state(path)
    assert ck.list_checkpoints(tmp_path) == []


def test_retention_keeps_recent_and_protected_checkpoints(tmp_path):
    run = tmp_path / "run"
    cps = run / "checkpoints"
    for i in range(1, 8):
        ck.save_checkpoint(cps, i, lambda tmp: None, {})
    ck.write_pointer(run, "best_verified", "checkpoints/iter_000002")
    (cps / ".iter_000009.tmp-1-2").mkdir()  # leftover from a crash
    removed = ck.prune_checkpoints(cps, keep_last=3, protect=ck.protected_paths(run))
    assert sorted(p.name for p in ck.list_checkpoints(cps)) == ["iter_000002", "iter_000005", "iter_000006", "iter_000007"]
    assert len(removed) == 3 and not (cps / ".iter_000009.tmp-1-2").exists()


def test_rng_state_round_trip():
    state = ck.capture_rng_state()
    a = (np.random.rand(3), torch.rand(3))
    ck.restore_rng_state(state)
    b = (np.random.rand(3), torch.rand(3))
    np.testing.assert_array_equal(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


# ------------------------------------------------------------------------------------ exports
@pytest.mark.parametrize("kind", ["mlp", "entity"])
def test_bundle_round_trip_reproduces_the_policy(tmp_path, kind):
    path = _bundle(tmp_path, model_kind=kind)
    bundle = load_bundle(path)
    torch.manual_seed(0)
    reference = build_actor(bundle.manifest["architecture"], OBS_SIZE, 90)
    reference.load_state_dict(torch.load(path / "policy.pt", weights_only=True))
    x = torch.randn(16, OBS_SIZE)
    with torch.no_grad():
        torch.testing.assert_close(bundle.actor(x), reference(x))
    m = bundle.manifest
    assert m["contract"]["action_repeat"] == 8 and m["contract"]["obs_spec_version"] == "obs-v1"
    assert m["dependencies"]["torch"] and m["dependencies"]["rocketsim"]


def test_deterministic_and_seeded_inference(tmp_path):
    bundle = load_bundle(_bundle(tmp_path))
    obs = np.random.default_rng(0).normal(size=(32, OBS_SIZE)).astype(np.float32)
    a = PolicyRunner(bundle, seed=5).act_batch(obs)
    b = PolicyRunner(bundle, seed=5).act_batch(obs)
    np.testing.assert_array_equal(a, b)
    det = PolicyRunner(bundle, deterministic=True).act_batch(obs)
    with torch.no_grad():
        np.testing.assert_array_equal(det, bundle.actor(torch.as_tensor(obs)).argmax(-1).numpy())


def test_existing_bundle_is_not_overwritten(tmp_path):
    _bundle(tmp_path, "same")
    with pytest.raises(FileExistsError):
        _bundle(tmp_path, "same")


@pytest.mark.parametrize("field,value,match", [
    ("obs_spec_hash", "0" * 64, "observation spec"),
    ("action_table_sha256", "f" * 64, "action table"),
    ("obs_size", 92, "observation size"),
])
def test_contract_mismatch_is_rejected_with_a_clear_error(tmp_path, field, value, match):
    path = _bundle(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["contract"][field] = value
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IncompatibleExportError, match=match):
        load_bundle(path)


def test_tampered_weights_and_wrong_architecture_are_rejected(tmp_path):
    path = _bundle(tmp_path)
    data = bytearray((path / "policy.pt").read_bytes())
    data[-10] ^= 0xFF
    (path / "policy.pt").write_bytes(bytes(data))
    with pytest.raises(IncompatibleExportError, match="sha256"):
        load_bundle(path)

    other = _bundle(tmp_path, "other")
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["architecture"]["actor_layers"] = [64, 64]  # claims a different size than the stored weights
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IncompatibleExportError, match="do not fit"):
        load_bundle(other)


def test_missing_bundle_is_a_clear_error(tmp_path):
    with pytest.raises(IncompatibleExportError, match="not an export bundle"):
        load_bundle(tmp_path / "nothing")


# ------------------------------------------------------------------------------------ config
def test_presets_validate():
    for name in ("default", "modest", "entity", "smoke"):
        load_config(ROOT / "configs" / f"{name}.toml")


@pytest.mark.parametrize("data,match", [
    ({"env": {"team_size": 4}}, "team_size=4 unsupported"),
    ({"env": {"tick_skip": 8}}, "unknown key"),
    ({"bogus": {}}, "unknown config section"),
    ({"model": {"kind": "transformer"}}, "kind must be"),
    ({"model": {"kind": "entity", "embed_dim": 100, "n_heads": 3}}, "divisible"),
    ({"rewards": {"weights": {"style": 1.0}}}, "unknown component"),
    ({"env": {"scenario_weights": {"kickoff": 0.0, "shot": 1.0}}}, "kickoff must keep"),
    ({"ppo": {"batch_size": 10_000_000}}, "exceeds"),
    ({"run": {"seed": 2_000_000}}, "reserved for evaluation"),
    ({"env": {"action_repeat": "8"}}, "must be an integer"),
])
def test_invalid_configs_fail_fast(data, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(data)


def test_learner_slot_count_accounts_for_frozen_opponents():
    cfg = config_from_dict({"collect": {"n_workers": 2, "envs_per_worker": 2, "rollout_steps": 64},
                            "ppo": {"batch_size": 64}, "self_play": {"frozen_opponent_fraction": 0.5}})
    assert learner_agent_slots(cfg) == 2 * 2 + 2 * 1
