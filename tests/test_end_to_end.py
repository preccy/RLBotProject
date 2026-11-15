"""The full smoke routine as a test (train, checkpoint, resume, export, runner load, evaluate)."""

from __future__ import annotations

import pytest

from botcore import checkpoint as ck
from botcore.config import load_config
from botcore.smoke import ROOT, run_smoke
from botcore.trainer import resume_run


@pytest.mark.slow
def test_smoke_routine_and_resume_guards(tmp_path):
    summary = run_smoke(tmp_path, log=lambda *_: None)
    assert all(c["ok"] for c in summary["checks"].values())
    run_dir = next(tmp_path.iterdir())

    changed = load_config(ROOT / "configs" / "smoke.toml")
    changed.rewards.weights["ball_touch"] = 1.0
    with pytest.raises(ck.CheckpointError, match="--allow-config-change"):
        resume_run(run_dir, cfg=changed, log=lambda *_: None)
    resume_run(run_dir, cfg=changed, allow_config_change=True, log=lambda *_: None)

    bigger = load_config(ROOT / "configs" / "smoke.toml")
    bigger.model.actor_layers = (128, 128)
    with pytest.raises(ck.CheckpointError, match="architecture changed"):
        resume_run(run_dir, cfg=bigger, allow_config_change=True, log=lambda *_: None)

    more_workers = load_config(ROOT / "configs" / "smoke.toml")
    more_workers.collect.n_workers = 1
    more_workers.collect.envs_per_worker = 4  # throughput settings may change freely
    resume_run(run_dir, cfg=more_workers, log=lambda *_: None)

    too_big = load_config(ROOT / "configs" / "smoke.toml")
    too_big.ppo.batch_size = 10_000  # edited in code after loading: must still be validated
    with pytest.raises(Exception, match="exceeds"):
        resume_run(run_dir, cfg=too_big, allow_config_change=True, log=lambda *_: None)
