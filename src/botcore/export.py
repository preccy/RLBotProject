"""Inference bundles: actor weights + manifest, and strict loading.

A bundle is a directory containing ``policy.pt`` (actor state_dict only) and ``manifest.json``. The manifest
records everything needed to decide whether the weights may be used with the current code: architecture,
observation spec hash, action table hash, action repeat, input-delay convention, dependency versions and
the identity of the source checkpoint. `load_bundle` refuses anything that does not match.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from botcore import __version__
from botcore.actions import get_action_table
from botcore.models import Actor, build_actor, count_parameters, sample_actions
from botcore.obs import OBS_SIZE, OBS_SPEC_HASH, OBS_SPEC_VERSION

MANIFEST_FORMAT = 1
WEIGHTS_FILE = "policy.pt"
MANIFEST_FILE = "manifest.json"
PACKAGES = ("torch", "numpy", "rlgym", "rlgym-rocket-league", "rocketsim", "rlgym-learn-algos",
            "rlgym-compat", "rlgym-rlbot", "rlbot", "rlbot_flatbuffers")


class IncompatibleExportError(RuntimeError):
    """The bundle cannot be used with this code (contract mismatch, corrupt weights, unknown format)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_versions() -> Dict[str, Optional[str]]:
    out = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def source_revision(repo: Path | None = None) -> Dict[str, Any]:
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"git_commit": commit, "dirty": dirty, "botcore_version": __version__}


def contract(action_repeat: int, rlbot_delay: bool) -> Dict[str, Any]:
    table = get_action_table()
    return {
        "obs_spec_version": OBS_SPEC_VERSION,
        "obs_spec_hash": OBS_SPEC_HASH,
        "obs_size": OBS_SIZE,
        "action_table_id": table.table_id,
        "action_table_sha256": table.sha256,
        "n_actions": table.n_actions,
        "action_repeat": int(action_repeat),
        "rlbot_delay": bool(rlbot_delay),
        "normalisation": "fixed scales defined by the obs spec; no running statistics",
    }


def write_bundle(out_dir: Path, actor: Actor, arch: dict, action_repeat: int, rlbot_delay: bool,
                 source: Dict[str, Any]) -> Path:
    """Write a bundle atomically. Refuses to overwrite an existing bundle."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"export target already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.mkdir()
    try:
        state = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        torch.save(state, tmp / WEIGHTS_FILE)
        manifest = {
            "manifest_format": MANIFEST_FORMAT,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "architecture": arch,
            "actor_parameters": count_parameters(actor),
            "contract": contract(action_repeat, rlbot_delay),
            "weights_sha256": sha256_file(tmp / WEIGHTS_FILE),
            "source": source,
            "dependencies": dependency_versions(),
            "code": source_revision(),
        }
        (tmp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for f in tmp.iterdir():
            with open(f, "rb+") as fh:
                os.fsync(fh.fileno())
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir


@dataclass
class Bundle:
    path: Path
    manifest: Dict[str, Any]
    actor: Actor

    @property
    def action_repeat(self) -> int:
        return int(self.manifest["contract"]["action_repeat"])

    @property
    def rlbot_delay(self) -> bool:
        return bool(self.manifest["contract"]["rlbot_delay"])

    @property
    def identity(self) -> str:
        return self.manifest["weights_sha256"][:16]


def check_contract(manifest: Dict[str, Any], where: str) -> None:
    if manifest.get("manifest_format") != MANIFEST_FORMAT:
        raise IncompatibleExportError(f"{where}: manifest format {manifest.get('manifest_format')!r} is not supported "
                                      f"(expected {MANIFEST_FORMAT})")
    c = manifest.get("contract", {})
    table = get_action_table()
    problems = []
    if c.get("obs_spec_hash") != OBS_SPEC_HASH:
        problems.append(f"observation spec {c.get('obs_spec_version')} ({str(c.get('obs_spec_hash'))[:12]}) != current "
                        f"{OBS_SPEC_VERSION} ({OBS_SPEC_HASH[:12]})")
    if c.get("obs_size") != OBS_SIZE:
        problems.append(f"observation size {c.get('obs_size')} != {OBS_SIZE}")
    if c.get("action_table_sha256") != table.sha256 or c.get("action_table_id") != table.table_id:
        problems.append(f"action table {c.get('action_table_id')} ({str(c.get('action_table_sha256'))[:12]}) != current "
                        f"{table.table_id} ({table.sha256[:12]})")
    if problems:
        raise IncompatibleExportError(
            f"{where} was trained against a different input/output contract and cannot be used without retraining, "
            "distillation or an explicit migration:\n  - " + "\n  - ".join(problems)
        )


def load_bundle(path: str | Path, device: str | torch.device = "cpu") -> Bundle:
    path = Path(path)
    manifest_path = path / MANIFEST_FILE
    weights_path = path / WEIGHTS_FILE
    if not manifest_path.is_file() or not weights_path.is_file():
        raise IncompatibleExportError(f"{path} is not an export bundle (needs {MANIFEST_FILE} and {WEIGHTS_FILE})")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_contract(manifest, str(path))
    digest = sha256_file(weights_path)
    if digest != manifest.get("weights_sha256"):
        raise IncompatibleExportError(f"{weights_path} sha256 {digest[:12]} does not match the manifest "
                                      f"({str(manifest.get('weights_sha256'))[:12]}); the file is corrupt or was replaced")
    actor = build_actor(manifest["architecture"], OBS_SIZE, get_action_table().n_actions)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    try:
        actor.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise IncompatibleExportError(f"{path}: weights do not fit the declared architecture: {exc}") from exc
    actor.to(device).eval()
    return Bundle(path, manifest, actor)


class PolicyRunner:
    """Single-agent inference used by the RLBot runner, evaluation and the latency benchmark."""

    def __init__(self, bundle: Bundle, device: str = "cpu", deterministic: bool = False, seed: Optional[int] = None):
        self.bundle = bundle
        self.device = torch.device(device)
        self.actor = bundle.actor.to(self.device).eval()
        self.deterministic = deterministic
        self.generator: Optional[torch.Generator] = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(seed))

    @torch.inference_mode()
    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        actions, _ = sample_actions(self.actor(x), self.deterministic, self.generator)
        return actions.cpu().numpy()

    def act(self, obs: np.ndarray) -> int:
        return int(self.act_batch(obs)[0])


def bundle_from_checkpoint(checkpoint_dir: str | Path, out_dir: str | Path) -> Path:
    """Export the actor of a resumable checkpoint as an inference bundle (critic/optimisers are dropped)."""
    from botcore import checkpoint as ck

    checkpoint_dir = Path(checkpoint_dir)
    state = ck.read_trainer_state(checkpoint_dir)
    manifest_like = {"manifest_format": MANIFEST_FORMAT, "contract": state["contract"]}
    check_contract(manifest_like, str(checkpoint_dir))
    weights = torch.load(checkpoint_dir / "ppo_learner" / "actor_critic.pt", map_location="cpu", weights_only=True)
    actor_state = {k[len("actor."):]: v for k, v in weights.items() if k.startswith("actor.")}
    if not actor_state:
        raise IncompatibleExportError(f"{checkpoint_dir}: no actor weights found in actor_critic.pt")
    actor = build_actor(state["architecture"], OBS_SIZE, get_action_table().n_actions)
    try:
        actor.load_state_dict(actor_state, strict=True)
    except RuntimeError as exc:
        raise IncompatibleExportError(f"{checkpoint_dir}: actor weights do not fit the recorded architecture: {exc}") from exc
    source = {
        "checkpoint": str(checkpoint_dir.resolve()),
        "run": state["config"]["run"]["name"],
        "iteration": state["iteration"],
        "learner_timesteps": state["learner_timesteps"],
        "curriculum_stage": state.get("curriculum_stage"),
    }
    c = state["contract"]
    return write_bundle(Path(out_dir), actor, state["architecture"], c["action_repeat"], c["rlbot_delay"], source)
