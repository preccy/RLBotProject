"""Atomic, resumable checkpoints and the latest / candidate / best-verified pointers.

Layout of a run directory::

    runs/<name>/
        config.toml               copy of the config the run was started with
        metrics.jsonl             one JSON line per iteration
        checkpoints/iter_000123/  complete resumable state (see save_checkpoint)
        opponents/                frozen policy snapshots (export bundles) for self-play/evaluation
        exports/                  inference bundles made by `botcore export`
        pointers.json             {"latest": ..., "candidate": ..., "best_verified": ...}

A checkpoint directory is written under a temporary name and renamed into place only after every file is
flushed to disk, so a crash mid-save can never leave a truncated checkpoint under a real name, and an
existing checkpoint is never overwritten.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

CHECKPOINT_FORMAT = 1
TRAINER_STATE_FILE = "trainer_state.json"
RNG_FILE = "rng_state.pt"
COMPLETE_MARKER = "COMPLETE"
POINTER_KEYS = ("latest", "candidate", "best_verified")


class CheckpointError(RuntimeError):
    """A checkpoint is missing, incomplete or incompatible."""


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with open(path, "rb+") as f:  # Windows needs a writable handle for fsync
                os.fsync(f.fileno())


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError:
            pass  # different GPU count; documented as non-exact resume


def checkpoint_name(iteration: int) -> str:
    return f"iter_{iteration:06d}"


def save_checkpoint(checkpoints_dir: Path, iteration: int, write_payload, trainer_state: Dict[str, Any]) -> Path:
    """Write a complete checkpoint atomically.

    :param write_payload: callable(tmp_dir: Path) that writes learner/optimizer/normaliser files.
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoints_dir / checkpoint_name(iteration)
    if final.exists():
        raise CheckpointError(f"refusing to overwrite existing checkpoint {final}")
    tmp = checkpoints_dir / f".{final.name}.tmp-{os.getpid()}-{time.time_ns()}"
    tmp.mkdir()
    try:
        write_payload(tmp)
        torch.save(capture_rng_state(), tmp / RNG_FILE)
        state = dict(trainer_state)
        state["checkpoint_format"] = CHECKPOINT_FORMAT
        state["iteration"] = iteration
        state["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (tmp / TRAINER_STATE_FILE).write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        (tmp / COMPLETE_MARKER).write_text("ok", encoding="utf-8")
        _fsync_tree(tmp)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def is_complete(path: Path) -> bool:
    path = Path(path)
    return (path / COMPLETE_MARKER).is_file() and (path / TRAINER_STATE_FILE).is_file()


def read_trainer_state(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_dir():
        raise CheckpointError(f"checkpoint not found: {path}")
    if not is_complete(path):
        raise CheckpointError(f"checkpoint {path} is incomplete (no {COMPLETE_MARKER} marker); refusing to load it")
    state = json.loads((path / TRAINER_STATE_FILE).read_text(encoding="utf-8"))
    if state.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"checkpoint {path} has format {state.get('checkpoint_format')}, expected {CHECKPOINT_FORMAT}")
    return state


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        return []
    return sorted(p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("iter_") and is_complete(p))


def prune_checkpoints(checkpoints_dir: Path, keep_last: int, protect: set[Path]) -> list[Path]:
    """Delete old complete checkpoints beyond `keep_last`, never touching protected ones
    (anything referenced by pointers.json). Leftover temporary directories are removed too."""
    removed = []
    for leftover in Path(checkpoints_dir).glob(".iter_*.tmp-*"):
        shutil.rmtree(leftover, ignore_errors=True)
    complete = list_checkpoints(checkpoints_dir)
    protected = {p.resolve() for p in protect}
    for path in complete[:-keep_last] if keep_last > 0 else complete:
        if path.resolve() in protected:
            continue
        shutil.rmtree(path)
        removed.append(path)
    return removed


# ----------------------------------------------------------------------------------------- pointers
def read_pointers(run_dir: Path) -> Dict[str, Optional[str]]:
    path = Path(run_dir) / "pointers.json"
    pointers = {key: None for key in POINTER_KEYS}
    if path.is_file():
        pointers.update(json.loads(path.read_text(encoding="utf-8")))
    return pointers


def write_pointer(run_dir: Path, key: str, value: Optional[str]) -> None:
    if key not in POINTER_KEYS:
        raise ValueError(f"unknown pointer {key!r}")
    pointers = read_pointers(run_dir)
    pointers[key] = value
    atomic_write_text(Path(run_dir) / "pointers.json", json.dumps(pointers, indent=2))


def protected_paths(run_dir: Path) -> set[Path]:
    out = set()
    for value in read_pointers(run_dir).values():
        if value and key_is_checkpoint(value):
            out.add(Path(run_dir) / value)
    return out


def key_is_checkpoint(value: str) -> bool:
    return value.replace("\\", "/").startswith("checkpoints/")


def resolve_checkpoint(spec: str | Path) -> Path:
    """Accept a checkpoint directory or a run directory (then use its `latest` pointer)."""
    path = Path(spec)
    if is_complete(path):
        return path
    pointers_file = path / "pointers.json"
    if pointers_file.is_file():
        latest = read_pointers(path).get("latest")
        if latest:
            return path / latest
    if path.is_dir() and path.name == "checkpoints":
        found = list_checkpoints(path)
        if found:
            return found[-1]
    raise CheckpointError(f"no complete checkpoint found at {path}")
