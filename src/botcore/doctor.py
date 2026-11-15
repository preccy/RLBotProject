"""Environment diagnosis with actionable messages."""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

EXPECTED = {
    "torch": "2.7.0+cu128",
    "numpy": "1.26.4",
    "rlgym": "2.0.1",
    "rlgym-rocket-league": "2.0.1",
    "rocketsim": "2.2.1",
    "rlgym-learn-algos": "0.4.2",
    "rlgym-learn": "2.0.0",
    "rlgym-compat": "2.3.6",
    "rlgym-rlbot": "0.3.0",
    "rlbot": "2.0.0b52",
    "rlbot_flatbuffers": "0.18.11",
}


def _check(results: list, name: str, fn: Callable[[], Tuple[str, str]]) -> None:
    t0 = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    results.append((status, name, detail, time.perf_counter() - t0))


def check_python():
    v = sys.version_info
    in_venv = sys.prefix != sys.base_prefix
    detail = f"{platform.python_version()} at {sys.executable}{' (venv)' if in_venv else ''}"
    if v[:2] != (3, 11):
        return WARN, detail + "; this project was verified on Python 3.11"
    if not in_venv:
        return WARN, detail + "; not inside a virtual environment"
    return PASS, detail


def check_packages():
    problems, found = [], []
    for pkg, want in EXPECTED.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            problems.append(f"{pkg} missing (pip install -r requirements-lock.txt)")
            continue
        found.append(f"{pkg}={have}")
        if have != want:
            problems.append(f"{pkg} {have} != verified {want}")
    if any("missing" in p for p in problems):
        return FAIL, "; ".join(problems)
    if problems:
        return WARN, "; ".join(problems) + " (may still work; re-run smoke-test)"
    return PASS, ", ".join(found)


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        return WARN, f"torch {torch.__version__} sees no CUDA device; training will use the CPU (much slower updates)"
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(4096, 256, device=dev)
    y = x.sum(1, keepdim=True)
    before = [p.detach().clone() for p in net.parameters()]
    first = None
    for _ in range(10):
        loss = torch.nn.functional.mse_loss(net(x), y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    changed = all((a != b).any().item() for a, b in zip(before, net.parameters()))
    if not (changed and loss.item() < first):
        return FAIL, "CUDA forward/backward ran but parameters/loss did not update"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    return PASS, (f"{name} sm_{cap[0]}{cap[1]}, torch {torch.__version__} (CUDA {torch.version.cuda}); "
                  f"forward+backward+Adam OK, loss {first:.1f} -> {loss.item():.1f}")


def check_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return WARN, "nvidia-smi not found on PATH"
    import subprocess

    out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    return PASS, out


def check_rocketsim():
    import numpy as np

    from botcore.config import Config
    from botcore.env import EnvSpec, build_env

    env = build_env(EnvSpec.from_config(Config()), seed=0)
    obs = env.reset()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = 500
    for _ in range(n):
        obs, rew, term, trunc = env.step({a: np.array([rng.integers(90)]) for a in env.agents})
        if any(term.values()) or any(trunc.values()):
            obs = env.reset()
    dt = time.perf_counter() - t0
    ok = all(np.isfinite(o).all() for o in obs.values())
    return (PASS if ok else FAIL), f"reset+{n} steps OK in one process ({n / dt:.0f} env steps/s incl. botcore obs/rewards)"


def check_multiprocessing():
    from botcore.collect import Collector
    from botcore.config import Config
    from botcore.env import EnvSpec

    import numpy as np

    t0 = time.perf_counter()
    collector = Collector(EnvSpec.from_config(Config()), n_workers=2, envs_per_worker=2, seed=11)
    try:
        collector.reset()
        for _ in range(20):
            collector.step(np.zeros((4, 2), dtype=np.int64))
    finally:
        collector.close()
    alive = collector.alive_workers()
    if alive:
        return FAIL, f"{alive} worker process(es) did not exit"
    return PASS, f"2 spawn workers started, stepped and shut down cleanly in {time.perf_counter() - t0:.1f}s"


def check_contracts():
    from botcore.actions import get_action_table
    from botcore.obs import OBS_SIZE, OBS_SPEC_HASH, OBS_SPEC_VERSION

    t = get_action_table()
    return PASS, f"{OBS_SPEC_VERSION} size {OBS_SIZE} hash {OBS_SPEC_HASH[:12]}; {t.table_id} sha {t.sha256[:12]}"


def check_disk():
    usage = shutil.disk_usage(ROOT)
    free = usage.free / 2 ** 30
    return (PASS if free > 20 else WARN), f"{free:.0f} GB free on the project drive"


def check_rlbot():
    local = os.environ.get("LOCALAPPDATA")
    server = Path(local) / "RLBot5" / "bin" / "RLBotServer.exe" if local else None
    if server is None or not server.is_file():
        return WARN, "RLBot v5 not found (expected %LOCALAPPDATA%\\RLBot5\\bin\\RLBotServer.exe); install from rlbot.org/v5"
    version = ""
    try:
        import subprocess

        ps = ("(Get-Item '%s').VersionInfo.ProductVersion" % server)
        version = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return PASS, f"{server} {version}".strip()


def _steam_libraries() -> List[Path]:
    roots = [Path("C:/Program Files (x86)/Steam")]
    vdf = roots[0] / "steamapps" / "libraryfolders.vdf"
    if vdf.is_file():
        for match in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="ignore")):
            roots.append(Path(match.group(1).replace("\\\\", "\\")))
    return roots


def check_rocket_league():
    candidates = [lib / "steamapps" / "common" / "rocketleague" for lib in _steam_libraries()]
    candidates += [Path("C:/Program Files/Epic Games/rocketleague")]
    for c in candidates:
        if (c / "Binaries" / "Win64" / "RocketLeague.exe").is_file() or c.is_dir():
            return PASS, f"found {c} (not launched by doctor)"
    return WARN, "Rocket League install not found in Steam libraries or the default Epic path"


def check_rlviser():
    exe = ROOT / "rlviser.exe"
    if exe.is_file():
        return PASS, str(exe)
    return WARN, "rlviser.exe not in the project root; `watch` needs it (see README)"


CHECKS = [
    ("python", check_python),
    ("packages", check_packages),
    ("nvidia driver", check_nvidia_smi),
    ("cuda training op", check_cuda),
    ("contracts", check_contracts),
    ("rocketsim env", check_rocketsim),
    ("multiprocessing", check_multiprocessing),
    ("disk", check_disk),
    ("rlbot v5", check_rlbot),
    ("rocket league", check_rocket_league),
    ("rlviser (optional)", check_rlviser),
]


def run_doctor(log=print) -> int:
    results: list = []
    for name, fn in CHECKS:
        _check(results, name, fn)
        status, n, detail, dt = results[-1]
        log(f"[{status}] {n:<20} {detail}  ({dt:.1f}s)")
    failed = [r for r in results if r[0] == FAIL]
    log(f"\n{len(results) - len(failed)}/{len(results)} checks without failures; "
        f"{sum(r[0] == WARN for r in results)} warnings.")
    return 1 if failed else 0
