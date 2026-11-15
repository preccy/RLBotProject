"""Bounded throughput, model and latency benchmarks."""

from __future__ import annotations

import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from botcore.config import Config, ModelConfig, load_config
from botcore.models import ActorCriticPolicy, arch_from_config, build_actor, build_critic, count_parameters
from botcore.obs import OBS_SIZE


def benchmark_workers(config_path: Path, worker_counts: Sequence[int], iterations: int, log=print) -> List[dict]:
    """Short real training runs (collection + GAE + PPO update + checkpoint) per worker count.

    The first iteration (process start, CUDA warm-up) is excluded from the medians.
    """
    from botcore.trainer import Trainer

    results = []
    for n in worker_counts:
        cfg = load_config(config_path)
        cfg.collect.n_workers = n
        cfg.run.total_iterations = iterations
        cfg.checkpoint.save_every_iterations = 2
        cfg.self_play.snapshot_every_iterations = 10_000
        tmp = Path(tempfile.mkdtemp(prefix=f"botcore-bench-{n}w-"))
        try:
            trainer = Trainer(cfg, tmp, log=lambda *_: None)
            t0 = time.perf_counter()
            reports = trainer.train()
            wall = time.perf_counter() - t0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        steady = [r.metrics for r in reports[1:]] or [reports[-1].metrics]

        def med(path):
            vals = []
            for m in steady:
                v = m
                for key in path:
                    v = v[key]
                vals.append(v)
            return statistics.median(vals)

        row = {
            "workers": n,
            "envs": cfg.n_envs,
            "learner_steps_per_iteration": reports[-1].metrics["learner_timesteps"] // len(reports),
            "learner_steps_per_s": med(("throughput", "learner_steps_per_s")),
            "env_steps_per_s": med(("throughput", "env_steps_per_s")),
            "physics_ticks_per_s": med(("throughput", "physics_ticks_per_s")),
            "game_time_x_realtime": med(("throughput", "physics_ticks_per_s")) / 120.0,
            "collect_s": med(("timing_s", "collect")),
            "inference_s": med(("timing_s", "inference")),
            "update_s": med(("timing_s", "update")),
            "checkpoint_s_max": max(m["timing_s"]["checkpoint"] for m in steady),
            "cpu_percent": med(("resources", "cpu_percent_system")),
            "ram_rss_gb": med(("resources", "ram_rss_gb_tree")),
            "vram_peak_mb": med(("resources", "vram_peak_mb")) if steady[0]["resources"]["vram_peak_mb"] else None,
            "iterations_measured": len(steady),
            "wall_s_incl_startup": wall,
        }
        results.append(row)
        log(f"{n} workers ({row['envs']} envs): {row['learner_steps_per_s']:,.0f} learner steps/s, "
            f"{row['env_steps_per_s']:,.0f} env steps/s, {row['physics_ticks_per_s']:,.0f} ticks/s "
            f"({row['game_time_x_realtime']:.0f}x real time); collect {row['collect_s']:.2f}s update {row['update_s']:.2f}s; "
            f"CPU {row['cpu_percent']:.0f}% RAM {row['ram_rss_gb']:.1f} GB VRAM {row['vram_peak_mb'] or 0:.0f} MB")
    return results


def _time(fn, repeats: int, warmup: int = 5) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(samples)
    return {"p50_ms": float(np.median(arr)), "p99_ms": float(np.percentile(arr, 99)), "max_ms": float(arr.max())}


def benchmark_models(models: Dict[str, ModelConfig], batch_size: int = 16384, log=print) -> List[dict]:
    """Parameter count, PPO-style update time/memory on GPU and single-observation inference latency."""
    rows = []
    cuda = torch.cuda.is_available()
    for name, mcfg in models.items():
        arch = arch_from_config(mcfg)
        torch.manual_seed(0)
        actor = build_actor(arch, OBS_SIZE, 90)
        critic = build_critic(arch, OBS_SIZE)
        row = {"model": name, "arch": arch, "actor_params": count_parameters(actor),
               "critic_params": count_parameters(critic)}
        obs1 = torch.randn(1, OBS_SIZE)
        with torch.inference_mode():
            actor.eval()
            torch.set_num_threads(1)
            row["cpu_1thread_infer_batch1"] = _time(lambda: actor(obs1), 300)
            torch.set_num_threads(max(1, torch.get_num_threads()))
        if cuda:
            dev = torch.device("cuda:0")
            policy = ActorCriticPolicy(actor, critic, dev)
            obs_gpu1 = obs1.to(dev)

            def gpu_infer():
                with torch.inference_mode():
                    policy.actor(obs_gpu1).argmax(-1).item()

            row["gpu_infer_batch1_incl_sync"] = _time(gpu_infer, 300)
            batch96 = torch.randn(96, OBS_SIZE, device=dev)

            def gpu_rollout_batch():
                with torch.inference_mode():
                    policy.act(batch96)[0].cpu()

            row["gpu_rollout_batch96"] = _time(gpu_rollout_batch, 200)
            opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
            obs = torch.randn(batch_size, OBS_SIZE, device=dev)
            acts = torch.randint(0, 90, (batch_size, 1), device=dev)
            torch.cuda.reset_peak_memory_stats()

            def update():
                lp, ent, val = policy.get_backprop_data(None, obs, acts)
                loss = -(lp.mean()) - 0.01 * ent + val.pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                torch.cuda.synchronize()

            row[f"gpu_update_batch{batch_size}"] = _time(update, 20, warmup=3)
            row["gpu_update_peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rows.append(row)
        log(json.dumps(row))
    return rows


def benchmark_deploy_latency(bundle_dir: Path, devices: Sequence[str], decisions: int = 2000, log=print) -> List[dict]:
    """Full per-decision cost of the deployment path on synthetic packets.

    Measured per decision: GamePacket -> rlgym-compat state update, pad/flip reconstruction, obs build, network
    forward and sampling, action parsing into 8 ControllerStates. Not measured: socket I/O and RLBotServer.
    """
    from rlbot import flat

    from botcore.actions import LookupActionParser
    from botcore.deploy.packets import CompatObsPipeline, field_info, state_to_packet
    from botcore.env import EnvSpec, build_env
    from botcore.export import PolicyRunner, load_bundle

    cfg = Config()
    env = build_env(EnvSpec.from_config(cfg), seed=21)
    env.reset()
    rng = np.random.default_rng(0)
    packets = []
    for i in range(decisions):
        env.step({a: np.array([rng.integers(90)]) for a in env.agents})
        packets.append(state_to_packet(env.state, 1000 + 8 * i, {"blue-0": 0, "orange-0": 1}))
        if env.state.goal_scored:
            env.reset()
    rows = []
    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            continue
        torch.set_num_threads(1)
        bundle = load_bundle(bundle_dir, device=device)
        runner = PolicyRunner(bundle, device=device, deterministic=False, seed=0)
        parser = LookupActionParser(bundle.action_repeat)
        pipe = CompatObsPipeline(field_info())
        shared: dict = {}
        prev = None
        timings = {"state_update": [], "obs": [], "policy": [], "parse": [], "total": []}
        for i, packet in enumerate(packets):
            t0 = time.perf_counter()
            pipe.update(packet)
            t1 = time.perf_counter()
            obs = pipe.obs(0, prev)
            t2 = time.perf_counter()
            idx = runner.act(obs)
            t3 = time.perf_counter()
            engine = parser.parse_actions({0: np.array([idx])}, pipe.state, shared)[0]
            _ = [flat.ControllerState(throttle=float(r[0]), steer=float(r[1]), pitch=float(r[2]), yaw=float(r[3]),
                                      roll=float(r[4]), jump=bool(r[5]), boost=bool(r[6]), handbrake=bool(r[7]))
                 for r in engine]
            prev = shared["prev_actions"][0]
            t4 = time.perf_counter()
            if i >= 50:
                for key, value in (("state_update", t1 - t0), ("obs", t2 - t1), ("policy", t3 - t2), ("parse", t4 - t3),
                                   ("total", t4 - t0)):
                    timings[key].append(value * 1000)
        row = {"device": device, "decisions": len(timings["total"]), "bundle": bundle.identity,
               "actor_params": bundle.manifest["actor_parameters"]}
        for key, vals in timings.items():
            arr = np.asarray(vals)
            row[key] = {"p50_ms": float(np.median(arr)), "p99_ms": float(np.percentile(arr, 99)), "max_ms": float(arr.max())}
        row["over_6ms"] = int(np.sum(np.asarray(timings["total"]) > 6.0))
        rows.append(row)
        log(f"{device}: total p50 {row['total']['p50_ms']:.3f} ms, p99 {row['total']['p99_ms']:.3f} ms, "
            f"max {row['total']['max_ms']:.2f} ms (policy p50 {row['policy']['p50_ms']:.3f} ms); "
            f"{row['over_6ms']}/{row['decisions']} over 6 ms")
    return rows
