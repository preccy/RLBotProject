"""Command line interface: python -m botcore <command> ..."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _resolve_policy(spec: str) -> Path:
    """An export bundle directory, a checkpoint directory, or a run directory (its latest checkpoint).
    Checkpoints are exported on demand into <run>/exports/<checkpoint name>."""
    from botcore import checkpoint as ck
    from botcore.export import MANIFEST_FILE, bundle_from_checkpoint

    path = Path(spec)
    if (path / MANIFEST_FILE).is_file():
        return path
    ckpt = ck.resolve_checkpoint(path)
    out = ckpt.parent.parent / "exports" / ckpt.name
    if not out.exists():
        bundle_from_checkpoint(ckpt, out)
    return out


def cmd_doctor(args) -> int:
    from botcore.doctor import run_doctor

    return run_doctor()


def cmd_smoke(args) -> int:
    from botcore.smoke import run_smoke

    run_smoke(Path(args.runs_dir) if args.runs_dir else None)
    return 0


def cmd_train(args) -> int:
    from botcore.config import load_config
    from botcore.trainer import start_run

    cfg = load_config(args.config, {"run": {"seed": args.seed}} if args.seed is not None else None)
    name = args.name or f"{cfg.run.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    cfg.run.name = name
    run_dir = ROOT / cfg.run.runs_dir / name
    trainer = start_run(cfg, run_dir, Path(args.config).read_text(encoding="utf-8"))
    print(f"run directory: {run_dir}")
    print("Ctrl+C (or creating a file named STOP in the run directory) checkpoints and stops cleanly.")
    trainer.train(max_iterations=args.iterations)
    return 0


def cmd_resume(args) -> int:
    from botcore.config import load_config
    from botcore.trainer import resume_run

    cfg = load_config(args.config) if args.config else None
    trainer = resume_run(Path(args.run), cfg=cfg, checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                         allow_config_change=args.allow_config_change)
    print(f"resuming {args.run} from iteration {trainer.iteration}")
    trainer.train(max_iterations=args.iterations)
    return 0


def cmd_export(args) -> int:
    from botcore import checkpoint as ck
    from botcore.export import bundle_from_checkpoint, load_bundle

    ckpt = ck.resolve_checkpoint(Path(args.checkpoint))
    run_dir = ckpt.parent.parent
    out = Path(args.out) if args.out else run_dir / "exports" / ckpt.name
    if not out.exists():
        bundle_from_checkpoint(ckpt, out)
    bundle = load_bundle(out)
    print(f"bundle {out} (weights {bundle.identity}, {bundle.manifest['actor_parameters']:,} actor params)")
    if args.mark_candidate:
        ck.write_pointer(run_dir, "candidate", f"checkpoints/{ckpt.name}")
        print(f"marked {ckpt.name} as the run's candidate")
    if args.to_bot:
        target = ROOT / "bot" / "model"
        if target.exists():
            previous = ROOT / "bot" / "model.previous"
            shutil.rmtree(previous, ignore_errors=True)
            target.rename(previous)
            print(f"previous bot model kept at {previous}")
        shutil.copytree(out, target)
        load_bundle(target)
        print(f"installed into {target}; RLBot will use it the next time the bot starts")
    return 0


def cmd_evaluate(args) -> int:
    from botcore.config import load_config
    from botcore.evaluate import MatchRules, Side, evaluate, format_summary

    cfg = load_config(args.config)
    a, b = _resolve_policy(args.a), _resolve_policy(args.b)
    rules = MatchRules(team_size=cfg.env.team_size,
                       match_seconds=args.match_seconds or cfg.eval.match_seconds,
                       overtime_max_seconds=cfg.eval.overtime_max_seconds if args.overtime is None else args.overtime)
    det = cfg.eval.deterministic
    report = evaluate(Side(str(a), args.a_deterministic or det), Side(str(b), args.b_deterministic or det), rules,
                      n_pairs=args.pairs or cfg.eval.n_seed_pairs, workers=args.workers, seed_offset=args.seed_offset)
    out = Path(args.out) if args.out else ROOT / "runs" / "eval" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_summary(report))
    print(f"report: {out}")
    return 0


def cmd_promote(args) -> int:
    from botcore import checkpoint as ck
    from botcore.evaluate import check_promotion
    from botcore.export import load_bundle

    run_dir = Path(args.run)
    pointers = ck.read_pointers(run_dir)
    if not pointers["candidate"]:
        print("no candidate marked; use `export --mark-candidate` first")
        return 1
    candidate = load_bundle(_resolve_policy(str(run_dir / pointers["candidate"])))
    best_sha = None
    if pointers["best_verified"]:
        best_sha = load_bundle(_resolve_policy(str(run_dir / pointers["best_verified"]))).manifest["weights_sha256"]
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    problems = check_promotion(candidate.manifest["weights_sha256"], best_sha, reports)
    if problems and not (args.initial and best_sha is None):
        print("NOT promoted:\n  - " + "\n  - ".join(problems))
        return 1
    ck.write_pointer(run_dir, "best_verified", pointers["candidate"])
    print(f"best_verified -> {pointers['candidate']}" + (" (initial best, set manually)" if problems else ""))
    return 0


def cmd_watch(args) -> int:
    from botcore.watch import watch

    a = _resolve_policy(args.policy)
    b = _resolve_policy(args.opponent) if args.opponent else None
    watch(a, b, args.seconds, args.scenario, args.seed, args.deterministic, args.speed, args.dry_run)
    return 0


def cmd_benchmark(args) -> int:
    from botcore.benchmark import benchmark_workers

    rows = benchmark_workers(Path(args.config), args.workers, args.iterations)
    out = ROOT / "runs" / "benchmarks" / f"workers-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return 0


def cmd_benchmark_models(args) -> int:
    from botcore.benchmark import benchmark_models
    from botcore.config import load_config

    models = {Path(p).stem: load_config(p).model for p in args.configs}
    rows = benchmark_models(models)
    out = ROOT / "runs" / "benchmarks" / f"models-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return 0


def cmd_latency(args) -> int:
    from botcore.benchmark import benchmark_deploy_latency

    rows = benchmark_deploy_latency(_resolve_policy(args.policy), args.devices, args.decisions)
    out = ROOT / "runs" / "benchmarks" / f"latency-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return 0


def cmd_replay(args) -> int:
    from botcore.deploy.recording import replay

    replay(args.recording, _resolve_policy(args.policy), args.device, args.player)
    return 0


def cmd_live_report(args) -> int:
    from botcore.deploy.live_report import live_report

    bundle = _resolve_policy(args.policy) if args.policy else ROOT / "bot" / "model"
    live_report(Path(args.dir), bundle if (Path(bundle) / "manifest.json").is_file() else None)
    return 0


def cmd_replay_html(args) -> int:
    from botcore.viz import frames_from_recording, frames_from_simulation, write_html

    if args.recording:
        data = frames_from_recording(Path(args.recording))
    else:
        if not args.policy:
            print("error: give --policy (and optionally --opponent) or --recording")
            return 2
        a = _resolve_policy(args.policy)
        b = _resolve_policy(args.opponent) if args.opponent else a
        data = frames_from_simulation(a, b, args.seconds, args.seed, args.scenario, args.deterministic)
    out = Path(args.out) if args.out else ROOT / "runs" / "viz" / f"replay-{time.strftime('%Y%m%d-%H%M%S')}.html"
    write_html(data, out)
    print(f"wrote {out} ({len(data['frames'])} frames); open it in a browser")
    return 0


def cmd_obs_doc(args) -> int:
    from botcore.obs.spec import render_markdown

    out = ROOT / "docs" / "OBSERVATION_SPEC.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown() + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_validate(args) -> int:
    from botcore.config import learner_agent_slots, load_config

    cfg = load_config(args.config)
    print(f"{args.config}: OK. {cfg.n_envs} envs, {learner_agent_slots(cfg) * cfg.collect.rollout_steps} learner "
          f"steps/iteration, {cfg.decision_hz:.1f} decisions/s, model {cfg.model.kind}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m botcore", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("doctor", help="diagnose the environment")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("smoke-test", help="bounded end-to-end plumbing test (~20 s)")
    s.add_argument("--runs-dir")
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("validate", help="validate a config without starting anything")
    s.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("train", help="start a new training run")
    s.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    s.add_argument("--name", help="run name (default: <config run.name>-<timestamp>)")
    s.add_argument("--iterations", type=int, help="stop after this many iterations (default: run.total_iterations)")
    s.add_argument("--seed", type=int, help="override [run] seed")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("resume", help="resume a run from its latest (or a given) checkpoint")
    s.add_argument("--run", required=True)
    s.add_argument("--checkpoint")
    s.add_argument("--config", help="new config (requires --allow-config-change if learning settings differ)")
    s.add_argument("--allow-config-change", action="store_true")
    s.add_argument("--iterations", type=int)
    s.set_defaults(fn=cmd_resume)

    s = sub.add_parser("export", help="export a checkpoint's actor as an inference bundle")
    s.add_argument("--checkpoint", required=True, help="checkpoint dir or run dir (latest)")
    s.add_argument("--out")
    s.add_argument("--to-bot", action="store_true", help="also install into bot/model for RLBot")
    s.add_argument("--mark-candidate", action="store_true")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("evaluate", help="headless colour-swapped comparison of two policies")
    s.add_argument("--a", required=True, help="bundle, checkpoint or run dir")
    s.add_argument("--b", required=True)
    s.add_argument("--config", default=str(ROOT / "configs" / "default.toml"), help="source of [eval] defaults")
    s.add_argument("--pairs", type=int)
    s.add_argument("--match-seconds", type=float)
    s.add_argument("--overtime", type=float)
    s.add_argument("--a-deterministic", action="store_true")
    s.add_argument("--b-deterministic", action="store_true")
    s.add_argument("--workers", type=int)
    s.add_argument("--seed-offset", type=int, default=0,
                   help="use evaluation seeds 1,000,000+offset upwards (fresh seeds for confirmation runs)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("promote", help="set best_verified to the candidate if evaluation reports pass the rule")
    s.add_argument("--run", required=True)
    s.add_argument("--reports", nargs="+", required=True)
    s.add_argument("--initial", action="store_true", help="allow setting the first best_verified without a prior best")
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("watch", help="render one simulated match in RLViser")
    s.add_argument("--policy", required=True)
    s.add_argument("--opponent")
    s.add_argument("--seconds", type=float, default=60.0)
    s.add_argument("--scenario", default="kickoff")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--deterministic", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="run without RLViser (checks everything but the window)")
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("benchmark", help="bounded training throughput for several worker counts")
    s.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    s.add_argument("--workers", type=int, nargs="+", default=[4, 6, 8])
    s.add_argument("--iterations", type=int, default=4)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("benchmark-models", help="parameter count, update speed/memory and inference latency")
    s.add_argument("--configs", nargs="+", default=[str(ROOT / "configs" / "default.toml"),
                                                    str(ROOT / "configs" / "entity.toml")])
    s.set_defaults(fn=cmd_benchmark_models)

    s = sub.add_parser("latency", help="deployment-path latency per decision on synthetic packets (CPU vs GPU)")
    s.add_argument("--policy", required=True)
    s.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    s.add_argument("--decisions", type=int, default=2000)
    s.set_defaults(fn=cmd_latency)

    s = sub.add_parser("replay", help="run a packet recording (BOTCORE_RECORD) through the deployment pipeline")
    s.add_argument("--recording", required=True)
    s.add_argument("--policy", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--player", type=int, default=0, help="index into packet.players of the car to evaluate")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("live-report", help="analyse a BOTCORE_LIVE_DIR folder from a real RLBot match")
    s.add_argument("--dir", required=True)
    s.add_argument("--policy", help="bundle used for the simulator reference (default bot\model)")
    s.set_defaults(fn=cmd_live_report)

    s = sub.add_parser("replay-html", help="write a self-contained HTML top-down replay (sim match or live recording)")
    s.add_argument("--policy")
    s.add_argument("--opponent")
    s.add_argument("--recording", help="a .bcp file from BOTCORE_LIVE_DIR")
    s.add_argument("--seconds", type=float, default=90.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--scenario", default="kickoff")
    s.add_argument("--deterministic", action="store_true")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_replay_html)

    s = sub.add_parser("obs-doc", help="regenerate docs/OBSERVATION_SPEC.md")
    s.set_defaults(fn=cmd_obs_doc)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except Exception as exc:  # noqa: BLE001 - expected user errors get a one-line message, bugs a traceback
        from botcore.checkpoint import CheckpointError
        from botcore.config import ConfigError
        from botcore.export import IncompatibleExportError

        if isinstance(exc, (ConfigError, CheckpointError, IncompatibleExportError, FileNotFoundError, FileExistsError)):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise
