"""End-to-end plumbing check: train -> checkpoint -> resume -> export -> RLBot runner load -> evaluate.

Validates that every stage connects and that state round-trips exactly. It runs a few tiny PPO updates and
says nothing about learning quality or Rocket League skill.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from botcore import checkpoint as ck
from botcore.config import load_config
from botcore.evaluate import MatchRules, Side, evaluate, format_summary
from botcore.export import bundle_from_checkpoint, load_bundle
from botcore.trainer import resume_run, start_run

ROOT = Path(__file__).resolve().parents[2]


def _optimizer_states_equal(a, b) -> bool:
    for oa, ob in zip(a.optimizers, b.optimizers):
        sa, sb = oa.state_dict()["state"], ob.state_dict()["state"]
        if sa.keys() != sb.keys():
            return False
        for k in sa:
            for name, va in sa[k].items():
                vb = sb[k][name]
                if torch.is_tensor(va) and not torch.equal(va.cpu(), vb.cpu()):
                    return False
    return True


def run_smoke(runs_dir: Path | None = None, log: Callable[[str], None] = print) -> dict:
    t_start = time.perf_counter()
    cfg_path = ROOT / "configs" / "smoke.toml"
    cfg = load_config(cfg_path)
    runs_dir = Path(runs_dir or ROOT / cfg.run.runs_dir)
    run_dir = runs_dir / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    checks: dict = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        log(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
        if not ok:
            raise AssertionError(f"smoke check failed: {name} {detail}")

    log(f"== smoke test in {run_dir}")
    trainer = start_run(cfg, run_dir, cfg_path.read_text(encoding="utf-8"))
    v0 = trainer.policy.get_actor_parameter_vector().detach().cpu().clone()
    c0 = trainer.policy.get_critic_parameter_vector().detach().cpu().clone()
    reports = trainer.train(max_iterations=2)
    v2 = trainer.policy.get_actor_parameter_vector().detach().cpu().clone()
    c2 = trainer.policy.get_critic_parameter_vector().detach().cpu().clone()
    finite = all(np.isfinite(json.loads(json.dumps(r.metrics["ppo"]))[k]) for r in reports for k in ("entropy", "value_loss"))
    check("ppo updates ran with finite metrics", len(reports) == 2 and finite,
          f"entropy {reports[-1].metrics['ppo']['entropy']:.4f}, value loss {reports[-1].metrics['ppo']['value_loss']:.4f}")
    check("actor and critic parameters changed", float((v2 - v0).norm()) > 0 and float((c2 - c0).norm()) > 0,
          f"|d actor| = {float((v2 - v0).norm()):.4g}, |d critic| = {float((c2 - c0).norm()):.4g}")
    frozen_ok = all(r.metrics["frozen_opponent"] is not None for r in reports)
    check("frozen-opponent environments ran", frozen_ok, f"opponent {reports[-1].metrics['frozen_opponent']}")

    ckpt2 = ck.resolve_checkpoint(run_dir)
    check("atomic checkpoint written", ck.is_complete(ckpt2) and ckpt2.name == "iter_000002", str(ckpt2))
    stats_before = trainer.trajectory_processor.rust_gae_trajectory_processor.state_dict()

    resumed = resume_run(run_dir, log=log)
    v_loaded = resumed.policy.get_actor_parameter_vector().detach().cpu()
    stats_after = resumed.trajectory_processor.rust_gae_trajectory_processor.state_dict()
    check("resume restores weights exactly", torch.equal(v_loaded, v2))
    check("resume restores optimiser state exactly", _optimizer_states_equal(trainer.learner, resumed.learner))
    check("resume restores return normaliser", stats_before == stats_after, json.dumps(stats_after))
    check("resume restores counters", resumed.iteration == 2 and resumed.learner_timesteps == trainer.learner_timesteps)
    resumed.train(max_iterations=1)
    ckpt3 = ck.resolve_checkpoint(run_dir)
    check("training continued after resume", ckpt3.name == "iter_000003")

    exports = run_dir / "exports"
    old = bundle_from_checkpoint(run_dir / "checkpoints" / "iter_000001", exports / "iter_000001")
    new = bundle_from_checkpoint(ckpt3, exports / "iter_000003")
    bundle = load_bundle(new)
    check("export bundle loads with contract check", bundle.manifest["source"]["iteration"] == 3,
          f"weights {bundle.identity}")

    from botcore.deploy.packets import CompatObsPipeline, field_info, state_to_packet
    from botcore.deploy.runner import BotCoreBot
    from botcore.env import EnvSpec, build_env

    bot = BotCoreBot(new, device="cpu")
    env = build_env(EnvSpec.from_config(cfg), seed=3)
    env.reset()
    pipe = CompatObsPipeline(field_info())
    pipe.update(state_to_packet(env.state, 120, {"blue-0": 0, "orange-0": 1}))
    action = bot.get_action(pipe.obs(0), pipe.state, None)
    check("RLBot runner loads the bundle and returns a legal action",
          bot.load_error is None and 0 <= int(action[0]) < 90, f"action {int(action[0])}")
    broken = BotCoreBot(run_dir / "does-not-exist", device="cpu")
    check("RLBot runner falls back to a logged zero action when the bundle is missing",
          broken.load_error is not None and int(broken.get_action(pipe.obs(0), pipe.state, None)[0]) == bot.zero_index)

    rules = MatchRules(team_size=cfg.env.team_size, match_seconds=cfg.eval.match_seconds,
                       overtime_max_seconds=cfg.eval.overtime_max_seconds)
    report = evaluate(Side(str(new)), Side(str(old)), rules, n_pairs=cfg.eval.n_seed_pairs, workers=2)
    report_path = run_dir / "eval_smoke.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(format_summary(report))
    check("evaluation harness ran colour-swapped seed pairs", report["summary"]["matches"] == 2 * cfg.eval.n_seed_pairs,
          str(report_path))

    summary = {"run_dir": str(run_dir), "checks": checks, "seconds": time.perf_counter() - t_start}
    (run_dir / "smoke_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"== smoke test passed in {summary['seconds']:.0f}s (plumbing only; not evidence of playing strength)")
    return summary
