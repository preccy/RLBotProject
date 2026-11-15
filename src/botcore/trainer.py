"""PPO training loop: botcore collector + rlgym-learn-algos PPOLearner / GAE / experience buffer.

Per iteration:
1. Collect `rollout_steps` decisions in every environment with the current policy (one policy version).
2. Cut unfinished episodes (truncation -> bootstrapped), compute value predictions for every observation
   and every trajectory's final observation with the current critic.
3. GAE (rlgym-learn-algos Rust processor): terminal steps use next value 0, truncated ones use
   V(final_obs); rewards are divided by a running return std and clipped.
4. PPO update over exactly this iteration's data (buffer cleared first; no off-policy reuse).
5. Log, snapshot, checkpoint.
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch
from rlgym_learn_algos.ppo import (
    DerivedExperienceBufferConfig,
    DerivedPPOLearnerConfig,
    ExperienceBufferConfigModel,
    GAETrajectoryProcessor,
    GAETrajectoryProcessorConfigModel,
    NumpyExperienceBuffer,
    PPOLearner,
    PPOLearnerConfigModel,
    Trajectory,
)
from rlgym_learn_algos.ppo.trajectory_processor import TRAJECTORY_PROCESSOR_FILE

from botcore import checkpoint as ck
from botcore.actions import get_action_table
from botcore.collect import Collector, ClosedTrajectory, role_map, run_rollout
from botcore.config import (EVAL_SEED_BASE, Config, ConfigError, config_from_dict, learner_agent_slots,
                            n_frozen_envs, validate)
from botcore.env import EnvSpec
from botcore.export import PolicyRunner, contract, dependency_versions, load_bundle, source_revision, write_bundle
from botcore.models import ActorCriticPolicy, arch_from_config, build_actor, build_critic, count_parameters
from botcore.obs import OBS_SIZE
from botcore.opponents import OpponentEntry, OpponentRegistry

AGENT_CONTROLLER_NAME = "botcore"
# Config sections whose change on resume only affects throughput/bookkeeping, never what is learned.
FREELY_CHANGEABLE = {("run", "total_iterations"), ("run", "device"), ("collect", "n_workers"),
                     ("collect", "envs_per_worker"), ("checkpoint", "save_every_iterations"),
                     ("checkpoint", "keep_last"), ("eval",)}


class TrainingDivergedError(RuntimeError):
    """Parameters became non-finite; the run stops without saving the broken state."""


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(f"device {name!r} requested but CUDA is not available")
    return torch.device(name)


def config_diff(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    diffs = []
    for section in sorted(set(old) | set(new)):
        a, b = old.get(section, {}), new.get(section, {})
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key), b.get(key)
            if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
                va, vb = list(va), list(vb)
            if va != vb and (section, key) not in FREELY_CHANGEABLE and (section,) not in FREELY_CHANGEABLE:
                diffs.append(f"[{section}] {key}: {va!r} -> {vb!r}")
    return diffs


def worker_seed_base(run_seed: int, start_iteration: int, n_envs: int) -> int:
    """Training env seeds: fixed per (run seed, resume point), always below the evaluation seed range."""
    span = EVAL_SEED_BASE - n_envs - 1
    return (run_seed * 7919 + start_iteration * 104729) % span


@dataclass
class IterationReport:
    iteration: int
    metrics: Dict[str, Any]


class Trainer:
    def __init__(self, cfg: Config, run_dir: Path, resume_from: Optional[Path] = None,
                 allow_config_change: bool = False, log=print):
        validate(cfg)  # configs may have been edited in code after loading
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log = log
        self.device = resolve_device(cfg.run.device)
        self.arch = arch_from_config(cfg.model)
        self.table = get_action_table()
        self.obs_space = ("real", OBS_SIZE)
        self.action_space = ("discrete", self.table.n_actions)
        self.steps_per_iteration = learner_agent_slots(cfg) * cfg.collect.rollout_steps
        unused = self.steps_per_iteration % cfg.ppo.batch_size
        if unused:
            log(f"note: {unused} of {self.steps_per_iteration} learner steps per epoch do not fill a whole batch of "
                f"{cfg.ppo.batch_size} and are skipped (a different random subset each epoch)")

        self.iteration = 0
        self.learner_timesteps = 0
        self.env_steps = 0
        self.wall_time = 0.0
        self.opponent_rng_state = None
        resume_state = None
        if resume_from is not None:
            resume_state = ck.read_trainer_state(resume_from)
            self._check_resume_compatible(resume_state, allow_config_change)
            self.iteration = int(resume_state["iteration"])
            self.learner_timesteps = int(resume_state["learner_timesteps"])
            self.env_steps = int(resume_state["env_steps"])
            self.wall_time = float(resume_state.get("wall_time", 0.0))
            self.opponent_rng_state = resume_state.get("opponent_rng_state")

        torch.manual_seed(cfg.run.seed)
        np.random.seed(cfg.run.seed % (2 ** 32))

        self.learner = PPOLearner(self._actor_critic_factory, self._optimizers_factory)
        self.learner.load(DerivedPPOLearnerConfig(
            learner_config=PPOLearnerConfigModel(
                dtype=torch.float32,
                n_epochs=cfg.ppo.n_epochs,
                batch_size=cfg.ppo.batch_size,
                n_minibatches=cfg.ppo.n_minibatches,
                ent_coef=cfg.ppo.ent_coef,
                clip_range=cfg.ppo.clip_range,
                max_grad_norm=cfg.ppo.max_grad_norm,
                optimizer_named_parameter_group_kwargs={"actor": {"lr": cfg.ppo.actor_lr},
                                                        "critic": {"lr": cfg.ppo.critic_lr}},
                advantage_standardization=cfg.ppo.advantage_standardization,
                device=str(self.device),
            ),
            agent_controller_name=AGENT_CONTROLLER_NAME,
            obs_space=self.obs_space,
            action_space=self.action_space,
            checkpoint_load_folder=None if resume_from is None else str(Path(resume_from) / "ppo_learner"),
        ))
        self.policy: ActorCriticPolicy = self.learner.actor_critic  # type: ignore[assignment]

        self.trajectory_processor = GAETrajectoryProcessor()
        self.buffer = NumpyExperienceBuffer(self.trajectory_processor)
        self.buffer.load(DerivedExperienceBufferConfig(
            experience_buffer_config=ExperienceBufferConfigModel(
                max_size=self.steps_per_iteration,
                device="cpu",
                save_experience_buffer_in_checkpoint=False,
                trajectory_processor_config=GAETrajectoryProcessorConfigModel(
                    gamma=cfg.ppo.gamma,
                    lmbda=cfg.ppo.gae_lambda,
                    standardize_rewards=cfg.ppo.standardize_returns,
                    reward_clip=cfg.ppo.reward_clip,
                ),
            ),
            agent_controller_name=AGENT_CONTROLLER_NAME,
            obs_space=self.obs_space,
            action_space=self.action_space,
            seed=cfg.run.seed,
            dtype=torch.float32,
        ))
        if resume_from is not None:
            # rlgym-learn-algos 0.4.2's ExperienceBuffer.load does not forward the checkpoint folder to the
            # trajectory processor, so the return-normaliser statistics are restored explicitly here.
            tp_file = Path(resume_from) / "trajectory_processor" / TRAJECTORY_PROCESSOR_FILE
            stats = json.loads(tp_file.read_text(encoding="utf-8"))["return_running_stats"]
            if stats is not None:
                self.trajectory_processor.rust_gae_trajectory_processor.load_state_dict(stats)
            ck.restore_rng_state(torch.load(Path(resume_from) / ck.RNG_FILE, weights_only=False))

        self.opponents = OpponentRegistry(self.run_dir / "opponents")
        self.n_frozen = n_frozen_envs(cfg)
        self.opponent_rng = np.random.default_rng(cfg.run.seed + 17)
        if self.opponent_rng_state is not None:
            self.opponent_rng.bit_generator.state = self.opponent_rng_state
        self.frozen_runner: Optional[PolicyRunner] = None
        self.frozen_id: Optional[str] = None
        self.collector: Optional[Collector] = None
        self._stop_requested = False

    # ------------------------------------------------------------------ factories
    def _actor_critic_factory(self, obs_space, action_space, dtype, device, name):
        actor = build_actor(self.arch, obs_space[1], action_space[1])
        critic = build_critic(self.arch, obs_space[1])
        return ActorCriticPolicy(actor, critic, device)

    @staticmethod
    def _optimizers_factory(actor_critic, kwargs, name):
        return [
            torch.optim.Adam(actor_critic.actor.parameters(), **kwargs["actor"]),
            torch.optim.Adam(actor_critic.critic.parameters(), **kwargs["critic"]),
        ]

    # ------------------------------------------------------------------ resume checks
    def _check_resume_compatible(self, state: Dict[str, Any], allow_config_change: bool) -> None:
        saved_contract = state["contract"]
        now = contract(self.cfg.env.action_repeat, self.cfg.env.rlbot_delay)
        for key in ("obs_spec_hash", "action_table_sha256", "obs_size", "n_actions"):
            if saved_contract.get(key) != now.get(key):
                raise ck.CheckpointError(
                    f"cannot resume: {key} changed ({saved_contract.get(key)} -> {now.get(key)}). The checkpoint was "
                    "trained on a different observation/action contract; that needs retraining or a migration."
                )
        if state["architecture"] != self.arch:
            raise ck.CheckpointError(
                f"cannot resume: architecture changed from {state['architecture']} to {self.arch}. Layer changes do "
                "not preserve learned weights; start a new run or distil into the new architecture."
            )
        diffs = config_diff(state["config"], self.cfg.to_dict())
        if diffs and not allow_config_change:
            raise ck.CheckpointError(
                "cannot resume: the config differs from the checkpoint in ways that change what is learned:\n  "
                + "\n  ".join(diffs)
                + "\nRe-run with --allow-config-change if this is intentional (e.g. a curriculum stage change)."
            )
        if diffs:
            self.log("Resuming with intentional config changes:\n  " + "\n  ".join(diffs))

    # ------------------------------------------------------------------ inference callables
    @torch.no_grad()
    def _act_learner(self, obs: np.ndarray):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions, log_probs = self.policy.act(x, deterministic=False)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def _act_frozen(self, obs: np.ndarray) -> np.ndarray:
        assert self.frozen_runner is not None
        return self.frozen_runner.act_batch(obs)

    # ------------------------------------------------------------------ opponents
    def snapshot_opponent(self) -> None:
        opp_id = f"iter_{self.iteration:06d}"
        if any(e.id == opp_id for e in self.opponents.entries):
            return
        write_bundle(self.opponents.root / opp_id, self.policy.actor, self.arch, self.cfg.env.action_repeat,
                     self.cfg.env.rlbot_delay, self._source_info())
        self.opponents.add(OpponentEntry(id=opp_id, compat="sim", path=opp_id, added_iteration=self.iteration,
                                         note="frozen self-play snapshot"))

    def _select_frozen_opponent(self) -> None:
        if self.n_frozen == 0:
            return
        entry = self.opponents.sample(self.opponent_rng)
        if entry.id != self.frozen_id:
            bundle = load_bundle(self.opponents.resolve(entry), device=self.device)
            self.frozen_runner = PolicyRunner(bundle, device=str(self.device), deterministic=False)
            self.frozen_id = entry.id

    # ------------------------------------------------------------------ PPO update
    def _to_algos_trajectories(self, closed: List[ClosedTrajectory]) -> List[Trajectory]:
        trajectories = [
            Trajectory(c.agent_id, c.obs, c.actions, torch.from_numpy(c.log_probs), c.rewards, None, c.final_obs,
                       None, c.truncated)
            for c in closed
        ]
        # value predictions for every observation and every final observation, in one batched pass
        all_obs = []
        spans = []
        for t in trajectories:
            start = len(all_obs)
            all_obs.extend(t.obs_list)
            all_obs.append(t.final_obs)
            spans.append((start, len(all_obs)))
        obs = torch.as_tensor(np.asarray(all_obs, dtype=np.float32))
        values = []
        with torch.no_grad():
            for chunk in torch.split(obs, 8192):  # large attention batches exceed CUDA launch limits
                values.append(self.policy.critic(chunk.to(self.device)).float().cpu())
        values = torch.cat(values)
        for t, (start, stop) in zip(trajectories, spans):
            t.val_preds = values[start:stop - 1]
            t.final_val_pred = values[stop - 1]
        return trajectories

    LOGP_CHECK_SAMPLES = 4096
    LOGP_CHECK_TOLERANCE = 1e-3  # GPU batch-size-dependent kernels give ~1e-6; misalignment gives O(1)

    @torch.no_grad()
    def _check_log_prob_alignment(self) -> float:
        """Recompute log-probs for a random sample of this iteration's transitions with the (not yet updated)
        policy. They must equal the stored ones: this fails if observations, actions and log-probs are
        misaligned, or if the data came from a different policy version than the one being updated."""
        obs = self.buffer.observations.array()
        actions = self.buffer.actions.array()
        stored = self.buffer.log_probs.tensor()
        n = len(obs)
        idx = np.random.default_rng(self.iteration).choice(n, size=min(n, self.LOGP_CHECK_SAMPLES), replace=False)
        log_probs, _, _ = self.policy.get_backprop_data(None, obs[idx], actions[idx])
        err = float((log_probs.float().cpu() - stored[idx].float().cpu()).abs().max())
        if not err <= self.LOGP_CHECK_TOLERANCE:
            raise TrainingDivergedError(
                f"stored and recomputed log-probabilities differ by {err:.3g} at iteration {self.iteration}: the rollout "
                "data is misaligned or was not generated by the current policy. Not updating or saving.")
        return err

    def _assert_finite(self) -> None:
        for name, vec in (("actor", self.policy.get_actor_parameter_vector()),
                          ("critic", self.policy.get_critic_parameter_vector())):
            if not torch.isfinite(vec).all():
                raise TrainingDivergedError(f"{name} parameters became non-finite at iteration {self.iteration}; "
                                            "not saving. Resume from the last checkpoint with a lower learning rate.")

    # ------------------------------------------------------------------ checkpointing
    def _source_info(self) -> Dict[str, Any]:
        return {"run": self.run_dir.name, "iteration": self.iteration, "learner_timesteps": self.learner_timesteps}

    def trainer_state(self) -> Dict[str, Any]:
        return {
            "learner_timesteps": self.learner_timesteps,
            "env_steps": self.env_steps,
            "physics_ticks": self.env_steps * self.cfg.env.action_repeat,
            "policy_version": self.iteration,
            "cumulative_model_updates": self.learner.cumulative_model_updates,
            "wall_time": self.wall_time,
            "config": self.cfg.to_dict(),
            "architecture": self.arch,
            "contract": contract(self.cfg.env.action_repeat, self.cfg.env.rlbot_delay),
            "curriculum_stage": self.cfg.run.stage,
            "opponents": [e.id for e in self.opponents.sim_entries()],
            "frozen_opponent_fraction": self.cfg.self_play.frozen_opponent_fraction,
            "opponent_rng_state": self.opponent_rng.bit_generator.state,
            "seed": self.cfg.run.seed,
            "dependencies": dependency_versions(),
            "code": source_revision(),
            "resume_semantics": (
                "Exact: actor/critic weights, Adam states, return normaliser, counters, torch/numpy/python RNG. "
                "Not exact: environments restart from fresh episodes (worker seeds derived from the resume "
                "iteration), so the episode stream differs from an uninterrupted run."
            ),
        }

    def save(self) -> Path:
        def payload(tmp: Path) -> None:
            self.learner.save_checkpoint(tmp / "ppo_learner")
            (tmp / "trajectory_processor").mkdir()
            self.trajectory_processor.save_checkpoint(tmp / "trajectory_processor")

        checkpoints = self.run_dir / "checkpoints"
        if (checkpoints / ck.checkpoint_name(self.iteration)).exists():
            return checkpoints / ck.checkpoint_name(self.iteration)
        path = ck.save_checkpoint(checkpoints, self.iteration, payload, self.trainer_state())
        ck.write_pointer(self.run_dir, "latest", f"checkpoints/{path.name}")
        ck.prune_checkpoints(checkpoints, self.cfg.checkpoint.keep_last, ck.protected_paths(self.run_dir))
        return path

    # ------------------------------------------------------------------ main loop
    def request_stop(self, *_):
        self._stop_requested = True

    def train(self, max_iterations: Optional[int] = None) -> List[IterationReport]:
        cfg = self.cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.iteration == 0 and not (self.run_dir / "opponents" / "iter_000000").exists():
            self.snapshot_opponent()
        target = cfg.run.total_iterations if max_iterations is None else min(cfg.run.total_iterations,
                                                                              self.iteration + max_iterations)
        reports: List[IterationReport] = []
        if self.iteration >= target:
            self.log(f"Nothing to do: iteration {self.iteration} >= target {target}")
            return reports

        spec = EnvSpec.from_config(cfg)
        seed_base = worker_seed_base(cfg.run.seed, self.iteration, cfg.n_envs)
        process = psutil.Process()
        psutil.cpu_percent(None)
        stop_file = self.run_dir / "STOP"
        previous_handler = signal.signal(signal.SIGINT, self.request_stop)
        metrics_file = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        try:
            t_spawn = time.perf_counter()
            self.collector = Collector(spec, cfg.collect.n_workers, cfg.collect.envs_per_worker, seed_base)
            obs = self.collector.reset()
            roles = role_map(cfg.n_envs, self.collector.agents, self.n_frozen)
            self.log(f"Started {cfg.collect.n_workers} workers x {cfg.collect.envs_per_worker} envs "
                     f"({cfg.n_envs} envs, {self.n_frozen} with a frozen opponent) in {time.perf_counter() - t_spawn:.1f}s; "
                     f"{self.steps_per_iteration} learner steps/iteration on {self.device}; "
                     f"actor {count_parameters(self.policy.actor):,} params, critic {count_parameters(self.policy.critic):,}")
            while self.iteration < target and not self._stop_requested:
                if stop_file.exists():
                    self.log(f"STOP file found ({stop_file}); stopping after this point")
                    stop_file.unlink()
                    break
                t0 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                self._select_frozen_opponent()
                closed, obs, timing = run_rollout(
                    self.collector, obs, roles, cfg.collect.rollout_steps, self._act_learner,
                    self._act_frozen if self.n_frozen else None,
                )
                t1 = time.perf_counter()
                trajectories = self._to_algos_trajectories(closed)
                self.buffer.clear()
                gae = self.buffer.submit_experience(trajectories)
                logp_err = self._check_log_prob_alignment()
                t2 = time.perf_counter()
                ppo = self.learner.learn(self.buffer)
                t3 = time.perf_counter()
                self._assert_finite()

                self.iteration += 1
                self.learner_timesteps += timing["learner_steps"]
                env_steps = cfg.n_envs * cfg.collect.rollout_steps
                self.env_steps += env_steps
                reward_stats, episodes = self.collector.pop_stats()

                saved = None
                if self.iteration % cfg.self_play.snapshot_every_iterations == 0:
                    self.snapshot_opponent()
                if self.iteration % cfg.checkpoint.save_every_iterations == 0 or self.iteration >= target:
                    saved = self.save()
                t4 = time.perf_counter()
                self.wall_time += t4 - t0

                metrics = self._metrics(timing, gae, ppo, reward_stats, episodes, env_steps,
                                        collect_s=t1 - t0, gae_s=t2 - t1, update_s=t3 - t2, save_s=t4 - t3,
                                        total_s=t4 - t0, process=process)
                metrics["checks"] = {"logp_recompute_max_abs_err": logp_err}
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                reports.append(IterationReport(self.iteration, metrics))
                self.log(self._summary_line(metrics, saved))
            if self._stop_requested or self.iteration < target:
                path = self.save()
                self.log(f"Stopped at iteration {self.iteration}; checkpoint {path}")
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            metrics_file.close()
            if self.collector is not None:
                self.collector.close()
                self.collector = None
        return reports

    # ------------------------------------------------------------------ reporting
    def _metrics(self, timing, gae, ppo, reward_stats, episodes, env_steps, collect_s, gae_s, update_s, save_s,
                 total_s, process) -> Dict[str, Any]:
        cfg = self.cfg
        n_ep = len(episodes)
        goals = sum(1 for e in episodes if e["terminated"])
        per_scenario: Dict[str, int] = {}
        for e in episodes:
            per_scenario[e["scenario"]] = per_scenario.get(e["scenario"], 0) + 1
        agent_steps = max(1, reward_stats["agent_steps"])
        children = process.children(recursive=True)
        rss = process.memory_info().rss + sum(c.memory_info().rss for c in children if c.is_running())
        return {
            "iteration": self.iteration,
            "policy_version": self.iteration,
            "learner_timesteps": self.learner_timesteps,
            "env_steps": self.env_steps,
            "physics_ticks": self.env_steps * cfg.env.action_repeat,
            "game_seconds_simulated": self.env_steps * cfg.env.action_repeat / 120.0,
            "throughput": {
                "learner_steps_per_s": timing["learner_steps"] / total_s,
                "env_steps_per_s": env_steps / total_s,
                "physics_ticks_per_s": env_steps * cfg.env.action_repeat / total_s,
                "collection_learner_steps_per_s": timing["learner_steps"] / collect_s,
            },
            "timing_s": {"collect": collect_s, "inference": timing["inference_s"], "env_wait": timing["env_wait_s"],
                         "gae": gae_s, "update": update_s, "checkpoint": save_s, "total": total_s},
            "ppo": {
                "entropy": ppo.actor_entropy, "approx_kl": ppo.kl_divergence, "clip_fraction": ppo.sb3_clip_fraction,
                "value_loss": ppo.critic_loss, "actor_update_norm": ppo.actor_update_magnitude,
                "critic_update_norm": ppo.critic_update_magnitude, "model_updates": ppo.cumulative_model_updates,
                "batch_time_s": ppo.batch_consumption_time,
            },
            "returns": {"avg_reward_per_step": gae.average_reward,
                        "avg_undiscounted_episode_return": gae.average_undiscounted_episodic_return,
                        "avg_return": float(gae.average_return), "return_std": float(gae.return_standard_deviation)},
            "reward_per_agent_step": {
                "raw": {k: v / agent_steps for k, v in reward_stats["raw"].items()},
                "weighted": {k: v / agent_steps for k, v in reward_stats["weighted"].items()},
            },
            "episodes": {"finished": n_ep, "goal_fraction": goals / n_ep if n_ep else None,
                         "mean_length_steps": float(np.mean([e["length"] for e in episodes])) if n_ep else None,
                         "by_scenario": per_scenario},
            "frozen_opponent": self.frozen_id if self.n_frozen else None,
            "resources": {"cpu_percent_system": psutil.cpu_percent(None),
                          "ram_used_gb_system": psutil.virtual_memory().used / 2 ** 30,
                          "ram_rss_gb_tree": rss / 2 ** 30,
                          "vram_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None},
        }

    @staticmethod
    def _summary_line(m: Dict[str, Any], saved: Optional[Path]) -> str:
        p, t, th = m["ppo"], m["timing_s"], m["throughput"]
        goal = m["episodes"]["goal_fraction"]
        return (f"it {m['iteration']:5d} | steps {m['learner_timesteps']:>11,} | {th['learner_steps_per_s']:7.0f} sps "
                f"(collect {t['collect']:.1f}s, update {t['update']:.1f}s) | ent {p['entropy']:.3f} kl {p['approx_kl']:.4f} "
                f"clip {p['clip_fraction']:.3f} vloss {p['value_loss']:.3f} | ep {m['episodes']['finished']} "
                f"goal% {'-' if goal is None else f'{100 * goal:.0f}'}" + (f" | saved {saved.name}" if saved else ""))


def start_run(cfg: Config, run_dir: Path, config_text: Optional[str] = None) -> Trainer:
    run_dir = Path(run_dir)
    if ck.list_checkpoints(run_dir / "checkpoints"):
        raise ck.CheckpointError(f"{run_dir} already has checkpoints; use `resume` (or pick a new run name)")
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_text is not None:
        (run_dir / "config.toml").write_text(config_text, encoding="utf-8")
    return Trainer(cfg, run_dir)


def resume_run(run_dir: Path, cfg: Optional[Config] = None, checkpoint: Optional[Path] = None,
               allow_config_change: bool = False, log=print) -> Trainer:
    run_dir = Path(run_dir)
    ckpt = Path(checkpoint) if checkpoint else ck.resolve_checkpoint(run_dir)
    state = ck.read_trainer_state(ckpt)
    if cfg is None:
        cfg = config_from_dict(state["config"])
    return Trainer(cfg, run_dir, resume_from=ckpt, allow_config_change=allow_config_change, log=log)
