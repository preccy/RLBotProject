"""Typed configuration: TOML -> validated dataclasses.

Validation happens before any process is spawned so incompatible settings fail in milliseconds, not
after an hour of training. Unknown keys are errors (a typo must never silently fall back to a default).
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from botcore import constants as C

# Evaluation seeds live in a reserved range that training/curriculum seeds may never enter.
EVAL_SEED_BASE = 1_000_000
MAX_TRAIN_SEED = EVAL_SEED_BASE - 1


class ConfigError(ValueError):
    """A configuration value is missing, unknown or inconsistent."""


@dataclass
class EnvConfig:
    team_size: int = 1
    action_repeat: int = 8
    rlbot_delay: bool = True  # RocketSimEngine one-tick input delay; must match the runner's step offset
    no_touch_timeout_seconds: float = 30.0
    episode_timeout_seconds: float = 300.0
    scenario_weights: Dict[str, float] = field(default_factory=lambda: {
        "kickoff": 0.50, "open_play": 0.20, "shot": 0.08, "save": 0.08,
        "low_boost_defence": 0.04, "recovery": 0.05, "aerial": 0.05,
    })


@dataclass
class RewardConfig:
    team_spirit: float = 0.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "goal_scored": 3.0,
        "goal_conceded": 3.0,
        "ball_touch": 0.5,
        "velocity_to_ball": 0.5,
        "ball_to_goal": 1.5,
        "boost_economy": 0.1,
    })


@dataclass
class ModelConfig:
    kind: str = "mlp"  # "mlp" | "entity"
    actor_layers: Tuple[int, ...] = (512, 512, 512)
    critic_layers: Tuple[int, ...] = (512, 512, 512)
    # entity encoder only
    embed_dim: int = 128
    n_heads: int = 4
    n_attention_layers: int = 2
    head_layers: Tuple[int, ...] = (256, 256)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 2
    batch_size: int = 24_576
    n_minibatches: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    advantage_standardization: bool = True
    standardize_returns: bool = True
    reward_clip: float = 10.0


@dataclass
class CollectConfig:
    n_workers: int = 12
    envs_per_worker: int = 6
    rollout_steps: int = 512  # decisions per environment per iteration


@dataclass
class SelfPlayConfig:
    # Fraction of environments in which the orange side is a frozen historical checkpoint instead of the
    # current policy. 0 = pure current-policy self-play.
    frozen_opponent_fraction: float = 0.0
    snapshot_every_iterations: int = 50  # how often a checkpoint is also kept as a historical opponent
    # Fraction of environments in which one side is the frozen external bot Necto V1 (botcore.necto), driven
    # inside the workers from the game state; its car produces no learner experience. 0 = off. 1v1 only.
    necto_fraction: float = 0.0
    necto_dir: str = "external/Necto/rlbot-support/Necto"


@dataclass
class ImitationConfig:
    # Auxiliary imitation term on the actor (botcore.imitation, docs/EXPERIMENT_IL_PILOT.md). coef 0 = off.
    coef: float = 0.0
    dataset: str = ""  # npz written by scripts/il_feasibility/build_demos.py
    matches: tuple = ()  # match indices in the npz used for training (held-out matches are simply not listed)
    batch_size: int = 2048


@dataclass
class CheckpointConfig:
    save_every_iterations: int = 10
    keep_last: int = 5


@dataclass
class EvalConfig:
    n_seed_pairs: int = 16
    match_seconds: float = 300.0
    overtime_max_seconds: float = 120.0
    deterministic: bool = False


@dataclass
class RunConfig:
    name: str = "default"
    seed: int = 1
    device: str = "auto"  # "auto" | "cpu" | "cuda:0"
    total_iterations: int = 100_000
    runs_dir: str = "runs"
    stage: str = "s1-contact"  # curriculum stage label recorded in checkpoints (see docs/DESIGN.md)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ derived
    @property
    def agents_per_env(self) -> int:
        return 2 * self.env.team_size

    @property
    def n_envs(self) -> int:
        return self.collect.n_workers * self.collect.envs_per_worker

    @property
    def decision_hz(self) -> float:
        return C.TICKS_PER_SECOND / self.env.action_repeat

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_SECTIONS = {f.name: f.type for f in dataclasses.fields(Config)}


def _coerce(section: str, key: str, value: Any, default: Any) -> Any:
    where = f"[{section}] {key}"
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true/false, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            raise ConfigError(f"{where} must be a list of integers, got {value!r}")
        return tuple(value)
    if isinstance(default, dict):
        if not isinstance(value, dict) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()
        ):
            raise ConfigError(f"{where} must be a table of numbers, got {value!r}")
        return {str(k): float(v) for k, v in value.items()}
    raise ConfigError(f"{where}: unsupported config type {type(default).__name__}")


def config_from_dict(data: Dict[str, Any]) -> Config:
    cfg = Config()
    unknown_sections = sorted(set(data) - set(_SECTIONS))
    if unknown_sections:
        raise ConfigError(f"unknown config section(s) {unknown_sections}; valid: {sorted(_SECTIONS)}")
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] must be a table")
        target = getattr(cfg, section)
        valid = {f.name for f in dataclasses.fields(target)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ConfigError(f"unknown key(s) {unknown} in [{section}]; valid: {sorted(valid)}")
        for key, value in values.items():
            setattr(target, key, _coerce(section, key, value, getattr(target, key)))
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    for section, values in (overrides or {}).items():
        data.setdefault(section, {}).update(values)
    return config_from_dict(data)


def validate(cfg: Config) -> None:
    """Raise ConfigError for anything that would make a run fail late or train on a wrong contract."""
    from botcore.rewards.components import COMPONENTS
    from botcore.scenarios.mutators import SCENARIOS

    def need(cond: bool, message: str) -> None:
        if not cond:
            raise ConfigError(message)

    e, p, c, m = cfg.env, cfg.ppo, cfg.collect, cfg.model
    need(1 <= e.team_size <= C.MAX_TEAM_SIZE,
         f"[env] team_size={e.team_size} unsupported: the obs-v1 contract covers 1v1 to {C.MAX_TEAM_SIZE}v{C.MAX_TEAM_SIZE}")
    need(1 <= e.action_repeat <= 120, "[env] action_repeat must be between 1 and 120 physics ticks")
    need(e.no_touch_timeout_seconds > 0 and e.episode_timeout_seconds > 0, "[env] timeouts must be positive")
    bad = sorted(set(e.scenario_weights) - set(SCENARIOS))
    need(not bad, f"[env.scenario_weights] unknown scenario(s) {bad}; valid: {sorted(SCENARIOS)}")
    need(all(w >= 0 for w in e.scenario_weights.values()) and sum(e.scenario_weights.values()) > 0,
         "[env.scenario_weights] weights must be >= 0 with a positive sum")
    need(e.scenario_weights.get("kickoff", 0.0) > 0,
         "[env.scenario_weights] kickoff must keep a positive weight so ordinary gameplay stays represented")

    bad = sorted(set(cfg.rewards.weights) - set(COMPONENTS))
    need(not bad, f"[rewards.weights] unknown component(s) {bad}; valid: {sorted(COMPONENTS)}")
    need(0.0 <= cfg.rewards.team_spirit <= 1.0, "[rewards] team_spirit must be in [0, 1]")

    need(m.kind in ("mlp", "entity"), f"[model] kind must be 'mlp' or 'entity', got {m.kind!r}")
    need(len(m.actor_layers) > 0 and len(m.critic_layers) > 0 and all(s > 0 for s in m.actor_layers + m.critic_layers),
         "[model] actor_layers/critic_layers need at least one positive layer size")
    if m.kind == "entity":
        need(m.embed_dim > 0 and m.n_heads > 0 and m.embed_dim % m.n_heads == 0,
             f"[model] embed_dim ({m.embed_dim}) must be divisible by n_heads ({m.n_heads})")
        need(m.n_attention_layers >= 1, "[model] n_attention_layers must be >= 1")
        need(len(m.head_layers) > 0 and all(s > 0 for s in m.head_layers), "[model] head_layers needs positive sizes")

    need(0.0 < p.gamma < 1.0 and 0.0 <= p.gae_lambda <= 1.0, "[ppo] gamma must be in (0,1) and gae_lambda in [0,1]")
    need(p.n_epochs >= 1 and p.n_minibatches >= 1, "[ppo] n_epochs and n_minibatches must be >= 1")
    need(p.actor_lr > 0 and p.critic_lr > 0, "[ppo] learning rates must be positive")
    need(0.0 < p.clip_range < 1.0, "[ppo] clip_range must be in (0, 1)")
    need(p.max_grad_norm > 0 and p.reward_clip > 0, "[ppo] max_grad_norm and reward_clip must be positive")

    need(c.n_workers >= 1 and c.envs_per_worker >= 1 and c.rollout_steps >= 2,
         "[collect] n_workers, envs_per_worker must be >= 1 and rollout_steps >= 2")
    f = cfg.self_play.frozen_opponent_fraction
    need(0.0 <= f < 1.0, "[self_play] frozen_opponent_fraction must be in [0, 1)")
    need(cfg.self_play.snapshot_every_iterations >= 1, "[self_play] snapshot_every_iterations must be >= 1")
    nf = cfg.self_play.necto_fraction
    need(0.0 <= nf < 1.0, "[self_play] necto_fraction must be in [0, 1)")
    if nf > 0:
        need(f == 0.0, "[self_play] necto_fraction and frozen_opponent_fraction cannot both be > 0")
        need(e.team_size == 1, "[self_play] necto_fraction needs team_size = 1")
        need(e.action_repeat == 8, "[self_play] Necto decides every 8 ticks; action_repeat must be 8")
    need(cfg.imitation.coef >= 0.0, "[imitation] coef must be >= 0")
    if cfg.imitation.coef > 0:
        need(bool(cfg.imitation.dataset), "[imitation] dataset is required when coef > 0")
        need(len(cfg.imitation.matches) > 0, "[imitation] matches must list the training match indices")
        need(cfg.imitation.batch_size >= 1, "[imitation] batch_size must be >= 1")

    learner_agents = learner_agent_slots(cfg)
    steps_per_iteration = learner_agents * c.rollout_steps
    need(p.batch_size >= 1, "[ppo] batch_size must be >= 1")
    need(p.batch_size <= steps_per_iteration,
         f"[ppo] batch_size={p.batch_size} exceeds the {steps_per_iteration} learner timesteps collected per "
         f"iteration ({learner_agents} learner agents x {c.rollout_steps} rollout_steps); no gradient step would run")

    need(cfg.checkpoint.save_every_iterations >= 1 and cfg.checkpoint.keep_last >= 1,
         "[checkpoint] save_every_iterations and keep_last must be >= 1")
    need(0 <= cfg.run.seed <= MAX_TRAIN_SEED,
         f"[run] seed must be in 0..{MAX_TRAIN_SEED}; seeds from {EVAL_SEED_BASE} upwards are reserved for evaluation")
    need(cfg.run.total_iterations >= 1, "[run] total_iterations must be >= 1")
    need(cfg.run.device == "auto" or cfg.run.device == "cpu" or cfg.run.device.startswith("cuda"),
         f"[run] device must be 'auto', 'cpu' or 'cuda[:N]', got {cfg.run.device!r}")
    need(cfg.eval.n_seed_pairs >= 1 and cfg.eval.match_seconds > 0 and cfg.eval.overtime_max_seconds >= 0,
         "[eval] n_seed_pairs >= 1, match_seconds > 0, overtime_max_seconds >= 0 required")


def n_frozen_envs(cfg: Config) -> int:
    """Number of environments whose orange side is driven by a frozen opponent."""
    return int(round(cfg.self_play.frozen_opponent_fraction * cfg.n_envs))


def n_necto_envs(cfg: Config) -> int:
    """Number of environments in which one side is the frozen external bot Necto."""
    return int(round(cfg.self_play.necto_fraction * cfg.n_envs))


def learner_agent_slots(cfg: Config) -> int:
    """Agent slots per rollout step that produce learner experience."""
    frozen = n_frozen_envs(cfg) + n_necto_envs(cfg)
    return (cfg.n_envs - frozen) * cfg.agents_per_env + frozen * cfg.env.team_size
