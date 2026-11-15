"""Architecture description, construction and the PPO actor-critic wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rlgym_learn_algos.ppo import ActorCritic
from torch import nn

from botcore.config import ModelConfig
from botcore.models.networks import Actor, Critic, EntityTrunk, MLPTrunk

ARCH_SCHEMA = 1


class ArchitectureError(ValueError):
    """An architecture description cannot be built by this code version."""


def arch_from_config(cfg: ModelConfig) -> dict:
    """JSON-serialisable architecture description stored in checkpoints and export manifests."""
    if cfg.kind == "mlp":
        return {
            "schema": ARCH_SCHEMA,
            "kind": "mlp",
            "actor_layers": list(cfg.actor_layers),
            "critic_layers": list(cfg.critic_layers),
        }
    if cfg.kind == "entity":
        return {
            "schema": ARCH_SCHEMA,
            "kind": "entity",
            "embed_dim": cfg.embed_dim,
            "n_heads": cfg.n_heads,
            "n_attention_layers": cfg.n_attention_layers,
            "head_layers": list(cfg.head_layers),
        }
    raise ArchitectureError(f"unknown model kind {cfg.kind!r}")


def _trunk(arch: dict, role: str, obs_size: int) -> nn.Module:
    if arch.get("schema") != ARCH_SCHEMA:
        raise ArchitectureError(f"architecture schema {arch.get('schema')!r} is not supported (expected {ARCH_SCHEMA})")
    kind = arch.get("kind")
    try:
        if kind == "mlp":
            return MLPTrunk(obs_size, arch[f"{role}_layers"])
        if kind == "entity":
            return EntityTrunk(arch["embed_dim"], arch["n_heads"], arch["n_attention_layers"], arch["head_layers"])
    except KeyError as exc:
        raise ArchitectureError(f"architecture description for kind {kind!r} is missing {exc}") from exc
    raise ArchitectureError(f"unknown model kind {kind!r}")


def build_actor(arch: dict, obs_size: int, n_actions: int) -> Actor:
    return Actor(_trunk(arch, "actor", obs_size), n_actions)


def build_critic(arch: dict, obs_size: int) -> Critic:
    return Critic(_trunk(arch, "critic", obs_size))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sample_actions(logits: torch.Tensor, deterministic: bool,
                   generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (actions [n], log-probabilities [n]) from unnormalised logits.

    The same function is used for training rollouts, evaluation and deployment, so "stochastic" means the
    identical categorical distribution everywhere; "deterministic" is argmax of the same logits.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if deterministic:
        actions = log_probs.argmax(dim=-1)
    else:
        actions = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)
    return actions, log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


class ActorCriticPolicy(ActorCritic[Any, np.ndarray, np.ndarray]):
    """Separate actor and critic behind the rlgym-learn-algos `ActorCritic` interface used by `PPOLearner`.

    There are no action masks in this version: every entry of the lookup table is legal in every state
    (some are merely redundant, e.g. boost with an empty tank), and masking redundant actions would need
    flip/boost state that is only reconstructed at deployment. If masks are added later they must be
    derived from the observation inside this class so sampling, log-probability and entropy all see
    the same mask.
    """

    def __init__(self, actor: Actor, critic: Critic, device: torch.device | str):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.device = torch.device(device)
        self.to(self.device)

    def _obs(self, obs_list) -> torch.Tensor:
        if isinstance(obs_list, torch.Tensor):
            return obs_list.to(self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------ rlgym-learn-algos interface
    def get_actions(self, agent_id_list, obs_list, **kwargs):
        actions, log_probs = sample_actions(self.actor(self._obs(obs_list)), bool(kwargs.get("deterministic")))
        return actions.unsqueeze(-1).cpu().numpy(), log_probs.cpu()

    def get_value_predictions(self, agent_id_list, obs_list) -> torch.Tensor:
        return self.critic(self._obs(obs_list))

    def get_backprop_data(self, agent_id_list: Sequence, obs_list, action_list, **kwargs):
        obs = self._obs(obs_list)
        if isinstance(action_list, torch.Tensor):
            actions = action_list.to(self.device)
        else:
            actions = torch.as_tensor(np.asarray(action_list), device=self.device)
        actions = actions.reshape(-1, 1).long()
        log_probs_all = F.log_softmax(self.actor(obs), dim=-1)
        log_probs = log_probs_all.gather(-1, actions).squeeze(-1)
        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1).mean()
        return log_probs, entropy, self.critic(obs)

    def get_actor_parameter_vector(self) -> torch.Tensor:
        return nn.utils.parameters_to_vector(self.actor.parameters())

    def get_critic_parameter_vector(self) -> torch.Tensor:
        return nn.utils.parameters_to_vector(self.critic.parameters())

    # ------------------------------------------------------------ batched rollout inference
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_actions(self.actor(obs), deterministic)
