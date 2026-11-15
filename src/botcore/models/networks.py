"""Actor/critic network bodies.

Two interchangeable trunks map the flat obs-v1 vector to a feature vector:

* ``MLPTrunk``    - the baseline: a plain feed-forward network over the whole vector. Fast and simple,
                    but it sees car slots positionally (slot order matters) and has a fixed player cap
                    baked into its first layer.
* ``EntityTrunk`` - a compact entity encoder: the controlled car is a query that cross-attends over
                    ball, car and boost-pad tokens with a padding mask for absent cars. Parameters are
                    shared across cars/pads, so it is insensitive to slot order and to player count.
                    It deliberately has no entity-to-entity self-attention yet; relations between
                    entities come from the engineered relative features (documented growth path).

Both consume exactly the same observation contract, so switching between them changes checkpoint
compatibility but not the environment, runner or export format.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from torch import nn

from botcore import constants as C
from botcore.obs import spec as S


def _init_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int | None = None, out_gain: float = 1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for size in hidden:
        linear = nn.Linear(prev, size)
        _init_linear(linear, gain=float(np.sqrt(2.0)))
        layers += [linear, nn.ReLU()]
        prev = size
    if out_dim is not None:
        head = nn.Linear(prev, out_dim)
        _init_linear(head, gain=out_gain)
        layers.append(head)
    return nn.Sequential(*layers)


class MLPTrunk(nn.Module):
    def __init__(self, obs_size: int, layers: Sequence[int]):
        super().__init__()
        self.net = make_mlp(obs_size, layers)
        self.out_dim = int(layers[-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class EntityTrunk(nn.Module):
    """Self-as-query cross-attention over ball, car and pad tokens (masked for absent cars)."""

    PAD_TOKEN_DIM = 9  # timer, pad xyz, is_big, pad-minus-self xyz, distance

    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, head_layers: Sequence[int]):
        super().__init__()
        layout = S.LAYOUT
        self.self_slice = layout.self_slice
        self.ball_slice = layout.ball_slice
        self.cars_slice = layout.cars_slice
        self.pads_slice = layout.pads_slice
        self.n_car_slots = layout.n_car_slots
        self.car_size = layout.car_size
        self.present_index = S.CAR_IDX["present"].start
        self.self_pos = S.SELF_IDX["pos"]

        # Team-relative pad geometry is identical for both teams (the layout is 180-degree symmetric).
        pad_pos = torch.tensor(BOOST_LOCATIONS, dtype=torch.float32) / S.POS_SCALE
        is_big = torch.zeros(C.NUM_BOOST_PADS, 1)
        is_big[list(C.BIG_PAD_INDICES)] = 1.0
        self.register_buffer("pad_pos", pad_pos, persistent=False)
        self.register_buffer("pad_is_big", is_big, persistent=False)

        d = embed_dim
        self.self_embed = make_mlp(layout.self_size, [d], d)
        self.ball_embed = make_mlp(layout.ball_size, [d], d)
        self.car_embed = make_mlp(layout.car_size, [d], d)
        self.pad_embed = make_mlp(self.PAD_TOKEN_DIM, [d], d)
        self.type_embed = nn.Parameter(torch.zeros(3, d))  # ball, car, pad
        nn.init.normal_(self.type_embed, std=0.02)

        self.attn = nn.ModuleList(nn.MultiheadAttention(d, n_heads, batch_first=True) for _ in range(n_layers))
        self.attn_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.ff = nn.ModuleList(make_mlp(d, [2 * d], d) for _ in range(n_layers))
        self.ff_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.head = make_mlp(2 * d, head_layers)
        self.out_dim = int(head_layers[-1])

    def tokens(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (query [B,1,d], entity tokens [B,T,d], key padding mask [B,T] with True = ignore)."""
        batch = obs.shape[0]
        me = obs[:, self.self_slice]
        cars = obs[:, self.cars_slice].reshape(batch, self.n_car_slots, self.car_size)
        timers = obs[:, self.pads_slice].unsqueeze(-1)

        pad_pos = self.pad_pos.unsqueeze(0).expand(batch, -1, -1)
        rel = pad_pos - me[:, self.self_pos].unsqueeze(1)
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        pad_tokens = torch.cat([timers, pad_pos, self.pad_is_big.unsqueeze(0).expand(batch, -1, -1), rel, dist], dim=-1)

        tokens = torch.cat(
            [
                self.ball_embed(obs[:, self.ball_slice]).unsqueeze(1) + self.type_embed[0],
                self.car_embed(cars) + self.type_embed[1],
                self.pad_embed(pad_tokens) + self.type_embed[2],
            ],
            dim=1,
        )
        absent = cars[:, :, self.present_index] < 0.5
        mask = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.bool, device=obs.device),  # the ball is always present
                absent,
                torch.zeros(batch, C.NUM_BOOST_PADS, dtype=torch.bool, device=obs.device),
            ],
            dim=1,
        )
        return self.self_embed(me).unsqueeze(1), tokens, mask

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        query, tokens, mask = self.tokens(obs)
        latent = query
        for attn, attn_norm, ff, ff_norm in zip(self.attn, self.attn_norm, self.ff, self.ff_norm):
            attended, _ = attn(latent, tokens, tokens, key_padding_mask=mask, need_weights=False)
            latent = attn_norm(latent + attended)
            latent = ff_norm(latent + ff(latent))
        return self.head(torch.cat([latent.squeeze(1), query.squeeze(1)], dim=-1))


class Actor(nn.Module):
    """Trunk + linear head producing unnormalised action logits."""

    def __init__(self, trunk: nn.Module, n_actions: int):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(trunk.out_dim, n_actions)
        _init_linear(self.head, gain=0.01)  # near-uniform initial policy
        self.n_actions = n_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


class Critic(nn.Module):
    """Trunk + scalar state-value head. Receives the same (deployable) observation as the actor."""

    def __init__(self, trunk: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(trunk.out_dim, 1)
        _init_linear(self.head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs)).squeeze(-1)
