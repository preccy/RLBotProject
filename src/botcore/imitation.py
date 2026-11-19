"""Auxiliary imitation term on the actor (docs/EXPERIMENT_IL_PILOT.md).

Loss for one demonstration sample: -log sum_{a in class} pi(a | s), where `class` is the set of table rows that are
equivalent to the human's label in that state (`equivalence_mask`), so the policy is not pushed towards one of
several rows that do the same thing.

Integration without touching rlgym-learn-algos: the actor optimizer's `step` is wrapped. When the library calls it,
the PPO gradients of the update are already accumulated and clipped; the wrapper adds coef * grad(imitation loss) on
one demonstration minibatch, re-clips the actor gradient to `max_grad_norm`, and steps. It records the imitation
loss and the gradient norms of both parts, so the contribution of each is logged.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from botcore.actions.table import get_action_table


def equivalence_mask(T: np.ndarray, label: int, on_ground: bool) -> np.ndarray:
    """Rows that act like `label` in the given state. On the ground (no jump) only throttle, steer, boost and
    handbrake matter; in the air (no jump) only pitch, yaw, roll and boost; jump rows are compared on jump, pitch,
    roll, boost and the yaw+roll sum that sets a dodge's side component."""
    t = T[label]
    if t[5] > 0.5:
        return np.all(T[:, [2, 4, 5, 6]] == t[[2, 4, 5, 6]], axis=1) & (T[:, 3] + T[:, 4] == t[3] + t[4])
    if on_ground:
        return (T[:, 5] == 0) & np.all(T[:, [0, 1, 6, 7]] == t[[0, 1, 6, 7]], axis=1)
    return (T[:, 5] == 0) & np.all(T[:, [2, 3, 4, 6]] == t[[2, 3, 4, 6]], axis=1)


def load_demos(path: str, matches, on_ground_index: int):
    """Usable demonstration blocks from a build_demos.py npz: obs, labels and equivalence masks.
    Blocks flagged press_inflight or dodge_lost are dropped (approx_dir dodges are kept). matches=None: all."""
    d = np.load(path)
    in_matches = np.ones(len(d["match"]), bool) if matches is None else np.isin(d["match"], np.asarray(list(matches)))
    sel = in_matches & ~d["press_inflight"] & ~d["dodge_lost"]
    obs = d["obs"][sel].astype(np.float32)
    labels = d["label"][sel].astype(np.int64)
    T = get_action_table().controls
    on_ground = obs[:, on_ground_index] > 0.5
    masks = np.stack([equivalence_mask(T, int(l), bool(g)) for l, g in zip(labels, on_ground)])
    return obs, labels, masks


def class_nll(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Per-sample -log P(equivalence class)."""
    lse_all = torch.logsumexp(logits, dim=-1)
    lse_cls = torch.logsumexp(logits.masked_fill(~masks, float("-inf")), dim=-1)
    return lse_all - lse_cls


class ImitationTerm:
    def __init__(self, actor: torch.nn.Module, obs: np.ndarray, masks: np.ndarray, coef: float, batch_size: int,
                 max_grad_norm: Optional[float], device: torch.device, seed: int,
                 mix: Optional[tuple] = None):
        """mix = (obs2, masks2, n2): n2 of every batch come from the second source, batch_size - n2 from the first."""
        if coef <= 0:
            raise ValueError("imitation coef must be > 0")
        self.actor = actor
        self.obs = torch.as_tensor(obs, device=device)
        self.masks = torch.as_tensor(masks, device=device)
        self.coef = float(coef)
        self.mix = None
        if mix is not None:
            obs2, masks2, n2 = mix
            self.mix = (torch.as_tensor(obs2, device=device), torch.as_tensor(masks2, device=device), int(n2))
            batch_size = int(batch_size) - int(n2)
        self.batch_size = int(min(batch_size, len(obs)))
        self.max_grad_norm = max_grad_norm
        self.rng = np.random.default_rng(seed)
        self._stats: List[Dict[str, float]] = []

    def wrap(self, optimizer: torch.optim.Optimizer) -> None:
        original_step = optimizer.step
        params = [p for p in self.actor.parameters() if p.requires_grad]

        def step(*args, **kwargs):
            ppo_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            ppo_norm = torch.norm(torch.stack([g.norm() for g in ppo_grads]))
            for p in params:
                p.grad = None
            idx = torch.as_tensor(self.rng.choice(len(self.obs), self.batch_size, replace=False), device=self.obs.device)
            extra = {}
            if self.mix is None:
                nll = class_nll(self.actor(self.obs[idx]), self.masks[idx]).mean()
            else:
                obs2, masks2, n2 = self.mix
                idx2 = torch.as_tensor(self.rng.choice(len(obs2), n2, replace=False), device=obs2.device)
                per = class_nll(self.actor(torch.cat([self.obs[idx], obs2[idx2]])), torch.cat([self.masks[idx], masks2[idx2]]))
                nll = per.mean()
                extra = {"loss_primary": per[:len(idx)].mean().item(), "loss_mix": per[len(idx):].mean().item()}
            (self.coef * nll).backward()
            il_grads = [p.grad.detach() if p.grad is not None else torch.zeros_like(p) for p in params]
            il_norm = torch.norm(torch.stack([g.norm() for g in il_grads]))
            if not (torch.isfinite(nll) and torch.isfinite(il_norm)):
                raise FloatingPointError(f"imitation term not finite: loss {nll.item()}, grad norm {il_norm.item()}")
            cos = torch.nn.functional.cosine_similarity(torch.cat([g.flatten() for g in ppo_grads]),
                                                        torch.cat([g.flatten() for g in il_grads]), dim=0)
            for p, g_ppo, g_il in zip(params, ppo_grads, il_grads):
                p.grad = g_ppo + g_il
            combined = torch.norm(torch.stack([p.grad.norm() for p in params]))
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self._stats.append({"loss": nll.item(), "weighted_loss": self.coef * nll.item(),
                                "ppo_actor_grad_norm": ppo_norm.item(), "il_grad_norm": il_norm.item(),
                                "combined_grad_norm_before_clip": combined.item(), "cos_ppo_il": cos.item(), **extra})
            return original_step(*args, **kwargs)

        optimizer.step = step

    def pop_stats(self) -> Dict[str, float]:
        if not self._stats:
            return {}
        out = {k: float(np.mean([s[k] for s in self._stats])) for k in self._stats[0]}
        out["updates"] = len(self._stats)
        self._stats.clear()
        return out

    @torch.no_grad()
    def evaluate(self, obs: np.ndarray, masks: np.ndarray, chunk: int = 8192) -> float:
        """Mean -log P(class) on a fixed set (used for the preflight and for reporting)."""
        out = []
        for i in range(0, len(obs), chunk):
            o = torch.as_tensor(obs[i:i + chunk], device=self.obs.device)
            m = torch.as_tensor(masks[i:i + chunk], device=self.obs.device)
            out.append(class_nll(self.actor(o), m))
        return float(torch.cat(out).mean())
