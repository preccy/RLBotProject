import numpy as np
import torch

from botcore.actions.table import get_action_table
from botcore.config import Config, config_from_dict
from botcore.imitation import ImitationTerm, class_nll, equivalence_mask
from botcore.trainer import config_diff

T = get_action_table().controls


def test_class_nll_is_minus_log_of_summed_class_probability():
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    mask = torch.tensor([[True, False, True, False]])
    p = torch.softmax(logits, -1)[0]
    assert torch.allclose(class_nll(logits, mask)[0], -torch.log(p[0] + p[2]))


def test_equivalence_mask_contains_label_and_merges_ground_rows():
    for label in range(len(T)):
        for g in (True, False):
            assert equivalence_mask(T, label, g)[label]
    # on the ground, throttle+boost with nose down is the same as plain throttle+boost
    ground = [i for i in range(len(T)) if list(T[i]) == [1, 0, 0, 0, 0, 0, 1, 0]][0]
    air = [i for i in range(len(T)) if list(T[i]) == [1, 0, -1, 0, 0, 0, 1, 0]][0]
    assert equivalence_mask(T, ground, True)[air]
    assert not equivalence_mask(T, ground, False)[air]


def test_step_wrapper_adds_scaled_imitation_gradient_and_logs_it():
    torch.manual_seed(0)
    actor = torch.nn.Linear(4, len(T))
    obs = np.random.default_rng(0).normal(size=(16, 4)).astype(np.float32)
    masks = np.zeros((16, len(T)), bool)
    masks[:, 3] = True
    class Recording(torch.optim.SGD):
        def step(self, *a, **k):
            self.seen = actor.weight.grad.clone()
            return super().step(*a, **k)

    # expected gradients, computed separately (batch = all 16 samples, so the minibatch mean is the same)
    actor.zero_grad()
    (actor(torch.as_tensor(obs)).sum() * 0.01).backward()
    g_ppo = actor.weight.grad.clone()
    actor.zero_grad()
    (0.5 * class_nll(actor(torch.as_tensor(obs)), torch.as_tensor(masks)).mean()).backward()
    g_il = actor.weight.grad.clone()

    opt = Recording(actor.parameters(), lr=0.0)  # lr 0: inspect gradients only
    term = ImitationTerm(actor, obs, masks, coef=0.5, batch_size=16, max_grad_norm=None,
                         device=torch.device("cpu"), seed=0)
    term.wrap(opt)
    opt.zero_grad()
    (actor(torch.as_tensor(obs)).sum() * 0.01).backward()  # stands in for the PPO gradient
    opt.step()
    assert torch.allclose(opt.seen, g_ppo + g_il, atol=1e-6)
    stats = term.pop_stats()
    assert stats["updates"] == 1 and np.isfinite(stats["loss"]) and stats["il_grad_norm"] > 0


def test_config_without_imitation_section_resumes_without_diff():
    old = Config().to_dict()
    old.pop("imitation")
    assert config_diff(old, Config().to_dict()) == []
    cfg = config_from_dict({"imitation": {"coef": 0.02, "dataset": "x.npz", "matches": [0, 1]}})
    assert any("[imitation] coef" in d for d in config_diff(old, cfg.to_dict()))
