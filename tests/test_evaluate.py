"""Evaluation harness: reproducibility, colour swapping, scoring, uncertainty, promotion rule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from botcore.config import EVAL_SEED_BASE, Config
from botcore.evaluate import (
    MatchResult,
    MatchRules,
    Side,
    bootstrap_ci,
    check_promotion,
    eval_seeds,
    play_match,
    summarise,
)
from botcore.export import write_bundle
from botcore.models import arch_from_config, build_actor
from botcore.obs import OBS_SIZE


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundles")
    cfg = Config()
    cfg.model.actor_layers = (32, 32)
    arch = arch_from_config(cfg.model)
    out = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        out.append(str(write_bundle(root / f"p{seed}", build_actor(arch, OBS_SIZE, 90), arch, 8, True, {})))
    return out


def test_eval_seeds_are_disjoint_from_training_seeds():
    seeds = eval_seeds(50)
    assert min(seeds) == EVAL_SEED_BASE and len(set(seeds)) == 50


def test_matches_are_reproducible_and_respect_the_clock(bundles):
    rules = MatchRules(match_seconds=30.0, overtime_max_seconds=10.0)
    a, b = Side(bundles[0]), Side(bundles[1])
    r1 = play_match(EVAL_SEED_BASE, a, b, "A", rules)
    r2 = play_match(EVAL_SEED_BASE, a, b, "A", rules)
    assert r1 == r2
    assert 30.0 <= r1.seconds_played <= 40.0 + 8 / 120
    if r1.goals_blue == r1.goals_orange:
        assert r1.overtime


def test_scoring_from_each_side():
    r = MatchResult(seed=1, blue="A", goals_blue=3, goals_orange=1, seconds_played=300, overtime=False)
    assert r.score_for("A") == 1.0 and r.score_for("B") == 0.0 and r.goal_diff_for("B") == -2
    swapped = MatchResult(seed=1, blue="B", goals_blue=2, goals_orange=2, seconds_played=420, overtime=True)
    assert swapped.score_for("A") == 0.5


def test_summary_uses_seed_pairs_as_the_sampling_unit():
    results = []
    seeds = [10, 11, 12]
    for s in seeds:
        results.append(MatchResult(s, "A", 5, 0, 300, False))  # A wins as blue
        results.append(MatchResult(s, "B", 1, 0, 300, False))  # A loses as orange
    summary = summarise(results, seeds)
    assert summary["A_score"] == 0.5 and summary["A_wins"] == 3 and summary["A_losses"] == 3
    assert summary["A_score_ci95_pair_bootstrap"] == (0.5, 0.5)  # every pair scores exactly 0.5
    assert summary["A_score_as_blue"] == 1.0 and summary["A_score_as_orange"] == 0.0
    assert summary["A_goals_for"] == 15 and summary["A_goals_against"] == 3


def test_bootstrap_interval_narrows_with_more_pairs():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(rng.integers(0, 2, 10).astype(float))
    large = bootstrap_ci(rng.integers(0, 2, 1000).astype(float))
    assert (large[1] - large[0]) < (small[1] - small[0])


def _report(a_sha, b_sha, pairs, score, lo):
    return {"created_at": "t", "sides": {"A": {"weights_sha256": a_sha}, "B": {"weights_sha256": b_sha}},
            "summary": {"seed_pairs": pairs, "A_score": score, "A_score_ci95_pair_bootstrap": (lo, lo + 0.1)}}


def test_promotion_rule():
    good = [_report("cand", "best", 120, 0.6, 0.55), _report("cand", "old", 120, 0.52, 0.45)]
    assert check_promotion("cand", "best", good) == []
    assert any("lower bound" in p for p in check_promotion("cand", "best", [_report("cand", "best", 120, 0.53, 0.49)]))
    assert any("seed pairs" in p for p in check_promotion("cand", "best", [_report("cand", "best", 10, 0.9, 0.8)]))
    assert any("regression" in p for p in check_promotion("cand", "best", good + [_report("cand", "x", 120, 0.4, 0.3)]))
    assert any("current best" in p for p in check_promotion("cand", "best", [_report("cand", "x", 120, 0.7, 0.6)]))
