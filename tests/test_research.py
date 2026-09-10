import copy
from argparse import Namespace

import numpy as np
import pytest
import torch

from nfsp.arena import Policy, checkpoint_policies, evaluate_match, heuristic_probabilities, pair_summary
from nfsp.config import Config
from nfsp.experiments import DQNTrainer
from nfsp.performance import rollout
from nfsp.ppo import PPOConfig, PPOTrainer, gae, legal_distribution, normalized_entropy


def tiny(**kwargs):
    values = dict(seed=91, num_envs=8, workers=2, hidden=(32,), device="cpu",
                  torch_threads=1, epochs=2, minibatch_size=32, deterministic=True)
    values.update(kwargs)
    return PPOConfig(**values)


def test_gae_terminal_no_cross_player_bootstrap():
    advantages, returns = gae([0, 0, 1], [.2, -.3, .1], 1., 1.)
    np.testing.assert_allclose(returns, [1, 1, 1], atol=1e-7)
    np.testing.assert_allclose(advantages, [.8, 1.3, .9], atol=1e-7)
    _, negative = gae([0, -1], [3, 4], .5, 1.)
    np.testing.assert_allclose(negative, [-.5, -1])


def test_masked_entropy_forced_moves_and_finite_gradients():
    logits = torch.zeros((3, 28), requires_grad=True)
    masks = torch.zeros((3, 28), dtype=torch.bool)
    masks[0, 2] = True
    masks[1, :2] = True
    masks[2, :10] = True
    dist = legal_distribution(logits, masks)
    entropy, choices = normalized_entropy(dist, masks)
    torch.testing.assert_close(entropy, torch.tensor([0., 1., 1.]))
    assert choices.tolist() == [False, True, True]
    assert not dist.probs[~masks].any()
    (entropy.sum() + dist.log_prob(torch.tensor([2, 0, 0])).sum()).backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[~masks].any()
    with pytest.raises(ValueError):
        legal_distribution(logits[:1], torch.zeros_like(masks[:1]))


def test_rollouts_assign_terminal_rewards_and_each_action_once():
    trainer = PPOTrainer(tiny(num_envs=64, gae_lambda=1.))
    batch, metrics = trainer.collect(64)
    assert len(batch["actions"]) == metrics["decisions"]
    assert batch["masks"][np.arange(len(batch["actions"])), batch["actions"]].all()
    assert batch["rewards"].sum() == 0
    assert np.count_nonzero(batch["rewards"]) == 128
    for tid in np.unique(batch["trajectory_ids"]):
        ids = np.flatnonzero(batch["trajectory_ids"] == tid)
        assert np.count_nonzero(batch["rewards"][ids]) == 1
        assert batch["rewards"][ids[-1]] in (-1, 1)
        np.testing.assert_allclose(batch["returns"][ids], batch["rewards"][ids[-1]], atol=2e-6)
    # Exchanges must connect to the same player's next decision without changing scores/talon.
    exchanges = np.flatnonzero(batch["actions"] >= 24)
    assert len(exchanges)
    for index in exchanges:
        ids = np.flatnonzero((batch["trajectory_ids"] == batch["trajectory_ids"][index]) & (np.arange(len(batch["actions"])) > index))
        assert len(ids)
        np.testing.assert_array_equal(batch["states"][index, :5], batch["states"][ids[0], :5])
        assert batch["states"][index, 12] == batch["states"][ids[0], 12] == 1


def test_ppo_resume_exact_and_evaluation_does_not_mutate_training(tmp_path):
    original = PPOTrainer(tiny())
    original.train_wave()
    rng = copy.deepcopy(original.rng.bit_generator.state)
    evaluate_match(Policy("ppo", original.model, original.device, "ppo"), Policy("random"), 16, 8, 1)
    assert original.rng.bit_generator.state == rng
    original.save(tmp_path / "full.pt")
    expected = original.train_wave()
    resumed = PPOTrainer.load(tmp_path / "full.pt", workers=1)
    actual = resumed.train_wave()
    assert actual["losses"] == expected["losses"]
    assert actual["total_decisions"] == expected["total_decisions"]
    for a, b in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    resumed.save(tmp_path / "policy.pt", export=True)
    policy = checkpoint_policies(tmp_path / "policy.pt")[0]
    assert policy.kind == "ppo"
    with pytest.raises(ValueError, match="cannot resume"):
        PPOTrainer.load(tmp_path / "policy.pt")


def test_adaptive_entropy_direction_and_fixed_ablation():
    trainer = PPOTrainer(tiny(target_entropy=.1))
    before = trainer.entropy_coef
    metrics = trainer.train_wave()
    assert metrics["losses"]["normalized_entropy"] > .1
    assert trainer.entropy_coef < before
    low = PPOTrainer(tiny(target_entropy=1.))
    with torch.no_grad():
        low.model.actor.bias.copy_(torch.arange(28.) * 5)
    before = low.entropy_coef
    low.train_wave()
    assert low.entropy_coef > before
    fixed = PPOTrainer(tiny(adaptive_entropy=False, entropy_coef=0))
    fixed.train_wave()
    assert fixed.entropy_coef == 0


def test_arena_results_invariant_to_batch_and_workers():
    left, right = Policy("random"), Policy("heuristic", kind="heuristic")
    a = evaluate_match(left, right, 64, 2, 1, 832)
    b = evaluate_match(left, right, 64, 32, 4, 832)
    for key in ("wins", "win_rate", "pair_win_rates", "pair_point_differences", "seat_win_rates"):
        assert a[key] == b[key]
    assert sum(a["seat_win_rates"]) / 2 == a["win_rate"]
    bound = pair_summary(np.ones(20), (0, 1))["ci95_hoeffding"]
    assert bound[0] < 1 and bound[1] == 1


def test_heuristic_wins_follow_and_respects_suit_mapping():
    states, masks = np.zeros((1, 465), np.float32), np.zeros((1, 28), bool)
    states[0, 5] = 1  # diamonds trump
    states[0, 445] = 1  # regular lead
    states[0, 448 + 1] = 1  # queen
    states[0, 461 + 1] = 1  # spades (wire suit=2)
    masks[0, [0, 5, 7, 9]] = True  # HJ, SJ, SK, SA
    assert heuristic_probabilities(states, masks).argmax() == 7


def test_dqn_baseline_has_no_sl_and_exports_greedy_q(tmp_path):
    trainer = DQNTrainer(Config(seed=4, num_envs=8, workers=1, device="cpu", torch_threads=1,
                                hidden=(32,), eta=1, batch_size=16, rl_warmup=16, sl_warmup=16,
                                replay_capacity=256, reservoir_capacity=256, learn_every=32))
    trainer.train_wave()
    assert all(a.rl_updates > 0 and a.sl_updates == 0 and len(a.reservoir) == 0 for a in trainer.learners)
    trainer.export(tmp_path / "dqn.pt")
    assert all(p.kind == "dqn" for p in checkpoint_policies(tmp_path / "dqn.pt"))
    trainer.save(tmp_path / "full.pt")
    assert all(p.kind == "dqn" for p in checkpoint_policies(tmp_path / "full.pt"))


def test_performance_freezes_same_trajectories():
    model = PPOTrainer(tiny()).model
    for policy in (Policy("random"), Policy("ppo", model, "cpu", "ppo")):
        a = rollout(policy, 24, 1, 1, 99)
        b = rollout(policy, 24, 8, 4, 99)
        assert a["trajectory_sha256"] == b["trajectory_sha256"]
        assert a["decisions"] == b["decisions"]


@pytest.mark.parametrize("kwargs", [{"entropy_coef": float("nan")}, {"gamma": float("nan")},
                                    {"epochs": 0}, {"entropy_min": 1, "entropy_max": .1}])
def test_invalid_ppo_config(kwargs):
    with pytest.raises(ValueError):
        tiny(**kwargs).validate()
