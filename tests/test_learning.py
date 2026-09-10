import copy
import numpy as np
import pytest
import torch
from torch import nn
from nfsp.buffers import ReplayBuffer, ReservoirBuffer
from nfsp.config import Config
from nfsp.learner import categorical
from nfsp.networks import dqn_targets
from nfsp.trainer import Trainer

def sl_batch(count, offset=0):
    states = np.zeros((count, 465), np.float32)
    states[:, 0] = np.arange(offset, offset + count)
    probs = np.zeros((count, 28), np.float32)
    probs[:, 0] = 1
    return {"states": states, "probs": probs, "masks": probs.astype(np.bool_)}

def rl_batch(count, offset=0):
    states = sl_batch(count, offset)["states"]
    return {"states": states, "next_states": states.copy(), "actions": np.zeros(count, np.int64),
            "rewards": np.zeros(count, np.float32), "next_masks": np.ones((count, 28), np.bool_),
            "dones": np.zeros(count, np.bool_)}

def test_ring_overflow_keeps_latest_and_restores_cursor():
    buffer = ReplayBuffer(5)
    buffer.add(rl_batch(3))
    buffer.add(rl_batch(11, 3))
    assert sorted(buffer.data["states"][:, 0]) == list(range(9, 14))
    restored = ReplayBuffer(5)
    restored.load_state_dict(buffer.state_dict())
    for target in (buffer, restored):
        target.add(rl_batch(3, 14))
    np.testing.assert_array_equal(buffer.data["states"], restored.data["states"])
    assert sorted(buffer.data["states"][:, 0]) == list(range(12, 17))

def test_reservoir_batch_matches_algorithm_r_and_resume():
    buffer = ReservoirBuffer(7, 19)
    reference = ReservoirBuffer(7, 19)
    buffer.add(sl_batch(100))
    for i in range(100):
        reference.add(sl_batch(1, i))
    np.testing.assert_array_equal(buffer.data["states"], reference.data["states"])
    restored = ReservoirBuffer(7, 999)
    restored.load_state_dict(buffer.state_dict())
    for target in (buffer, restored):
        target.add(sl_batch(50, 100))
    assert buffer.seen == restored.seen == 150
    np.testing.assert_array_equal(buffer.sample(7)["states"], restored.sample(7)["states"])

def test_reservoir_has_no_recency_bias():
    inclusions = np.zeros(100, np.int64)
    for seed in range(400):
        buffer = ReservoirBuffer(10, seed)
        buffer.add(sl_batch(100))
        inclusions[buffer.data["states"][:, 0].astype(int)] += 1
    # Expected inclusion is 40 per item; broad deterministic statistical regression.
    assert np.max(np.abs(inclusions - 40)) < 28
    assert abs(inclusions[:50].mean() - inclusions[50:].mean()) < 5

class FixedValues(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))
    def forward(self, states):
        return self.values.expand(len(states), -1)

def test_double_dqn_masks_illegal_actions_and_terminal_rows():
    online = FixedValues([1000, 5, 4])
    target = FixedValues([1000, 2, 8])
    rewards = torch.tensor([0., -1.])
    states = torch.zeros((2, 1))
    masks = torch.tensor([[False, True, True], [False, False, False]])
    dones = torch.tensor([False, True])
    actual = dqn_targets(online, target, rewards, states, masks, dones, 1., True)
    torch.testing.assert_close(actual, torch.tensor([2., -1.]))
    vanilla = dqn_targets(online, target, rewards, states, masks, dones, 1., False)
    torch.testing.assert_close(vanilla, torch.tensor([8., -1.]))

def test_categorical_never_samples_zero_mass_actions():
    probs = np.zeros((10000, 28), np.float32)
    probs[:, 2] = .1
    probs[:, 5] = .9
    actions = categorical(probs, np.random.default_rng(8))
    assert set(actions) == {2, 5}

def tiny_config(**kwargs):
    values = dict(seed=123, device="cpu", num_envs=8, workers=2, hidden=(32,),
                  replay_capacity=512, reservoir_capacity=512, batch_size=16,
                  rl_warmup=16, sl_warmup=16, learn_every=32, target_every=2, eta=.5,
                  deterministic=True)
    values.update(kwargs)
    return Config(**values)

@pytest.mark.parametrize("eta", [0., 1.])
def test_collector_modes_rewards_and_every_action_recorded(eta):
    trainer = Trainer(tiny_config(eta=eta))
    rl, sl, metrics = trainer.collector.collect(trainer.learners, 0)
    assert sum(len(b["actions"]) for b in rl) == metrics["decisions"]
    assert sum(float(b["rewards"].sum()) for b in rl) == 0.
    for r, s in zip(rl, sl):
        assert r["dones"].sum() == 8
        assert not r["next_masks"][r["dones"]].any()
        assert r["next_masks"][~r["dones"]].any(axis=1).all()
        assert len(s["states"]) == (len(r["states"]) if eta else 0)
        if eta:
            np.testing.assert_allclose(s["probs"].sum(axis=1), 1., atol=1e-6)
            assert not s["probs"][~s["masks"]].any()
            assert ((s["probs"] > 0).sum(axis=1) > 1).any()
    # Collecting does not mutate the learner's reservoirs or counters.
    assert all(agent.decisions == 0 and len(agent.replay) == 0 for agent in trainer.learners)

def test_evaluation_isolated_and_full_resume_exact(tmp_path):
    original = Trainer(tiny_config())
    for _ in range(3):
        original.train_wave()
    before = copy.deepcopy(original.collector.state_dict())
    counters = [(a.decisions, a.replay.seen, a.reservoir.seen) for a in original.learners]
    original.evaluate(16)
    assert original.collector.state_dict() == before
    assert counters == [(a.decisions, a.replay.seen, a.reservoir.seen) for a in original.learners]
    original.save(tmp_path / "resume.pt")
    expected = original.train_wave()
    restored = Trainer.load(tmp_path / "resume.pt", workers=1)
    actual = restored.train_wave()
    assert actual["total_decisions"] == expected["total_decisions"]
    assert actual["losses"] == expected["losses"]
    for a, b in zip(original.learners, restored.learners):
        for name in ("q", "target", "policy"):
            for p, q in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        np.testing.assert_array_equal(a.reservoir.data["states"][:len(a.reservoir)],
                                      b.reservoir.data["states"][:len(b.reservoir)])
        assert a.rl_updates > 0 and a.sl_updates > 0

def test_old_checkpoint_rejected(tmp_path):
    torch.save({"epoch": 100}, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="incompatible"):
        Trainer.load(tmp_path / "legacy.pt")

def test_importing_main_has_no_training_side_effects():
    import main
    assert callable(main.main)

def test_exchange_transition_stays_with_the_same_player():
    trainer = Trainer(tiny_config(num_envs=64, eta=1., epsilon_start=0., epsilon_end=0.))
    for learner in trainer.learners:
        learner.q = FixedValues([0.] * 24 + [10.] * 4)
    rl, _, _ = trainer.collector.collect(trainer.learners, 0)
    exchanges = 0
    wire_suits = [0, 2, 1, 3]
    for batch in rl:
        for index in np.flatnonzero(batch["actions"] >= 24):
            exchanges += 1
            before, after = batch["states"][index], batch["next_states"][index]
            assert not batch["dones"][index]
            assert before[12] == after[12] == 1  # Same acting player is still leader.
            np.testing.assert_array_equal(before[:5], after[:5])  # Scores/talon unchanged.
            suit = wire_suits[batch["actions"][index] - 24]
            card_offset = 13 + (suit * 5) * 6
            assert before[card_offset + 5] == 1  # Jack in own hand.
            assert after[card_offset + 1] == 1   # Jack now the public trump card.
            assert after[card_offset + 5] == 0
    assert exchanges > 0
