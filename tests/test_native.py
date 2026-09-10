import numpy as np
import pytest
from nfsp._native import BatchEnv, STATE_DIM, FEATURE_VERSION

def choose(masks, rng):
    result = np.full(len(masks), -1, np.int16)
    for i, mask in enumerate(masks):
        legal = np.flatnonzero(mask)
        if len(legal):
            result[i] = rng.choice(legal)
    return result

def test_workers_preserve_complete_trajectories():
    a, b = BatchEnv(64, 1), BatchEnv(64, 4)
    x, y = a.reset(42, 100), b.reset(42, 100)
    rng = np.random.default_rng(123)
    steps = 0
    while True:
        for left, right in zip(x, y):
            np.testing.assert_array_equal(left, right)
        if (x[2] < 0).all():
            break
        actions = choose(x[1], rng)
        x, y = a.step(actions), b.step(actions)
        steps += 1
        assert steps < 40
    assert (x[4] >= 1).all() and (x[4] <= 3).all()
    assert not x[1].any()

def test_rejected_batch_does_not_advance_any_lane():
    a, b = BatchEnv(8, 4), BatchEnv(8, 1)
    x = a.reset(7, 0)
    b.reset(7, 0)
    actions = choose(x[1], np.random.default_rng(1))
    bad = actions.copy()
    bad[-1] = 28
    with pytest.raises(ValueError):
        a.step(bad)
    for left, right in zip(a.step(actions), b.step(actions)):
        np.testing.assert_array_equal(left, right)

def test_paired_decks_and_owned_outputs():
    env = BatchEnv(4, 2)
    before = env.reset(9, 0, True)
    saved = before[0].copy()
    np.testing.assert_array_equal(before[0][0], before[0][1])
    np.testing.assert_array_equal(before[0][2], before[0][3])
    env.step(choose(before[1], np.random.default_rng(0)))
    np.testing.assert_array_equal(saved, before[0])
    assert before[0].shape == (4, STATE_DIM)
    assert FEATURE_VERSION == 2

@pytest.mark.parametrize("size,workers", [(0, 1), (1, 0)])
def test_invalid_configuration(size, workers):
    with pytest.raises(ValueError):
        BatchEnv(size, workers)
