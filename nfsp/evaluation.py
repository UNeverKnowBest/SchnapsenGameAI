"""Read-only average-policy evaluation, with paired decks and both initial seats."""
from time import perf_counter
import numpy as np
from ._native import BatchEnv
from .learner import action_probabilities, categorical

def evaluate_policy(policy, device, games=1000, num_envs=256, workers=8, seed=1000042,
                    opponent=None, opponent_device=None):
    if games <= 0 or games % 2:
        raise ValueError("evaluation games must be positive and even for paired seats")
    if num_envs < 2:
        num_envs = 2
    width = num_envs - num_envs % 2
    rng = np.random.default_rng(seed)
    wins, point_diff = 0, 0
    pair_scores = []
    started = perf_counter()
    env, previous_count = None, 0
    for first in range(0, games, width):
        count = min(width, games - first)
        if count != previous_count:
            env, previous_count = BatchEnv(count, workers), count
        states, masks, players, winners, points = env.reset(seed, first, True)
        seats = ((first + np.arange(count)) % 2).astype(np.int8)
        while (players >= 0).any():
            masks = masks.astype(np.bool_)
            actions = np.full(count, -1, np.int16)
            for ours in (False, True):
                lanes = np.flatnonzero((players >= 0) & ((players == seats) == ours))
                if not len(lanes):
                    continue
                model = policy if ours else opponent
                if model is None:
                    probs = masks[lanes].astype(np.float32)
                    probs /= probs.sum(axis=1, keepdims=True)
                else:
                    probs = action_probabilities(model, states[lanes], masks[lanes],
                                                 device if ours else (opponent_device or device))
                actions[lanes] = categorical(probs, rng)
            states, masks, players, winners, points = env.step(actions)
        won = winners == seats
        wins += int(won.sum())
        point_diff += int(np.where(won, points, -points.astype(np.int16)).sum())
        pair_scores.extend(won.reshape(-1, 2).mean(axis=1).tolist())
    # Paired observations are correlated; uncertainty uses independent deck-pair means.
    pairs = np.asarray(pair_scores)
    stderr = float(pairs.std(ddof=1) / np.sqrt(len(pairs))) if len(pairs) > 1 else None
    return {"games": games, "wins": wins, "win_rate": wins / games,
            "paired_standard_error": stderr, "mean_game_point_difference": point_diff / games,
            "seconds": perf_counter() - started, "opponent": "random" if opponent is None else "checkpoint",
            "policy": "average", "seed": seed}
