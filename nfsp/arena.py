"""Common, paired-seat evaluation for every algorithm and public-information bot."""
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ._native import ACTION_DIM, FEATURE_VERSION, STATE_DIM, BatchEnv
from .learner import action_probabilities
from .networks import Network
from .ppo import ActorCritic, validate_checkpoint


class Policy:
    def __init__(self, name, model=None, device="cpu", kind="random", metadata=None):
        self.name, self.model, self.device, self.kind = name, model, device, kind
        self.metadata = metadata or {}

    def probabilities(self, states, masks):
        if self.model is not None:
            return action_probabilities(self.model, states, masks, self.device,
                                        epsilon=0. if self.kind == "dqn" else None)
        if self.kind == "random":
            probs = masks.astype(np.float32)
            return probs / probs.sum(axis=1, keepdims=True)
        return heuristic_probabilities(states, masks)


def heuristic_probabilities(states, masks):
    """Greedy public bot: exchange/marry, cheapest winning follow, cheap lead.

    It never inspects hidden cards or searches privileged engine states.
    Native rank order J,Q,K,10,A; action suit order H,S,C,D.
    """
    count = len(states)
    costs = np.array([2, 3, 4, 10, 11] * 4, np.float32)
    scores = np.tile(-costs, (count, 1))
    wire_suits = np.array([0, 2, 1, 3])
    suits = np.repeat(wire_suits, 5)
    trump = 3 - states[:, 5:9].argmax(axis=1)
    scores -= (suits[None, :] == trump[:, None]) * 4
    lead = states[:, 445:465]
    following = lead[:, :3].sum(axis=1) > 0
    lead_suit = 3 - lead[:, 16:20].argmax(axis=1)
    old_ranks = lead[:, 3:16].argmax(axis=1)
    ranks = np.zeros(count, int)
    for old, new in ((0, 2), (1, 1), (2, 0), (3, 3), (12, 4)):
        ranks[old_ranks == old] = new
    same = suits[None, :] == lead_suit[:, None]
    beats = (same & (np.tile(np.arange(5), 4)[None, :] > ranks[:, None])) | (
        (suits[None, :] == trump[:, None]) & (lead_suit != trump)[:, None])
    scores += (following[:, None] & beats) * 50
    all_scores = np.full((count, ACTION_DIM), -np.inf, np.float32)
    all_scores[:, :20] = scores
    all_scores[:, 20:24] = 70 + (wire_suits[None, :] == trump[:, None]) * 20
    all_scores[:, 24:28] = 100
    all_scores[~masks] = -np.inf
    probs = np.zeros_like(all_scores)
    probs[np.arange(count), all_scores.argmax(axis=1)] = 1.
    return probs


def checkpoint_policies(path, device="cpu", name=None):
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"path": str(path.resolve()), "sha256": digest, "training_games": state.get("games"),
            "config": state.get("config")}
    name = name or path.stem
    if state.get("algorithm") == "ppo":
        validate_checkpoint(state)
        model = ActorCritic(state["config"]["hidden"]).to(device).eval()
        model.load_state_dict(state["model"])
        return [Policy(name, model, device, "ppo", meta)]
    if state.get("feature_version") != FEATURE_VERSION or state.get("version") not in (1, 2):
        raise ValueError("incompatible checkpoint feature/version")
    kind = "dqn" if state.get("algorithm") == "dqn" else "nfsp"
    if kind == "nfsp" and state.get("version") != 2:
        raise ValueError("incompatible NFSP checkpoint")
    weights = state.get("policies")
    if weights is None:
        weights = [agent["q" if kind == "dqn" else "policy"] for agent in state["learners"]]
    result = []
    for i, weight in enumerate(weights):
        model = Network(hidden=state["config"]["hidden"]).to(device).eval()
        model.load_state_dict(weight)
        result.append(Policy(f"{name}/p{i}", model, device, kind, meta))
    return result


def pair_summary(values, bounds, rng=None, bootstrap=2000):
    values = np.asarray(values, np.float64)
    n = len(values)
    if not n:
        raise ValueError("empty pair sample")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    # A distribution-free bound remains meaningful even with 0/100% observed wins.
    radius = (bounds[1] - bounds[0]) * np.sqrt(np.log(40) / (2 * n))
    result = {"mean": mean, "standard_error": se,
              "ci95_hoeffding": [max(bounds[0], mean-radius), min(bounds[1], mean+radius)]}
    if rng is not None and bootstrap:
        means = []
        for start in range(0, bootstrap, 100):
            indices = rng.integers(n, size=(min(100, bootstrap-start), n))
            means.extend(values[indices].mean(axis=1))
        result["ci95_bootstrap"] = np.quantile(means, [.025, .975]).tolist()
    return result


def counter_uniform(seed, game_ids, steps):
    """Stateless game-local RNG: scheduling, worker count and batch width invariant."""
    with np.errstate(over="ignore"):
        x = np.asarray(game_ids, np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        x = x + np.asarray(steps, np.uint64) * np.uint64(0xD1B54A32D192ED03) + np.uint64(seed)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / 2**53


def evaluate_match(policy, opponent, games=2000, num_envs=256, workers=1, seed=1000042):
    if games <= 0 or games % 2 or num_envs <= 0 or workers <= 0 or not 0 <= seed < 2**63:
        raise ValueError("positive even games, positive envs/workers and valid seed required")
    width = max(2, num_envs - num_envs % 2)
    outcomes, awards, seat_wins = [], [], [0, 0]
    entropy_sum, choice_count, decisions = 0., 0, 0
    started = perf_counter()
    env, previous = None, 0
    for first in range(0, games, width):
        count = min(width, games-first)
        if count != previous:
            env, previous = BatchEnv(count, workers), count
        states, masks, players, winners, points = env.reset(seed, first, True)
        ids = first + np.arange(count)
        seats = (ids % 2).astype(np.int8)
        steps = np.zeros(count, np.int64)
        while (players >= 0).any():
            actions = np.full(count, -1, np.int16)
            active = players >= 0
            for ours, actor in ((True, policy), (False, opponent)):
                lanes = np.flatnonzero(active & ((players == seats) == ours))
                if not len(lanes):
                    continue
                legal = masks[lanes].astype(np.bool_)
                probs = actor.probabilities(states[lanes], legal)
                if (not np.isfinite(probs).all() or (probs < 0).any()
                        or probs[~legal].any() or not np.allclose(probs.sum(1), 1, atol=1e-5)):
                    raise ValueError("policy returned invalid legal probabilities")
                cumulative = probs.cumsum(axis=1, dtype=np.float64)
                draws = counter_uniform(seed + 17, ids[lanes], steps[lanes]) * cumulative[:, -1]
                draws = np.minimum(draws, np.nextafter(cumulative[:, -1], 0))
                actions[lanes] = (draws[:, None] >= cumulative).sum(axis=1)
                if ours:
                    choices = legal.sum(1) > 1
                    ent = -(probs * np.log(np.maximum(probs, 1e-30))).sum(1)
                    entropy_sum += float((ent[choices] / np.log(legal.sum(1)[choices])).sum())
                    choice_count += int(choices.sum())
                    decisions += len(lanes)
            steps[active] += 1
            states, masks, players, winners, points = env.step(actions)
        won = winners == seats
        outcomes.extend(won.astype(float))
        awards.extend(np.where(won, points.astype(int), -points.astype(int)))
        for seat in range(2):
            seat_wins[seat] += int(won[seats == seat].sum())
    win_pairs = np.asarray(outcomes).reshape(-1, 2).mean(axis=1)
    point_pairs = np.asarray(awards).reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(seed + 100)
    return {"policy": policy.name, "opponent": opponent.name, "games": games, "seed": seed,
            "wins": int(sum(outcomes)), "win_rate": float(win_pairs.mean()),
            "win_uncertainty": pair_summary(win_pairs, (0, 1), rng),
            "mean_game_point_difference": float(point_pairs.mean()),
            "point_uncertainty": pair_summary(point_pairs, (-3, 3), rng),
            "seat_win_rates": [w / (games / 2) for w in seat_wins],
            "normalized_entropy": entropy_sum / choice_count if choice_count else None,
            "decisions": decisions, "seconds": perf_counter() - started,
            "pair_win_rates": win_pairs.tolist(), "pair_point_differences": point_pairs.tolist()}
