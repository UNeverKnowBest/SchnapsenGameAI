"""Synchronous waves keep model weights fixed while complete games are collected."""
import copy
from time import perf_counter
import numpy as np
from ._native import BatchEnv, STATE_DIM, ACTION_DIM
from .buffers import RL_SCHEMA, SL_SCHEMA
from .learner import action_probabilities, categorical

def empty_batch(schema):
    return {key: np.empty((0, *shape), dtype=dtype) for key, (shape, dtype) in schema.items()}

def concatenate(parts, schema):
    if not parts:
        return empty_batch(schema)
    return {key: np.concatenate([part[key] for part in parts]) for key in schema}

class Collector:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed + 300)
        self.env = None
        self.num_envs = 0

    def collect(self, learners, first_game, count=None):
        c = self.config
        count = c.num_envs if count is None else count
        if count <= 0:
            raise ValueError("count must be positive")
        started = perf_counter()
        if count != self.num_envs:
            self.env = BatchEnv(count, c.workers)
            self.num_envs = count
        t = perf_counter()
        states, masks, players, winners, points = self.env.reset(c.seed, first_game)
        environment_seconds = perf_counter() - t
        inference_seconds = 0.
        masks = masks.astype(np.bool_)
        # Players have stable identities; network identities alternate initial seats.
        swaps = ((first_game + np.arange(count)) % 2).astype(np.int8)
        modes = self.rng.random((2, count)) < c.eta
        epsilons = [c.epsilon(agent.decisions) for agent in learners]
        pending = np.zeros((2, count), dtype=np.bool_)
        previous_states = np.zeros((2, count, STATE_DIM), dtype=np.float32)
        previous_actions = np.zeros((2, count), dtype=np.int64)
        rl_parts, sl_parts = [[], []], [[], []]
        decisions = 0

        while (players >= 0).any():
            actions = np.full(count, -1, dtype=np.int16)
            actor = players ^ swaps
            for agent_id, learner in enumerate(learners):
                lanes = np.flatnonzero((players >= 0) & (actor == agent_id))
                if not len(lanes):
                    continue
                prior = lanes[pending[agent_id, lanes]]
                if len(prior):
                    rl_parts[agent_id].append({
                        "states": previous_states[agent_id, prior].copy(),
                        "actions": previous_actions[agent_id, prior].copy(),
                        "rewards": np.zeros(len(prior), np.float32),
                        "next_states": states[prior].copy(),
                        "next_masks": masks[prior].copy(),
                        "dones": np.zeros(len(prior), np.bool_),
                    })
                for best_response in (False, True):
                    group = lanes[modes[agent_id, lanes] == best_response]
                    if not len(group):
                        continue
                    t = perf_counter()
                    probs = action_probabilities(
                        learner.q if best_response else learner.policy, states[group], masks[group],
                        learner.device, epsilons[agent_id] if best_response else None)
                    selected = categorical(probs, self.rng)
                    inference_seconds += perf_counter() - t
                    actions[group] = selected
                    if best_response:
                        targets = probs
                        if not c.soft_sl_targets:
                            targets = np.zeros_like(probs)
                            targets[np.arange(len(group)), selected] = 1.
                        sl_parts[agent_id].append({
                            "states": states[group].copy(), "probs": targets.copy(), "masks": masks[group].copy(),
                        })
                previous_states[agent_id, lanes] = states[lanes]
                previous_actions[agent_id, lanes] = actions[lanes]
                pending[agent_id, lanes] = True

            decisions += int((players >= 0).sum())
            active = players >= 0
            t = perf_counter()
            states, masks, players, winners, points = self.env.step(actions)
            environment_seconds += perf_counter() - t
            masks = masks.astype(np.bool_)
            finished = active & (players < 0)
            for agent_id in range(2):
                lanes = np.flatnonzero(finished & pending[agent_id])
                if len(lanes):
                    rl_parts[agent_id].append({
                        "states": previous_states[agent_id, lanes].copy(),
                        "actions": previous_actions[agent_id, lanes].copy(),
                        "rewards": np.where((winners[lanes] ^ swaps[lanes]) == agent_id, 1., -1.).astype(np.float32),
                        "next_states": np.zeros((len(lanes), STATE_DIM), np.float32),
                        "next_masks": np.zeros((len(lanes), ACTION_DIM), np.bool_),
                        "dones": np.ones(len(lanes), np.bool_),
                    })
                    pending[agent_id, lanes] = False
        if pending.any():
            raise RuntimeError("unfinished player transitions at wave boundary")
        rl = [concatenate(parts, RL_SCHEMA) for parts in rl_parts]
        sl = [concatenate(parts, SL_SCHEMA) for parts in sl_parts]
        if sum(len(batch["actions"]) for batch in rl) != decisions:
            raise RuntimeError("each executed action must generate exactly one transition")
        elapsed = perf_counter() - started
        return rl, sl, {
            "games": count, "decisions": decisions, "collect_seconds": elapsed,
            "environment_seconds": environment_seconds, "inference_seconds": inference_seconds,
            "collection_overhead_seconds": max(0., elapsed - environment_seconds - inference_seconds),
            "wins": [int(((winners ^ swaps) == i).sum()) for i in range(2)],
            "game_points": [int(points[(winners ^ swaps) == i].sum()) for i in range(2)],
        }

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
