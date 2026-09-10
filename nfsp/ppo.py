"""On-policy, shared-policy self-play with entropy normalized over legal actions.

Each trajectory follows ONE player's decisions (including consecutive exchanges).
Complete waves keep behavior weights fixed and avoid cross-player value bootstrap.
"""
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from ._native import ACTION_DIM, FEATURE_VERSION, STATE_DIM, BatchEnv
from .learner import categorical
from .networks import masked_logits
from .trainer import atomic_save, resolve_device


@dataclass
class PPOConfig:
    seed: int = 42
    num_envs: int = 256
    workers: int = 1
    device: str = "auto"
    torch_threads: int = 4
    hidden: tuple[int, ...] = (256, 128, 64)
    lr: float = 0.0003
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    grad_clip: float = 0.5
    epochs: int = 4
    minibatch_size: int = 512
    target_kl: float = 0.02
    entropy_coef: float = 0.05
    adaptive_entropy: bool = True
    target_entropy: float = 0.8
    entropy_lr: float = 0.05
    entropy_min: float = 0.001
    entropy_max: float = 0.5
    reward: str = "win"
    deterministic: bool = False
    rollout_device: str = "same"
    adaptive_lr: bool = False
    lr_min: float = 0.00001
    lr_max: float = 0.001

    def validate(self):
        for key in ("num_envs", "workers", "torch_threads", "epochs", "minibatch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if not self.hidden or any(type(n) is not int or n <= 0 for n in self.hidden):
            raise ValueError("hidden must contain positive widths")
        for key in ("gamma", "gae_lambda", "target_entropy"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        for key in ("lr", "grad_clip", "clip", "target_kl", "entropy_lr", "entropy_min", "entropy_max"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.clip >= 1 or self.entropy_min > self.entropy_max:
            raise ValueError("invalid clip or entropy bounds")
        for key in ("value_coef", "entropy_coef"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.adaptive_entropy and not self.entropy_min <= self.entropy_coef <= self.entropy_max:
            raise ValueError("adaptive entropy coefficient must start inside its bounds")
        if self.reward not in ("win", "game_points"):
            raise ValueError("reward must be win or game_points")
        if self.rollout_device not in ("same", "cpu"):
            raise ValueError("rollout_device must be same or cpu")
        if not (math.isfinite(self.lr_min) and math.isfinite(self.lr_max)
                and 0 < self.lr_min <= self.lr_max):
            raise ValueError("invalid learning rate bounds")
        if self.adaptive_lr and not self.lr_min <= self.lr <= self.lr_max:
            raise ValueError("adaptive learning rate must start inside bounds")
        return self

    def to_dict(self):
        return asdict(self)


class ActorCritic(nn.Module):
    def __init__(self, hidden=(256, 128, 64)):
        super().__init__()
        layers, width = [], STATE_DIM
        for next_width in hidden:
            layer = nn.Linear(width, next_width)
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            width = next_width
        self.body = nn.Sequential(*layers)
        self.actor, self.critic = nn.Linear(width, ACTION_DIM), nn.Linear(width, 1)
        for layer, gain in ((self.actor, .01), (self.critic, 1.)):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)

    def forward(self, states):
        return self.actor(self.body(states))

    def both(self, states):
        hidden = self.body(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def legal_distribution(logits, masks, validate=True):
    if validate and not masks.any(dim=-1).all():
        raise ValueError("cannot act without a legal action")
    return torch.distributions.Categorical(logits=masked_logits(logits, masks), validate_args=False)


def normalized_entropy(distribution, masks):
    counts = masks.sum(dim=-1)
    # Forced actions have zero entropy and MUST be excluded from the controller.
    # Mask before multiplication: Categorical.entropy can overflow its clamped
    # illegal-logit gradient for a uniform forced-action row in float32.
    safe_logs = distribution.logits.masked_fill(~masks, 0.)
    entropy = -(distribution.probs * safe_logs).sum(dim=-1)
    return entropy / counts.float().clamp_min(2).log(), counts > 1


def gae(rewards, values, gamma, lam):
    """One complete same-player trajectory; terminal value is exactly zero."""
    advantages = np.empty(len(rewards), np.float32)
    carry, next_value = 0., 0.
    for i in range(len(rewards) - 1, -1, -1):
        carry = rewards[i] + gamma * next_value - values[i] + gamma * lam * carry
        advantages[i], next_value = carry, values[i]
    return advantages, advantages + np.asarray(values, np.float32)


class PPOTrainer:
    def __init__(self, config):
        self.config = config.validate()
        if config.deterministic:
            import os
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.device = resolve_device(config.device)
        torch.set_num_threads(config.torch_threads)
        torch.use_deterministic_algorithms(config.deterministic)
        torch.manual_seed(config.seed)
        self.model = ActorCritic(config.hidden).to(self.device)
        self.rollout_model = (copy.deepcopy(self.model).cpu().requires_grad_(False)
                              if config.rollout_device == "cpu" and self.device.type != "cpu" else self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.rng = np.random.default_rng(config.seed + 400)
        self.entropy_coef = config.entropy_coef
        self.games = self.decisions = self.waves = self.updates = 0
        self.training_seconds = 0.
        self.env, self.width = None, 0

    @torch.no_grad()
    def collect(self, count):
        if count <= 0:
            raise ValueError("count must be positive")
        c, started = self.config, perf_counter()
        if self.rollout_model is not self.model:
            self.rollout_model.load_state_dict(self.model.state_dict())
        actor_device = next(self.rollout_model.parameters()).device
        if count != self.width:
            self.env, self.width = BatchEnv(count, c.workers), count
        t = perf_counter()
        states, masks, players, winners, points = self.env.reset(c.seed, self.games)
        env_seconds, inference_seconds = perf_counter() - t, 0.
        trajectories = [[[] for _ in range(2)] for _ in range(count)]
        chunks, offset = [], 0
        while (players >= 0).any():
            lanes = np.flatnonzero(players >= 0)
            legal = masks[lanes].astype(np.bool_)
            t = perf_counter()
            logits, values = self.rollout_model.both(torch.from_numpy(states[lanes]).to(actor_device))
            dist = legal_distribution(logits, torch.from_numpy(legal).to(actor_device), validate=False)
            # Transfer probabilities, log probabilities and values together; sample on CPU.
            packed = torch.cat((dist.probs, dist.logits, values[:, None]), dim=1).cpu().numpy()
            chosen = categorical(packed[:, :ACTION_DIM], self.rng)
            logs = packed[np.arange(len(lanes)), ACTION_DIM + chosen].copy()
            values = packed[:, -1].copy()
            inference_seconds += perf_counter() - t
            chunks.append(dict(states=states[lanes].copy(), masks=legal, actions=chosen,
                               old_log_probs=logs, values=values))
            for j, lane in enumerate(lanes):
                trajectories[lane][players[lane]].append(offset + j)
            offset += len(lanes)
            actions = np.full(count, -1, np.int16)
            actions[lanes] = chosen
            t = perf_counter()
            states, masks, players, winners, points = self.env.step(actions)
            env_seconds += perf_counter() - t
        batch = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        batch["advantages"] = np.empty(offset, np.float32)
        batch["returns"] = np.empty(offset, np.float32)
        batch["rewards"] = np.zeros(offset, np.float32)
        batch["trajectory_ids"] = np.empty(offset, np.int64)
        for lane, pair in enumerate(trajectories):
            for player, indices in enumerate(pair):
                if not indices:
                    continue
                reward = (1. if winners[lane] == player else -1.)
                if c.reward == "game_points":
                    reward *= float(points[lane]) / 3.
                batch["rewards"][indices[-1]] = reward
                batch["trajectory_ids"][indices] = lane * 2 + player
                adv, returns = gae(batch["rewards"][indices], batch["values"][indices], c.gamma, c.gae_lambda)
                batch["advantages"][indices], batch["returns"][indices] = adv, returns
        elapsed = perf_counter() - started
        return batch, dict(games=count, decisions=offset, collect_seconds=elapsed,
                           environment_seconds=env_seconds, inference_seconds=inference_seconds,
                           collection_overhead_seconds=max(0., elapsed - env_seconds - inference_seconds))

    def update(self, batch):
        c = self.config
        keys = ("states", "masks", "actions", "old_log_probs", "advantages", "returns")
        data = {k: torch.from_numpy(batch[k]).to(self.device) for k in keys}
        used_lr = self.optimizer.param_groups[0]["lr"]
        advantages = data["advantages"]
        data["advantages"] = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        n, records, stopped = len(advantages), [], False
        for _ in range(c.epochs):
            permutation = self.rng.permutation(n)
            for start in range(0, n, c.minibatch_size):
                idx = torch.from_numpy(permutation[start:start + c.minibatch_size]).to(self.device)
                logits, values = self.model.both(data["states"][idx])
                masks = data["masks"][idx]
                dist = legal_distribution(logits, masks, validate=False)
                log_ratio = dist.log_prob(data["actions"][idx]) - data["old_log_probs"][idx]
                ratio = log_ratio.exp()
                kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(kl):
                    raise FloatingPointError("nonfinite PPO KL")
                if kl.item() > c.target_kl:
                    stopped = True
                    break
                adv = data["advantages"][idx]
                policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv).mean()
                value_loss = .5 * (values - data["returns"][idx]).square().mean()
                ent, choices = normalized_entropy(dist, masks)
                entropy = (ent * choices).sum() / choices.sum().clamp_min(1)
                loss = policy_loss + c.value_coef * value_loss - self.entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip, error_if_nonfinite=True)
                self.optimizer.step()
                self.updates += 1
                records.append(torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach(), kl.detach(),
                                            ((ratio - 1).abs() > c.clip).float().mean().detach(), norm.detach())))
            if stopped:
                break
        # Controller measures the final policy, once per wave, on choice states only.
        entropy_sum, choice_count, final_values, kl_sum = 0., 0, [], 0.
        with torch.no_grad():
            for start in range(0, n, c.minibatch_size):
                x, masks = data["states"][start:start+c.minibatch_size], data["masks"][start:start+c.minibatch_size]
                logits, values = self.model.both(x)
                dist = legal_distribution(logits, masks, validate=False)
                ent, choices = normalized_entropy(dist, masks)
                entropy_sum = entropy_sum + (ent * choices).sum()
                choice_count = choice_count + choices.sum()
                log_ratio = dist.log_prob(data["actions"][start:start+c.minibatch_size]) - data["old_log_probs"][start:start+c.minibatch_size]
                kl_sum = kl_sum + (log_ratio.exp() - 1 - log_ratio).sum()
                final_values.append(values)
            values = torch.cat(final_values)
            variance = data["returns"].var(unbiased=False).item()
            explained = 1 - (data["returns"] - values).var(unbiased=False).item() / variance if variance > 1e-8 else None
        choice_count = int(choice_count.item())
        measured = entropy_sum.item() / choice_count if choice_count else None
        final_kl = kl_sum.item() / n
        if not math.isfinite(final_kl):
            raise FloatingPointError("nonfinite final PPO KL")
        next_lr = used_lr
        if c.adaptive_lr:
            if stopped or final_kl > c.target_kl:
                next_lr = max(c.lr_min, used_lr / 1.5)
            elif final_kl < c.target_kl / 2:
                next_lr = min(c.lr_max, used_lr * 1.1)
            for group in self.optimizer.param_groups:
                group["lr"] = next_lr
        used_coef = self.entropy_coef
        if c.adaptive_entropy and measured is not None:
            self.entropy_coef = float(np.clip(self.entropy_coef * math.exp(c.entropy_lr * (c.target_entropy - measured)),
                                              c.entropy_min, c.entropy_max))
        names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
        metrics = dict(zip(names, torch.stack(records).mean(0).cpu().tolist())) if records else dict.fromkeys(names)
        return {**metrics, "normalized_entropy": measured, "entropy_coef_used": used_coef,
                "entropy_coef_next": self.entropy_coef, "explained_variance": explained,
                "optimizer_steps": len(records), "early_stop_kl": stopped,
                "final_kl": final_kl, "lr_used": used_lr, "lr_next": next_lr}

    def train_wave(self, count=None):
        started = perf_counter()
        batch, metrics = self.collect(self.config.num_envs if count is None else count)
        t = perf_counter()
        metrics["losses"] = self.update(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics["update_seconds"] = perf_counter() - t
        elapsed = perf_counter() - started
        self.games += metrics["games"]
        self.decisions += metrics["decisions"]
        self.waves += 1
        self.training_seconds += elapsed
        metrics.update(total_games=self.games, total_decisions=self.decisions, wave=self.waves,
                       seconds=elapsed, games_per_second=metrics["games"] / elapsed,
                       decisions_per_second=metrics["decisions"] / elapsed)
        return metrics

    def save(self, path, export=False):
        state = dict(algorithm="ppo", version=1, feature_version=FEATURE_VERSION,
                     state_dim=STATE_DIM, action_dim=ACTION_DIM, config=self.config.to_dict(),
                     model=self.model.state_dict(), games=self.games, decisions=self.decisions,
                     waves=self.waves, updates=self.updates, training_seconds=self.training_seconds,
                     entropy_coef=self.entropy_coef)
        if not export:
            state.update(optimizer=self.optimizer.state_dict(), rng=copy.deepcopy(self.rng.bit_generator.state),
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        atomic_save(state, path)

    @classmethod
    def load(cls, path, device=None, workers=None):
        state = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(state)
        if "optimizer" not in state:
            raise ValueError("policy export cannot resume training; use latest.pt")
        config = PPOConfig(**state["config"])
        if device is not None:
            config.device = device
        if workers is not None:
            config.workers = workers
        result = cls(config)
        result.model.load_state_dict(state["model"])
        result.optimizer.load_state_dict(state["optimizer"])
        for key in ("games", "decisions", "waves", "updates", "training_seconds", "entropy_coef"):
            setattr(result, key, state[key])
        result.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return result

    def run(self, games, output, save_every=10000, eval_every=10000, eval_games=1000):
        from .evaluation import evaluate_policy
        if games < self.games or save_every < 0 or eval_every < 0:
            raise ValueError("invalid absolute target or interval")
        if eval_every and (eval_games <= 0 or eval_games % 2):
            raise ValueError("evaluation games must be positive and even")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            while self.games < games:
                previous = self.games
                metrics = self.train_wave(min(self.config.num_envs, games - self.games))
                log.write(json.dumps({"kind": "train", **metrics}) + "\n")
                if eval_every and self.games // eval_every > previous // eval_every:
                    result = evaluate_policy(self.model, self.device, eval_games, self.config.num_envs,
                                             self.config.workers, self.config.seed + 1_000_000)
                    result["policy"] = "ppo"
                    log.write(json.dumps({"kind": "evaluation", "games": self.games, **result}) + "\n")
                log.flush()
                if save_every and self.games // save_every > previous // save_every:
                    self.save(output / "latest.pt")
                    self.save(output / f"policy_{self.games}.pt", export=True)
                if self.waves == 1 or self.waves % 10 == 0 or self.games == games:
                    print(json.dumps({"games": self.games, "losses": metrics["losses"]}), flush=True)
        self.save(output / "latest.pt")
        self.save(output / "policy.pt", export=True)
        return dict(games=self.games, decisions=self.decisions, training_seconds=self.training_seconds)


def validate_checkpoint(state):
    expected = dict(algorithm="ppo", version=1, feature_version=FEATURE_VERSION,
                    state_dim=STATE_DIM, action_dim=ACTION_DIM)
    if any(state.get(k) != v for k, v in expected.items()):
        raise ValueError("incompatible PPO checkpoint feature/version/dimensions")
