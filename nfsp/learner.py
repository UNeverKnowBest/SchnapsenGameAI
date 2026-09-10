import copy
import numpy as np
import torch
from torch.nn import functional as F
from .buffers import ReplayBuffer, ReservoirBuffer
from .networks import Network, dqn_targets, masked_logits

class Learner:
    def __init__(self, config, player, device):
        self.config, self.device = config, device
        self.q = Network(hidden=config.hidden).to(device)
        self.target = copy.deepcopy(self.q).requires_grad_(False).eval()
        self.policy = Network(hidden=config.hidden).to(device)
        self.rl_optimizer = torch.optim.Adam(self.q.parameters(), lr=config.rl_lr)
        self.sl_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.sl_lr)
        self.replay = ReplayBuffer(config.replay_capacity, config.seed + 100 + player)
        self.reservoir = ReservoirBuffer(config.reservoir_capacity, config.seed + 200 + player)
        self.decisions = self.rl_updates = self.sl_updates = 0
        self.rl_credit = self.sl_credit = 0

    def _tensors(self, batch):
        return {key: torch.from_numpy(value).to(self.device) for key, value in batch.items()}

    def ingest(self, rl, sl):
        count = len(rl["actions"])
        self.replay.add(rl)
        self.reservoir.add(sl)
        self.decisions += count
        # Warm-up does not accumulate a backlog of gradient updates.
        if len(self.replay) >= max(self.config.rl_warmup, self.config.batch_size):
            self.rl_credit += count
        if len(self.reservoir) >= max(self.config.sl_warmup, self.config.batch_size):
            self.sl_credit += count

    def update(self):
        c = self.config
        losses = {"rl": [], "sl": []}
        while self.rl_credit >= c.learn_every:
            batch = self._tensors(self.replay.sample(c.batch_size))
            predicted = self.q(batch["states"]).gather(1, batch["actions"][:, None]).squeeze(1)
            targets = dqn_targets(self.q, self.target, batch["rewards"], batch["next_states"],
                                  batch["next_masks"], batch["dones"], c.gamma, c.double_dqn)
            loss = F.smooth_l1_loss(predicted, targets)
            self._step(loss, self.rl_optimizer, self.q)
            losses["rl"].append(loss.detach())
            self.rl_credit -= c.learn_every
            self.rl_updates += 1
            if self.rl_updates % c.target_every == 0:
                self.target.load_state_dict(self.q.state_dict())
        while self.sl_credit >= c.learn_every:
            batch = self._tensors(self.reservoir.sample(c.batch_size))
            logits = masked_logits(self.policy(batch["states"]), batch["masks"])
            # Zero out illegal log probabilities before multiplying zero target mass.
            log_probs = F.log_softmax(logits, dim=1).masked_fill(~batch["masks"], 0)
            loss = -(batch["probs"] * log_probs).sum(dim=1).mean()
            self._step(loss, self.sl_optimizer, self.policy)
            losses["sl"].append(loss.detach())
            self.sl_credit -= c.learn_every
            self.sl_updates += 1
        return {key: torch.stack(values).mean().item() if values else None
                for key, values in losses.items()}

    def _step(self, loss, optimizer, model):
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip,
                                       error_if_nonfinite=True)
        optimizer.step()

    def state_dict(self):
        return {
            "q": self.q.state_dict(), "target": self.target.state_dict(),
            "policy": self.policy.state_dict(),
            "rl_optimizer": self.rl_optimizer.state_dict(),
            "sl_optimizer": self.sl_optimizer.state_dict(),
            "replay": self.replay.state_dict(), "reservoir": self.reservoir.state_dict(),
            "decisions": self.decisions, "rl_updates": self.rl_updates, "sl_updates": self.sl_updates,
            "rl_credit": self.rl_credit, "sl_credit": self.sl_credit,
        }

    def load_state_dict(self, state):
        for key in ("q", "target", "policy", "rl_optimizer", "sl_optimizer", "replay", "reservoir"):
            getattr(self, key).load_state_dict(state[key])
        for key in ("decisions", "rl_updates", "sl_updates", "rl_credit", "sl_credit"):
            setattr(self, key, state[key])

@torch.inference_mode()
def action_probabilities(model, states, masks, device, epsilon=None):
    if not len(states):
        return np.empty((0, 28), dtype=np.float32)
    if not masks.any(axis=1).all():
        raise ValueError("cannot act without a legal action")
    x = torch.from_numpy(states).to(device)
    legal = torch.from_numpy(masks).to(device)
    values = masked_logits(model(x), legal)
    if epsilon is None:
        probs = torch.softmax(values, dim=1)
    else:
        probs = legal.to(torch.float32)
        probs *= epsilon / probs.sum(dim=1, keepdim=True)
        probs.scatter_add_(1, values.argmax(dim=1, keepdim=True),
                           torch.full((len(states), 1), 1 - epsilon, device=device))
    return probs.cpu().numpy()

def categorical(probs, rng):
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise FloatingPointError("invalid action probabilities")
    cumulative = np.cumsum(probs, axis=1, dtype=np.float64)
    total = cumulative[:, -1]
    if (total <= 0).any():
        raise ValueError("empty action distribution")
    draws = np.minimum(rng.random(len(probs)) * total, np.nextafter(total, 0.))
    return (draws[:, None] >= cumulative).sum(axis=1).astype(np.int64)
