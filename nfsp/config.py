from dataclasses import asdict, dataclass
import math

@dataclass
class Config:
    seed: int = 42
    num_envs: int = 256
    workers: int = 8
    device: str = "auto"
    torch_threads: int = 4
    hidden: tuple[int, ...] = (256, 128, 64)
    eta: float = 0.1
    gamma: float = 1.0
    epsilon_start: float = 0.2
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1_000_000
    replay_capacity: int = 100_000
    reservoir_capacity: int = 200_000
    batch_size: int = 256
    rl_warmup: int = 2_000
    sl_warmup: int = 1_000
    learn_every: int = 128
    target_every: int = 1_000
    rl_lr: float = 0.0003
    sl_lr: float = 0.0003
    grad_clip: float = 5.0
    double_dqn: bool = True
    soft_sl_targets: bool = True
    deterministic: bool = False

    def validate(self):
        for name in ("num_envs", "workers", "torch_threads", "replay_capacity",
                     "reservoir_capacity", "batch_size", "rl_warmup", "sl_warmup",
                     "learn_every", "target_every", "epsilon_decay_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("eta", "gamma", "epsilon_start", "epsilon_end"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        if not self.hidden or any(not isinstance(n, int) or n <= 0 for n in self.hidden):
            raise ValueError("hidden must contain positive widths")
        for name in ("rl_lr", "sl_lr", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.replay_capacity < max(self.batch_size, self.rl_warmup):
            raise ValueError("replay_capacity smaller than batch/warmup")
        if self.reservoir_capacity < max(self.batch_size, self.sl_warmup):
            raise ValueError("reservoir_capacity smaller than batch/warmup")
        return self

    def epsilon(self, decisions: int) -> float:
        fraction = min(decisions / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def to_dict(self):
        return asdict(self)
