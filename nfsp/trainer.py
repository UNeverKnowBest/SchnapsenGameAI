import json
import os
import random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ._native import STATE_DIM, ACTION_DIM, FEATURE_VERSION
from .config import Config
from .collector import Collector
from .learner import Learner
from .evaluation import evaluate_policy

CHECKPOINT_VERSION = 2

def resolve_device(request):
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install the CUDA PyTorch wheel")
    return device

def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

class Trainer:
    def __init__(self, config):
        self.config = config.validate()
        if config.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.device = resolve_device(config.device)
        torch.set_num_threads(config.torch_threads)
        torch.use_deterministic_algorithms(config.deterministic)
        random.seed(config.seed)
        np.random.seed(config.seed % 2**32)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.learners = [Learner(config, i, self.device) for i in range(2)]
        self.collector = Collector(config)
        self.games = self.decisions = self.waves = 0
        self.training_seconds = 0.

    def train_wave(self, count=None):
        started = perf_counter()
        rl, sl, metrics = self.collector.collect(self.learners, self.games, count)
        t = perf_counter()
        for i, learner in enumerate(self.learners):
            learner.ingest(rl[i], sl[i])
        metrics["buffer_seconds"] = perf_counter() - t
        t = perf_counter()
        metrics["losses"] = [learner.update() for learner in self.learners]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics["update_seconds"] = perf_counter() - t
        elapsed = perf_counter() - started
        self.games += metrics["games"]
        self.decisions += metrics["decisions"]
        self.waves += 1
        self.training_seconds += elapsed
        metrics.update({
            "total_games": self.games, "total_decisions": self.decisions, "wave": self.waves,
            "seconds": elapsed, "games_per_second": metrics["games"] / elapsed,
            "decisions_per_second": metrics["decisions"] / elapsed,
            "epsilon": [self.config.epsilon(agent.decisions) for agent in self.learners],
            "rl_updates": [agent.rl_updates for agent in self.learners],
            "sl_updates": [agent.sl_updates for agent in self.learners],
            "replay_sizes": [len(agent.replay) for agent in self.learners],
            "reservoir_sizes": [len(agent.reservoir) for agent in self.learners],
            "reservoir_seen": [agent.reservoir.seen for agent in self.learners],
        })
        return metrics

    def evaluate(self, games=1000, seed=1000042):
        return [evaluate_policy(agent.policy, self.device, games, self.config.num_envs,
                                self.config.workers, seed) for agent in self.learners]

    def save(self, path):
        # Called only after a complete wave: no pending games/transitions are discarded.
        state = {
            "version": CHECKPOINT_VERSION, "feature_version": FEATURE_VERSION,
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "config": self.config.to_dict(),
            "games": self.games, "decisions": self.decisions, "waves": self.waves,
            "training_seconds": self.training_seconds,
            "learners": [agent.state_dict() for agent in self.learners],
            "collector": self.collector.state_dict(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        atomic_save(state, path)

    def export(self, path):
        atomic_save({
            "version": CHECKPOINT_VERSION, "feature_version": FEATURE_VERSION,
            "config": self.config.to_dict(), "games": self.games,
            "policies": [agent.policy.state_dict() for agent in self.learners],
        }, path)

    @classmethod
    def load(cls, path, device=None, workers=None):
        # Full training snapshots include NumPy RNG/buffer objects. Load only trusted local files.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION or state.get("feature_version") != FEATURE_VERSION:
            raise ValueError("incompatible checkpoint: legacy weights require an explicit feature migration")
        if state.get("state_dim") != STATE_DIM or state.get("action_dim") != ACTION_DIM:
            raise ValueError("incompatible observation/action dimensions")
        config = Config(**state["config"])
        if device is not None:
            config.device = device
        if workers is not None:
            config.workers = workers
        result = cls(config)
        for agent, saved in zip(result.learners, state["learners"], strict=True):
            agent.load_state_dict(saved)
        result.collector.load_state_dict(state["collector"])
        for name in ("games", "decisions", "waves", "training_seconds"):
            setattr(result, name, state[name])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return result

    def run(self, games, output, save_every=10000, eval_every=10000, eval_games=1000):
        if games < self.games or save_every < 0 or eval_every < 0:
            raise ValueError("games is an absolute target; intervals must be nonnegative")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        next_save = (self.games // save_every + 1) * save_every if save_every else float("inf")
        next_eval = (self.games // eval_every + 1) * eval_every if eval_every else float("inf")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            while self.games < games:
                metrics = self.train_wave(min(self.config.num_envs, games - self.games))
                log.write(json.dumps({"kind": "train", **metrics}) + "\n")
                log.flush()
                if self.waves == 1 or self.waves % 10 == 0 or self.games == games:
                    print(json.dumps({"games": self.games, "games_per_second": round(metrics["games_per_second"], 1),
                                      "losses": metrics["losses"]}), flush=True)
                if self.games >= next_eval:
                    evaluation = self.evaluate(eval_games, self.config.seed + 1_000_000)
                    log.write(json.dumps({"kind": "evaluation", "games": self.games, "players": evaluation}) + "\n")
                    log.flush()
                    print(json.dumps({"evaluation": evaluation}), flush=True)
                    next_eval = (self.games // eval_every + 1) * eval_every
                if self.games >= next_save:
                    self.save(output / "latest.pt")
                    self.export(output / f"policy_{self.games}.pt")
                    next_save = (self.games // save_every + 1) * save_every
        self.save(output / "latest.pt")
        self.export(output / "policy.pt")
        return {"games": self.games, "decisions": self.decisions, "training_seconds": self.training_seconds}
