"""CPU array storage; reservoir Algorithm R samples uniformly over the full stream."""
import copy
import numpy as np

STATE_DIM = 465
ACTION_DIM = 28
RL_SCHEMA = {
    "states": ((STATE_DIM,), np.float32), "actions": ((), np.int64),
    "rewards": ((), np.float32), "next_states": ((STATE_DIM,), np.float32),
    "next_masks": ((ACTION_DIM,), np.bool_), "dones": ((), np.bool_),
}
SL_SCHEMA = {
    "states": ((STATE_DIM,), np.float32), "probs": ((ACTION_DIM,), np.float32),
    "masks": ((ACTION_DIM,), np.bool_),
}

class ArrayBuffer:
    def __init__(self, capacity, schema, seed):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.schema = capacity, schema
        self.data = {key: np.empty((capacity, *shape), dtype=dtype)
                     for key, (shape, dtype) in schema.items()}
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return min(self.seen, self.capacity)

    def _validate(self, batch):
        if batch.keys() != self.schema.keys():
            raise ValueError("incorrect buffer fields")
        count = len(next(iter(batch.values())))
        for key, (shape, dtype) in self.schema.items():
            if batch[key].shape != (count, *shape) or batch[key].dtype != np.dtype(dtype):
                raise ValueError(f"invalid shape/dtype for {key}")
        return count

    def sample(self, batch_size):
        if batch_size > len(self):
            raise ValueError("not enough samples")
        ids = self.rng.choice(len(self), batch_size, replace=False)
        return {key: array[ids] for key, array in self.data.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "seen": self.seen,
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "data": {key: array[:len(self)].copy() for key, array in self.data.items()}}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state["seen"] < 0:
            raise ValueError("incompatible buffer checkpoint")
        count = self._validate(state["data"])
        if count != min(state["seen"], self.capacity):
            raise ValueError("invalid buffer checkpoint length")
        for key in self.data:
            self.data[key][:count] = state["data"][key]
        self.seen = state["seen"]
        self.rng.bit_generator.state = state["rng"]

class ReplayBuffer(ArrayBuffer):
    def __init__(self, capacity, seed=0):
        super().__init__(capacity, RL_SCHEMA, seed)

    def add(self, batch):
        count = self._validate(batch)
        # If a batch exceeds capacity, keep exactly its most recent capacity rows.
        start = max(0, count - self.capacity)
        ids = (self.seen + np.arange(start, count)) % self.capacity
        for key in self.data:
            self.data[key][ids] = batch[key][start:]
        self.seen += count

class ReservoirBuffer(ArrayBuffer):
    def __init__(self, capacity, seed=0):
        super().__init__(capacity, SL_SCHEMA, seed)

    def add(self, batch):
        count = self._validate(batch)
        fill = min(count, max(0, self.capacity - self.seen))
        if fill:
            for key in self.data:
                self.data[key][self.seen:self.seen + fill] = batch[key][:fill]
        remaining = count - fill
        if remaining:
            # Independent Algorithm R draws, one for each incoming stream position.
            high = np.arange(self.seen + fill + 1, self.seen + count + 1, dtype=np.int64)
            destinations = self.rng.integers(0, high)
            sources = np.flatnonzero(destinations < self.capacity)
            destinations = destinations[sources]
            # Multiple incoming rows may replace the same slot: the last one wins.
            _, reverse_ids = np.unique(destinations[::-1], return_index=True)
            keep = len(destinations) - 1 - reverse_ids
            for key in self.data:
                self.data[key][destinations[keep]] = batch[key][fill + sources[keep]]
        self.seen += count
