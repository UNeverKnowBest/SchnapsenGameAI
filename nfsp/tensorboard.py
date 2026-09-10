"""Mirror existing JSONL experiment events into TensorBoard without training.

python -m nfsp.tensorboard --events runs/<run>/events.jsonl --logdir runs/tensorboard/<run> --follow
"""
import argparse
import json
import math
from pathlib import Path
import re
import time


def scalars(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalars(child, f"{prefix}/{key}" if prefix else key)
    elif isinstance(value, list):
        # Per-player loss dictionaries are useful; raw trajectories are not scalars.
        for index, child in enumerate(value):
            if isinstance(child, dict) or (prefix in ("epsilon", "rl_updates", "sl_updates", "replay_sizes", "reservoir_sizes", "reservoir_seen") and isinstance(child, (int, float))):
                yield from scalars(child, f"{prefix}/player_{index}")
    elif isinstance(value, (int, float)) and math.isfinite(value):
        yield prefix, float(value)


class EventMirror:
    def __init__(self, events, logdir, writer_factory=None):
        self.events, self.logdir = Path(events).resolve(), Path(logdir).resolve()
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.logdir / "mirror_state.json"
        self.state = dict(source=str(self.events), offset=0, counters={}, identity=None)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["source"] != str(self.events):
                raise ValueError("logdir belongs to another event source; choose a new logdir")
        if writer_factory is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError('Install monitoring dependencies: python -m pip install "tensorboard>=2.18,<3"') from error
            writer_factory = lambda path: SummaryWriter(log_dir=str(path), flush_secs=2)
        self.writer_factory, self.writers = writer_factory, {}

    def emit(self, row):
        kind = row.get("kind")
        if kind not in ("train", "evaluation", "validation", "final_test", "test"):
            return
        algorithm = row.get("algorithm")
        seed = row.get("training_seed", row.get("seed"))
        if algorithm is None or seed is None:
            return
        algorithm = re.sub(r"[^A-Za-z0-9_.-]", "_", str(algorithm)).strip(".") or "unknown"
        seed = re.sub(r"[^A-Za-z0-9_.-]", "_", str(seed))
        key = f"{algorithm}/seed_{seed}"
        if key not in self.writers:
            self.writers[key] = self.writer_factory(self.logdir / key)
        writer = self.writers[key]
        counts = self.state["counters"].setdefault(key, dict(games=0, decisions=0, seconds=0.))
        if kind == "train":
            counts["games"] = int(row["total_games"])
            counts["decisions"] = int(row["total_decisions"])
            counts["seconds"] += float(row.get("seconds", 0.))
            for tag, value in scalars(row):
                if tag not in ("seed", "training_seed"):
                    writer.add_scalar(f"train/{tag}", value, counts["games"])
            return
        final = kind in ("final_test", "test")
        if not final:
            for name in ("games", "decisions", "seconds"):
                if name in row:
                    counts[name] = row[name]
        baseline = row.get("baseline", row.get("opponent", "unknown"))
        group = "test" if final else "eval"
        for name in ("win_rate", "mean_game_point_difference", "normalized_entropy"):
            value = row.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value):
                writer.add_scalar(f"{group}/{baseline}/{name}", value, int(counts["games"]))
        if "win_rate" in row:
            writer.add_scalar(f"{group}_by_decisions/{baseline}/win_rate", row["win_rate"], int(counts["decisions"]))
            writer.add_scalar(f"{group}_by_training_ms/{baseline}/win_rate", row["win_rate"], round(counts["seconds"]*1000))

    def read_available(self, limit=512):
        if not self.events.exists():
            return 0
        stat = self.events.stat()
        identity = [stat.st_dev, stat.st_ino]
        if self.state["identity"] not in (None, identity) or stat.st_size < self.state["offset"]:
            raise ValueError("event source was replaced or truncated; choose a new logdir")
        self.state["identity"] = identity
        processed = 0
        with self.events.open("rb") as stream:
            stream.seek(self.state["offset"])
            while processed < limit:
                line = stream.readline()
                # A writer may be in the middle of a JSON record. Wait for the newline.
                if not line or not line.endswith(b"\n"):
                    break
                self.emit(json.loads(line))
                self.state["offset"] = stream.tell()
                processed += 1
        self.flush()
        return processed

    def flush(self):
        for writer in self.writers.values():
            writer.flush()
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.state_path)

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--follow", action="store_true", help="poll appended events; Ctrl+C stops monitoring only")
    parser.add_argument("--poll-seconds", type=float, default=1.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or not .1 <= args.poll_seconds <= 30:
        parser.error("poll-seconds must be in [0.1, 30]")
    if not args.follow and not args.events.is_file():
        parser.error("events file does not exist; --follow can wait for a future file")
    mirror = EventMirror(args.events, args.logdir)
    total = 0
    try:
        while True:
            count = mirror.read_available()
            total += count
            if count == 0:
                if not args.follow:
                    break
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Monitoring stopped; training processes were not modified.")
    finally:
        mirror.close()
    print(json.dumps(dict(processed=total, logdir=str(args.logdir.resolve()))))


if __name__ == "__main__":
    main()
