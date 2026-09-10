import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import torch
from .config import Config
from .trainer import Trainer, CHECKPOINT_VERSION, resolve_device
from ._native import FEATURE_VERSION
from .networks import Network
from .evaluation import evaluate_policy

def make_parser():
    parser = argparse.ArgumentParser(description="Rust batched Schnapsen self-play and research benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    from .research_cli import register
    register(sub)
    train = sub.add_parser("train", help="train or resume a full NFSP snapshot")
    train.add_argument("--config", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--games", type=int, default=100_000, help="absolute completed-game target")
    train.add_argument("--output", type=Path, default=Path("runs/nfsp"))
    train.add_argument("--device")
    train.add_argument("--workers", type=int)
    train.add_argument("--num-envs", type=int)
    train.add_argument("--save-every", type=int, default=10_000)
    train.add_argument("--eval-every", type=int, default=10_000)
    train.add_argument("--eval-games", type=int, default=1_000)

    evaluate = sub.add_parser("evaluate", help="evaluate exported average policies")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--opponent", type=Path, help="exported opponent policy checkpoint")
    evaluate.add_argument("--games", type=int, default=2_000)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--workers", type=int, default=8)
    evaluate.add_argument("--num-envs", type=int, default=256)
    evaluate.add_argument("--seed", type=int, default=1_000_042)
    evaluate.add_argument("--output", type=Path)

    bench = sub.add_parser("benchmark", help="measure inference, environment and end-to-end training")
    bench.add_argument("--config", type=Path)
    bench.add_argument("--games", type=int, default=1_024)
    bench.add_argument("--env-counts", default="1,64,256")
    bench.add_argument("--worker-counts", default="1,8")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    bench.add_argument("--seed", type=int, default=42)
    return parser

def load_policies(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != CHECKPOINT_VERSION or state.get("feature_version") != FEATURE_VERSION:
        raise ValueError("incompatible checkpoint feature/version")
    config = Config(**state["config"]).validate()
    weights = state.get("policies")
    if weights is None:
        weights = [agent["policy"] for agent in state["learners"]]
    result = []
    for weight in weights:
        model = Network(hidden=config.hidden).to(device).eval()
        model.load_state_dict(weight)
        result.append(model)
    return result

def benchmark(args):
    if args.games <= 0:
        raise ValueError("games must be positive")
    counts = [int(n) for n in args.env_counts.split(",")]
    workers = [int(n) for n in args.worker_counts.split(",")]
    base = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config(
        replay_capacity=20_000, reservoir_capacity=20_000, rl_warmup=256, sl_warmup=256)
    results = []
    for count in counts:
        for worker in workers:
            config = replace(base, seed=args.seed, num_envs=count, workers=worker, device=args.device)
            trainer = Trainer(config)
            # Warm kernels/thread pools using an independent collector pass, without learning.
            trainer.collector.collect(trainer.learners, 10_000_000, min(32, count))
            totals = dict.fromkeys(("environment_seconds", "inference_seconds", "collection_overhead_seconds",
                                   "buffer_seconds", "update_seconds"), 0.)
            started = perf_counter()
            while trainer.games < args.games:
                metrics = trainer.train_wave(min(count, args.games - trainer.games))
                for key in totals:
                    totals[key] += metrics[key]
            elapsed = perf_counter() - started
            row = {"num_envs": count, "workers": worker, "games": trainer.games,
                   "decisions": trainer.decisions, "seconds": elapsed,
                   "games_per_second": trainer.games / elapsed,
                   "decisions_per_second": trainer.decisions / elapsed,
                   "rl_updates": [a.rl_updates for a in trainer.learners],
                   "sl_updates": [a.sl_updates for a in trainer.learners], **totals}
            results.append(row)
            print(json.dumps(row), flush=True)
            del trainer
    report = {
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(resolve_device(args.device)), "logical_cpus": os.cpu_count(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "seed": args.seed, "config": base.to_dict(), "results": results,
        "note": "End-to-end training, excluding checkpoint/evaluation. Different wave sizes change update timing and sampled trajectories; this is throughput evidence, not equal learning quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

def main(argv=None):
    args = make_parser().parse_args(argv)
    from .research_cli import dispatch
    if dispatch(args):
        return
    if args.command == "train":
        if args.games <= 0 or args.save_every < 0 or args.eval_every < 0:
            raise ValueError("games must be positive; intervals must be nonnegative")
        if args.eval_every and (args.eval_games <= 0 or args.eval_games % 2):
            raise ValueError("eval-games must be positive and even")
        if args.resume:
            if args.config or args.num_envs is not None:
                raise ValueError("resume restores its configuration; only device/workers may be overridden")
            trainer = Trainer.load(args.resume, args.device, args.workers)
        else:
            if (args.output / "latest.pt").exists():
                raise FileExistsError("output already contains a training checkpoint; use --resume or a new output directory")
            config = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config()
            for name in ("device", "workers", "num_envs"):
                value = getattr(args, name)
                if value is not None:
                    setattr(config, name, value)
            trainer = Trainer(config)
        print(json.dumps({"device": str(trainer.device), "torch": torch.__version__,
                          "config": trainer.config.to_dict()}), flush=True)
        print(json.dumps(trainer.run(args.games, args.output, args.save_every, args.eval_every, args.eval_games)))
    elif args.command == "evaluate":
        device = resolve_device(args.device)
        torch.set_num_threads(4)
        policies = load_policies(args.checkpoint, device)
        opponents = load_policies(args.opponent, device) if args.opponent else [None, None]
        report = [evaluate_policy(p, device, args.games, args.num_envs, args.workers, args.seed, o)
                  for p, o in zip(policies, opponents, strict=True)]
        print(json.dumps(report, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        benchmark(args)
