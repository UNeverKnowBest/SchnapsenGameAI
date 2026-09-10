"""Repeated controlled benchmarks separating batching, threads and training."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from ._native import BatchEnv, FEATURE_VERSION
from .arena import Policy, counter_uniform
from .ppo import PPOConfig, PPOTrainer
from .reports import bars, stacked_times, table, write_csv, write_report
from .trainer import resolve_device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rollout(policy, games, width, workers, seed, env=None):
    """Frozen model and per-game action RNG make the workload scheduling invariant."""
    histories, waves = [], []
    env_seconds = inference_seconds = 0.
    decisions = 0
    current_width = width if env is not None else 0
    started = perf_counter()
    for first in range(0, games, width):
        count = min(width, games-first)
        if count != current_width:
            env, current_width = BatchEnv(count, workers), count
        t = perf_counter()
        states, masks, players, winners, points = env.reset(seed, first)
        env_seconds += perf_counter() - t
        steps = np.zeros(count, np.int64)
        histories.extend([[] for _ in range(count)])
        while (players >= 0).any():
            wave_start = perf_counter()
            lanes = np.flatnonzero(players >= 0)
            t = perf_counter()
            probs = policy.probabilities(states[lanes], masks[lanes].astype(np.bool_))
            cumulative = probs.cumsum(axis=1, dtype=np.float64)
            draws = counter_uniform(seed + 17, first+lanes, steps[lanes]) * cumulative[:, -1]
            draws = np.minimum(draws, np.nextafter(cumulative[:, -1], 0))
            chosen = (draws[:, None] >= cumulative).sum(axis=1)
            inference_seconds += perf_counter() - t
            actions = np.full(count, -1, np.int16)
            actions[lanes] = chosen
            for lane, action in zip(lanes, chosen):
                histories[first+lane].append(int(action))
            steps[lanes] += 1
            decisions += len(lanes)
            t = perf_counter()
            states, masks, players, winners, points = env.step(actions)
            env_seconds += perf_counter() - t
            waves.append((perf_counter() - wave_start) * 1000)
        for lane in range(count):
            histories[first+lane].extend([100+int(winners[lane]), 110+int(points[lane])])
    elapsed = perf_counter() - started
    digest = hashlib.sha256(json.dumps(histories, separators=(",", ":")).encode()).hexdigest()
    return dict(seconds=elapsed, games=games, decisions=decisions, games_per_second=games/elapsed,
                decisions_per_second=decisions/elapsed, environment_seconds=env_seconds,
                inference_seconds=inference_seconds, update_seconds=0.,
                overhead_seconds=max(0., elapsed-env_seconds-inference_seconds),
                batch_latency_p50_ms=float(np.quantile(waves, .5)), batch_latency_p95_ms=float(np.quantile(waves, .95)),
                trajectory_sha256=digest, optimizer_steps=0)


def measure_training(config, games):
    # Warm optimizer and device kernels on a disposable learner, then reset ALL state.
    warm = PPOTrainer(config)
    warm.train_wave(min(games, config.num_envs))
    del warm
    trainer = PPOTrainer(config)
    trainer.env, trainer.width = BatchEnv(config.num_envs, config.workers), config.num_envs
    trainer.env.reset(config.seed, 10_000_000)
    synchronize(trainer.device)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    totals = dict.fromkeys(("environment_seconds", "inference_seconds", "update_seconds", "collection_overhead_seconds"), 0.)
    latencies = []
    started = perf_counter()
    while trainer.games < games:
        metrics = trainer.train_wave(min(config.num_envs, games-trainer.games))
        for key in totals:
            totals[key] += metrics[key]
        latencies.append(metrics["seconds"]*1000)
    synchronize(trainer.device)
    elapsed = perf_counter() - started
    digest = hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in trainer.model.parameters())).hexdigest()
    return dict(seconds=elapsed, games=games, decisions=trainer.decisions, games_per_second=games/elapsed,
                decisions_per_second=trainer.decisions/elapsed, **totals,
                overhead_seconds=max(0., elapsed-sum(totals[k] for k in totals if k != "collection_overhead_seconds")),
                batch_latency_p50_ms=float(np.quantile(latencies, .5)), batch_latency_p95_ms=float(np.quantile(latencies, .95)),
                model_sha256=digest, optimizer_steps=trainer.updates,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None)


def run_performance(args):
    widths = sorted({1, *[int(x) for x in args.env_counts.split(",")]})
    workers = sorted({1, *[int(x) for x in args.worker_counts.split(",")]})
    if min(widths) <= 0 or min(workers) <= 0 or args.games <= 0 or args.repeats < 2:
        raise ValueError("positive games/envs/workers and at least two repeats required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("benchmark output must be empty; choose a new directory")
    device = resolve_device(args.device)
    config = PPOConfig(seed=args.seed, device=str(device), torch_threads=args.torch_threads,
                       num_envs=max(widths), hidden=tuple(args.hidden), deterministic=True)
    initial = PPOTrainer(config)
    model = initial.model.eval()
    policies = {"environment": Policy("random"), "frozen_policy": Policy("frozen", model, device, "ppo")}
    modes = [(kind, width, worker) for kind in policies for width in widths for worker in workers
             if width > 1 or worker == 1]
    if not args.no_training:
        modes += [("training", max(widths), worker) for worker in workers]
    raw = []
    rng = np.random.default_rng(args.seed + 900)
    with (output / "measurements.jsonl").open("w", encoding="utf-8") as log:
        for repeat in range(args.repeats):
            # Randomized order mitigates systematic warm-up / thermal ordering bias.
            for index in rng.permutation(len(modes)):
                kind, width, worker = modes[index]
                if kind == "training":
                    measured = measure_training(replace(config, num_envs=width, workers=worker), args.games)
                else:
                    env = BatchEnv(width, worker)
                    rollout(policies[kind], width, width, worker, args.seed + 10_000_000, env)
                    synchronize(device)
                    measured = rollout(policies[kind], args.games, width, worker, args.seed, env)
                row = dict(mode=kind, num_envs=width, workers=worker, repeat=repeat, **measured)
                raw.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({k: row[k] for k in ("mode", "num_envs", "workers", "repeat", "games_per_second")}), flush=True)
    # Refuse to claim controlled speedup when executed trajectories differ.
    checks = {}
    for kind in policies:
        checks[kind] = len({r["trajectory_sha256"] for r in raw if r["mode"] == kind}) == 1
    if not args.no_training:
        checks["training"] = len({r["model_sha256"] for r in raw if r["mode"] == "training"}) == 1
    summary = []
    for kind, width, worker in modes:
        rows = [r for r in raw if (r["mode"], r["num_envs"], r["workers"]) == (kind, width, worker)]
        baseline_width = max(widths) if kind == "training" else 1
        baselines = [r for r in raw if (r["mode"], r["num_envs"], r["workers"]) == (kind, baseline_width, 1)]
        thread_baselines = [r for r in raw if (r["mode"], r["num_envs"], r["workers"]) == (kind, width, 1)]
        throughput = [r["games_per_second"] for r in rows]
        ratios = [r["games_per_second"]/next(b["games_per_second"] for b in baselines if b["repeat"] == r["repeat"]) for r in rows]
        thread_ratios = [r["games_per_second"]/next(b["games_per_second"] for b in thread_baselines if b["repeat"] == r["repeat"]) for r in rows]
        row = dict(mode=kind, num_envs=width, workers=worker, repeats=args.repeats,
                   games_per_second_mean=float(np.mean(throughput)), games_per_second_std=float(np.std(throughput, ddof=1)),
                   games_per_second_min=min(throughput), games_per_second_max=max(throughput),
                   speedup_mean=float(np.mean(ratios)), thread_speedup_mean=float(np.mean(thread_ratios)),
                   thread_efficiency=float(np.mean(thread_ratios))/worker,
                   matched_workload=checks[kind])
        for key in ("environment_seconds", "inference_seconds", "update_seconds", "overhead_seconds",
                    "batch_latency_p50_ms", "batch_latency_p95_ms", "optimizer_steps"):
            row[key] = float(np.mean([r[key] for r in rows]))
        summary.append(row)
    metadata = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    os=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    device=str(device), feature_version=FEATURE_VERSION, config=config.to_dict(),
                    games=args.games, repeats=args.repeats,
                    methodology="Fixed seeds and game-local RNG; warm-up excluded; randomized order; synchronize GPU. Frozen modes vary batch/threads; training holds rollout size and update budget fixed. Instrumentation included; checkpoint/evaluation excluded.")
    data = dict(metadata=metadata, matched_workload_checks=checks, summary=summary, measurements=raw)
    render_performance(output, data)
    if not all(checks.values()):
        raise RuntimeError(f"workload mismatch: {checks}; report retained, speedup is not a controlled comparison")
    return data


def render_performance(output, data):
    summary, metadata = data["summary"], data["metadata"]
    body = '<p>分离环境执行、冻结策略采样和完整 PPO 训练。误差线表示重复测量的最小值至最大值；不是置信区间。模型与轨迹哈希检查用于核实并行前后的工作量。</p>'
    for kind, title in (("environment", "随机动作环境执行（含 Python 数据传输）"),
                        ("frozen_policy", "冻结神经网络 · 批量推理与线程"), ("training", "完整 PPO 训练 · 固定采样批次和更新预算")):
        rows = [r for r in summary if r["mode"] == kind]
        if not rows:
            continue
        labels = [f'{r["num_envs"]} env / {r["workers"]} workers' for r in rows]
        body += bars(title, labels, [r["games_per_second_mean"] for r in rows], "games/s",
                     [[r["games_per_second_min"], r["games_per_second_max"]] for r in rows])
        body += table(["并行局数", "线程", "整体加速", "同批次线程加速", "线程效率", "P95 批次延迟 ms", "工作量一致"],
                      [[r["num_envs"], r["workers"], f'{r["speedup_mean"]:.2f}×', f'{r["thread_speedup_mean"]:.2f}×',
                        f'{r["thread_efficiency"]:.1%}', f'{r["batch_latency_p95_ms"]:.2f}', r["matched_workload"]] for r in rows])
        body += stacked_times(labels, rows)
        body += table(["配置", "环境 s", "推理 s", "更新 s", "其他 s"],
                      [[label, *[f'{r[k]:.4f}' for k in ("environment_seconds", "inference_seconds", "update_seconds", "overhead_seconds")]]
                       for label, r in zip(labels, rows)])
    body += '<p>整体加速的参考是单局单线程；完整训练的参考是相同批次下单线程。批次延迟是一轮 step/训练 wave 的延迟，不是单局响应时间。增加线程可能变慢，结果保留负收益。未比较旧 Python 引擎与 Rust，也未将批量化收益归因于线程。</p>'
    body += table(["设备", "CPU 逻辑核心", "PyTorch 线程", "重复次数", "局数"],
                  [[metadata["device"], metadata["logical_cpus"], metadata["config"]["torch_threads"], metadata["repeats"], metadata["games"]]])
    write_report(output, data, "Schnapsen · 并行性能分析", body)
    write_csv(output, summary)
