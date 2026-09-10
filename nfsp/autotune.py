"""Hardware calibration and end-to-end policy request latency (no network service)."""
from dataclasses import replace
import gc
import json
import os
from time import perf_counter

import numpy as np
import torch

from .arena import Policy
from .learner import categorical
from .performance import synchronize
from .ppo import ActorCritic, PPOTrainer


def available_devices():
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def calibrate(config, output, repeats=3, requested_device="auto", max_envs=1024):
    """Select by median full training time, including collection and optimization.

    Widths have equal game/epoch budgets, but may have different update counts;
    these are configuration trials, not an isolated hardware speedup experiment.
    """
    widths = sorted({min(256, max(1, max_envs // 2)), max_envs})
    threads = sorted({1, min(4, os.cpu_count() or 1)})
    workers = sorted({1, min(4, os.cpu_count() or 1)})
    candidates = []
    for width in widths:
        for thread in threads:
            candidates.append(replace(config, device="cpu", rollout_device="same", num_envs=width,
                                      torch_threads=thread, workers=1, adaptive_lr=False))
        if len(workers) > 1:
            candidates.append(replace(config, device="cpu", rollout_device="same", num_envs=width,
                                      torch_threads=1, workers=workers[-1], adaptive_lr=False))
        if torch.cuda.is_available():
            for actor in ("same", "cpu"):
                candidates.append(replace(config, device="cuda", rollout_device=actor, num_envs=width,
                                          torch_threads=1, workers=1, adaptive_lr=False))
    games = max(widths) * 2
    raw, errors, warmed = [], [], set()
    rng = np.random.default_rng(config.seed + 77)
    with (output / "calibration.jsonl").open("w", encoding="utf-8") as log:
        for repeat in range(repeats):
            for index in rng.permutation(len(candidates)):
                index = int(index)
                c = candidates[index]
                if any(e["candidate"] == index for e in errors):
                    continue
                trainer = warm = None
                try:
                    if index not in warmed:
                        warm = PPOTrainer(c)
                        warm.train_wave()
                        warm = None
                        warmed.add(index)
                    trainer = PPOTrainer(c)
                    synchronize(trainer.device)
                    if c.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    start = perf_counter()
                    breakdown = dict(environment_seconds=0., inference_seconds=0., update_seconds=0.)
                    while trainer.games < games:
                        metrics = trainer.train_wave(min(c.num_envs, games - trainer.games))
                        for key in breakdown:
                            breakdown[key] += metrics[key]
                    synchronize(trainer.device)
                    elapsed = perf_counter() - start
                    row = dict(candidate=index, repeat=repeat, config=c.to_dict(), games=trainer.games,
                               decisions=trainer.decisions, optimizer_steps=trainer.updates,
                               seconds=elapsed, games_per_second=games/elapsed,
                               decisions_per_second=trainer.decisions/elapsed, **breakdown,
                               cuda_peak_bytes=torch.cuda.max_memory_allocated() if c.device == "cuda" else None)
                    raw.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(dict(stage="calibration", candidate=index, repeat=repeat,
                                          device=c.device, actor=c.rollout_device, envs=c.num_envs,
                                          threads=c.torch_threads, workers=c.workers,
                                          games_per_second=round(row["games_per_second"], 1))), flush=True)
                    trainer = None
                except torch.cuda.OutOfMemoryError as exc:
                    errors.append(dict(candidate=index, error=str(exc)))
                    trainer = warm = None
                    gc.collect()
                    torch.cuda.empty_cache()
    summary = []
    for index, c in enumerate(candidates):
        rows = [r for r in raw if r["candidate"] == index]
        if len(rows) != repeats:
            continue
        speeds = [r["games_per_second"] for r in rows]
        summary.append(dict(candidate=index, config=c.to_dict(),
                            median_games_per_second=float(np.median(speeds)),
                            min_games_per_second=min(speeds), max_games_per_second=max(speeds),
                            median_decisions_per_second=float(np.median([r["decisions_per_second"] for r in rows])),
                            optimizer_steps=[r["optimizer_steps"] for r in rows]))
    eligible = [r for r in summary if requested_device == "auto" or r["config"]["device"] == requested_device]
    if not eligible:
        raise RuntimeError(f"no successful calibration for device {requested_device}")
    fastest = max(eligible, key=lambda r: r["median_games_per_second"])
    # Prefer simpler CPU execution when performance differs by less than 5%.
    near = [r for r in eligible if r["median_games_per_second"] >= fastest["median_games_per_second"] * .95]
    best = min(near, key=lambda r: (r["config"]["device"] != "cpu", r["config"]["workers"],
                                    r["config"]["torch_threads"], -r["median_games_per_second"]))
    selected = replace(config, **{k: best["config"][k] for k in
                                 ("device", "rollout_device", "num_envs", "workers", "torch_threads")})
    data = dict(selected=selected.to_dict(), summary=summary, measurements=raw, errors=errors,
                gpu_status="measured" if torch.cuda.is_available() else "unavailable; CPU only",
                methodology="Warm-up excluded; randomized repeated trials; equal games and PPO epochs. "
                            "Widths change rollout/update counts, so configuration speed ratios are not isolated hardware speedups. "
                            "Selection uses median full training throughput, preferring CPU within 5%.")
    (output / "calibration.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return selected, data


@torch.inference_mode()
def measure_latency(model, states, masks, widths=(1, 256), repeats=100, threads=1):
    """Same weights and observations on every device; host input to host action."""
    torch.set_num_threads(threads)
    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hidden = [layer.out_features for layer in model.body if isinstance(layer, torch.nn.Linear)]
    results = []
    for device in available_devices():
        actor = ActorCritic(hidden).to(device).eval()
        actor.load_state_dict(weights)
        policy = Policy("latency", actor, torch.device(device), "ppo")
        for width in sorted(set(widths)):
            ids = np.arange(width) % len(states)
            x, legal = states[ids], masks[ids]
            rng = np.random.default_rng(900)
            for _ in range(10):
                categorical(policy.probabilities(x, legal), rng)
            synchronize(torch.device(device))
            latencies = []
            for _ in range(repeats):
                started = perf_counter()
                # probabilities() returns a CPU NumPy array, so its CUDA-to-CPU
                # copy already waits for inference. A second device barrier would
                # measure extra instrumentation rather than host-action latency.
                actions = categorical(policy.probabilities(x, legal), rng)
                latencies.append((perf_counter() - started) * 1000)
                if not legal[np.arange(width), actions].all():
                    raise RuntimeError("latency actor produced an illegal action")
            results.append(dict(device=device, batch_size=width, samples=repeats,
                                p50_ms=float(np.quantile(latencies, .5)),
                                p95_ms=float(np.quantile(latencies, .95)),
                                p99_ms=float(np.quantile(latencies, .99)),
                                mean_ms=float(np.mean(latencies)),
                                amortized_ms_per_action=float(np.mean(latencies))/width,
                                actions_per_second=width/(float(np.mean(latencies))/1000),
                                raw_ms=latencies))
        del actor, policy
    return results
