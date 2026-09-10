"""One command: calibrate hardware, train, compare held-out baselines, export figures.

Run with the project interpreter: .venv/Scripts/python.exe experiment.py --demo
"""
import argparse
import copy
import csv
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
from time import perf_counter

import numpy as np
import torch

from nfsp._native import FEATURE_VERSION
from nfsp.arena import Policy, checkpoint_policies, evaluate_match
from nfsp.autotune import calibrate, measure_latency
from nfsp.learning import (learning_chart, seed_summary, add_learning_arguments,
                           learning_protocol, export_learning, plateau_table, write_live)
from nfsp.ppo import PPOConfig, PPOTrainer
from nfsp.reports import bars, escape, percent_interval, table, write_csv, write_report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_learning_arguments(p)
    p.add_argument("--demo", action="store_true", help="4 x 512 games, one seed; pipeline/performance check")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--rounds", type=int, help="evaluation rounds (default 20, demo 4)")
    p.add_argument("--games-per-round", type=int, help="completed games per round (default 5000, demo 512)")
    p.add_argument("--epochs", type=int, default=4, help="maximum PPO optimization passes per rollout")
    p.add_argument("--max-decisions", type=int, help="per-seed primary decision budget; finish current wave, then match PPO games")
    p.add_argument("--eval-games", type=int, help="even validation games per opponent (default 1000, demo 256)")
    p.add_argument("--test-games", type=int, help="even FINAL held-out games per opponent (default 4000, demo 512)")
    p.add_argument("--seeds", help="distinct training seeds, default 42,43,44; demo 42")
    p.add_argument("--patience", type=int, default=5, help="rounds without validation improvement before early stopping")
    p.add_argument("--min-rounds", type=int, default=8)
    p.add_argument("--tune-repeats", type=int, default=3)
    p.add_argument("--latency-repeats", type=int, default=100)
    p.add_argument("--skip-tuning", action="store_true", help="use conservative CPU defaults; no hardware speed claim")
    p.add_argument("--num-envs", type=int, help="fixed width without tuning, otherwise maximum candidate width")
    p.add_argument("--workers", type=int, default=1, help="worker count without tuning")
    p.add_argument("--torch-threads", type=int, default=1, help="PyTorch threads without tuning")
    p.add_argument("--baseline-checkpoint", type=Path, action="append", default=[])
    p.add_argument("--render-only", type=Path, help="regenerate report/figures from an existing results.json")
    return p


def settings(args):
    learning_protocol(args)
    defaults = dict(rounds=4 if args.demo else 20, games_per_round=512 if args.demo else 5000,
                    eval_games=256 if args.demo else 1000, test_games=512 if args.demo else 4000,
                    num_envs=256 if args.demo else 1024)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.seeds = [int(s) for s in (args.seeds or ("42" if args.demo else "42,43,44")).split(",")]
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**63-1000 for s in args.seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**63-1000)")
    for key in (*defaults, "epochs", "workers", "torch_threads", "patience", "min_rounds", "latency_repeats"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.eval_games % 2 or args.test_games % 2:
        raise ValueError("eval-games and test-games must be even for paired seats")
    if args.tune_repeats < 2:
        raise ValueError("tune-repeats must be at least 2")
    if args.max_decisions is not None and args.max_decisions <= 0:
        raise ValueError("max-decisions must be positive")
    if set(args.seeds) & {2_000_042, 7_000_042}:
        raise ValueError("training seeds must differ from reserved validation/test seeds")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    for path in args.baseline_checkpoint:
        if not path.is_file():
            raise FileNotFoundError(path)
    return args


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def csv_file(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def curve_figure(rows, key, title):
    if not rows:
        return ""
    xmax = max(r["total_games"] for r in rows)
    values = [r[key] for r in rows if r.get(key) is not None]
    high = max([1e-9, *values]) * 1.1
    low = min([0., *values])
    svg = f'<svg viewBox="0 0 1100 340" role="img"><title>{escape(title)}</title><path d="M70 25V290H1060" stroke="#aaa" fill="none"/>'
    groups = sorted({(r["algorithm"], r["seed"]) for r in rows})
    for i, (algorithm, seed) in enumerate(groups):
        group = [r for r in rows if r["algorithm"] == algorithm and r["seed"] == seed and r.get(key) is not None]
        color = ["#55cad2", "#ffa366", "#ae9cff", "#e97baa", "#bcd75e", "#9eafce"][i % 6]
        points = " ".join(f'{70+r["total_games"]/max(xmax,1)*970:.2f},{290-(r[key]-low)/(high-low)*250:.2f}' for r in group)
        svg += f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>'
        svg += f'<text x="{80+i*160}" y="330" fill="{color}" font-size="12">{escape(algorithm)} / {seed}</text>'
    svg += f'<text x="2" y="35" fill="#ddd">{high:.3g}</text><text x="2" y="290" fill="#ddd">{low:.3g}</text><text x="760" y="310" fill="#ddd">{xmax} completed games</text></svg>'
    return f'<div class="card"><h2>{escape(title)}</h2>{svg}</div>'


def assess_performance(data):
    summary = data["calibration"].get("summary", [])
    best = {}
    for device in ("cpu", "cuda"):
        rows = [r for r in summary if r["config"]["device"] == device]
        if rows:
            best[device] = max(rows, key=lambda r: r["median_games_per_second"])
    ratio = (best["cuda"]["median_games_per_second"] / best["cpu"]["median_games_per_second"]
             if len(best) == 2 else None)
    requests = []
    for width in sorted({r["batch_size"] for r in data["latency"]}):
        rows = [r for r in data["latency"] if r["batch_size"] == width]
        chosen = min(rows, key=lambda r: r["p95_ms"])
        requests.append(dict(batch_size=width, preferred_device=chosen["device"], p95_ms=chosen["p95_ms"]))
    return dict(best_training_configs=best, cuda_over_cpu_best_config_ratio=ratio,
                cuda_slower_than_best_cpu=None if ratio is None else ratio < 1,
                latency_recommendations=requests,
                qualification="Measured configuration medians, not a guarantee under changed load; CPU and CUDA optima may use different thread/batch settings.")


def render(output, data):
    data["performance_assessment"] = assessment = assess_performance(data)
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True, parents=True)
    cards = {}
    for baseline in ("random", "heuristic"):
        cards[f"learning_{baseline}"] = learning_chart(data["curves"], baseline)
        for axis in ("decisions", "seconds"):
            cards[f"learning_{baseline}_{axis}"] = learning_chart(data["curves"], baseline, axis)
    for key, title in (("lr_used", "Learning rate"), ("normalized_entropy", "Normalized legal entropy"),
                       ("final_kl", "Final policy KL"), ("games_per_second", "Training throughput (games/s)"),
                       ("value_loss", "Value loss")):
        cards[key] = curve_figure(data["training"], key, title)
    finals = data["summary"]
    cards["baselines"] = bars("Held-out win rate", [f'{r["algorithm"]} vs {r["opponent"]}' for r in finals],
                               [r["mean"]*100 for r in finals], "%",
                               [[x*100 for x in (r["ci95"] or [r["mean"], r["mean"]])] for r in finals])
    latencies = data["latency"]
    cards["latency"] = bars("Policy request latency P95", [f'{r["device"]} / batch {r["batch_size"]}' for r in latencies],
                             [r["p95_ms"] for r in latencies], "ms")
    calibration = data["calibration"]
    if calibration.get("summary"):
        rows = calibration["summary"]
        cards["calibration"] = bars("Full training configuration trials", [f'{r["config"]["device"]}/{r["config"]["rollout_device"]} e{r["config"]["num_envs"]} t{r["config"]["torch_threads"]} w{r["config"]["workers"]}' for r in rows],
                                    [r["median_games_per_second"] for r in rows], "games/s",
                                    [[r["min_games_per_second"], r["max_games_per_second"]] for r in rows])
    links = []
    for name, card in cards.items():
        for svg in re.findall(r"<svg\b.*?</svg>", card, re.S):
            standalone = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
            standalone = standalone.replace('role="img"', 'style="background:#19263b;font-family:Arial,sans-serif" role="img"')
            (figures / f"{name}.svg").write_text(standalone, encoding="utf-8")
            links.append(f'<a href="figures/{name}.svg">{name}.svg</a>')
    selected = data["config"]
    body = f'<p>训练设备：{escape(selected["device"])}；采样设备：{escape(selected["rollout_device"])}；并行局数：{selected["num_envs"]}；PyTorch 线程：{selected["torch_threads"]}；环境线程：{selected["workers"]}。</p>'
    ratio = assessment["cuda_over_cpu_best_config_ratio"]
    if ratio is not None:
        body += f'<p>当前校准中，CUDA 最佳配置吞吐量 / CPU 最佳配置吞吐量 = {ratio:.2f}。这是各自调优配置的比较，包含批量和线程差异。完整训练设备按重复测量选取；短测和长期负载下可能不同。</p>'
    for recommendation in assessment["latency_recommendations"]:
        body += f'<p>batch={recommendation["batch_size"]} 的实测低延迟选择：{escape(recommendation["preferred_device"])}，P95 {recommendation["p95_ms"]:.3f} ms。</p>'
    body += '<p>训练轮次是评估区间；wave 是冻结行为策略采集的一批完整对局；decision 是环境动作；optimizer step 是一次梯度更新。完整记录保留所有计数。</p>'
    body += table(["算法", "seed", "训练局数", "decisions", "optimizer steps", "轮次", "训练秒数", "停止原因", "最佳轮次"],
                  [[r[k] for k in ("algorithm", "seed", "games", "decisions", "updates", "rounds", "training_seconds", "stop_reason", "best_round")] for r in data["runs"]])
    body += '<p>标准 PPO 使用同样网络、训练局数和最大优化 epochs，固定 lr=3e-4 / entropy=0.01；主算法自适应 lr 与 entropy。按各自验证集成绩选最佳检查点，最终评估使用独立种子。轨迹长度与 KL 提前停止会使实际梯度步数不同。</p>'
    body += '<p>baseline 胜率图误差线是训练种子间 95% t 区间；单种子没有训练不确定性区间。下面表格给出每次测试的成对牌组 bootstrap 区间。小 demo 仅验证流程与速度，不能证明收敛或优于 baseline。</p>'
    body += table(["算法", "seed", "对手", "胜率", "成对 bootstrap 95% CI", "游戏点差"],
                  [[m["algorithm"], m["training_seed"], m["opponent"], f'{m["win_rate"]:.2%}',
                    percent_interval(m["win_uncertainty"]["ci95_bootstrap"]), f'{m["mean_game_point_difference"]:.3f}'] for m in data["matches"]])
    body += ''.join(cards.values())
    body += '<p>Latency 从 CPU 观测输入计时至 CPU 动作输出，含传输、合法动作 softmax 和采样；GPU 预热并同步。同模型/同输入，batch=1 表示单次决策，批量耗时除以动作数只表示摊销成本；不含 HTTP、排队或完整游戏时间。</p>'
    body += table(["设备", "batch", "P50 ms", "P95 ms", "P99 ms", "actions/s"],
                  [[r["device"], r["batch_size"], *[f'{r[k]:.3f}' for k in ("p50_ms", "p95_ms", "p99_ms", "actions_per_second")]] for r in latencies])
    body += f'<p>{escape(calibration.get("methodology", "Hardware tuning skipped."))}</p>'
    body += '<p>独立 SVG 文件：' + ' · '.join(links) + '</p>'
    analysis_protocol = data["protocol"].get("learning_analysis", dict(window=5, delta=.02, min_rounds=10))
    data["plateau"] = export_learning(output, data["curves"], analysis_protocol)
    body += plateau_table(data["plateau"], analysis_protocol)
    write_report(output, data, "Schnapsen · 一键训练与系统性能", body)
    write_live(output, data["curves"], analysis_protocol, complete=True)
    write_csv(output, finals)
    csv_file(output / "training.csv", data["training"])
    csv_file(output / "learning.csv", data["curves"])
    csv_file(output / "latency.csv", [{k:v for k,v in r.items() if k != "raw_ms"} for r in latencies])


def run(args):
    args = settings(args)
    started = perf_counter()
    output = args.output or Path("runs") / ("demo_" if args.demo else "experiment_") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("output must be empty; choose a new directory")
    config = PPOConfig(seed=args.seeds[0], device="cpu" if args.device == "auto" else args.device,
                       num_envs=min(256, args.num_envs), workers=args.workers, torch_threads=args.torch_threads,
                       epochs=args.epochs, adaptive_lr=True)
    frozen = []
    for i, path in enumerate(args.baseline_checkpoint):
        frozen.extend(checkpoint_policies(path, "cpu", f"frozen{i}:{path.stem}"))
    protocol = dict(arguments={k:str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "baseline_checkpoint" else v for k,v in vars(args).items()},
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    feature_version=FEATURE_VERSION, validation_seed=2_000_042, test_seed=7_000_042,
                    frozen_baselines=[p.metadata for p in frozen],
                    budget="Per seed: rounds * games_per_round maximum; optional decision cap checked after complete primary waves; standard PPO matches actual primary games.",
                    selection="Mean random/heuristic validation win rate; best checkpoint includes round zero. Early stop after patience rounds with no improvement of >=0.005, after min_rounds.")
    protocol["learning_analysis"] = learning_protocol(args)
    dump(output / "protocol.json", protocol)
    if args.skip_tuning:
        config = replace(config, num_envs=args.num_envs)
        calibration = dict(summary=[], selected=config.to_dict(), methodology="Calibration skipped; no CPU/GPU training comparison available.")
    else:
        config, calibration = calibrate(config, output, args.tune_repeats, args.device, min(args.num_envs, args.games_per_round))
    dump(output / "config.json", config.to_dict())
    print(json.dumps(dict(stage="selected", config=config.to_dict(), output=str(output))), flush=True)
    curves, training, runs, matches, latency = [], [], [], [], []
    write_live(output, curves, protocol["learning_analysis"])
    with (output / "events.jsonl").open("w", encoding="utf-8") as events:
        def event(data):
            events.write(json.dumps(data, allow_nan=False) + "\n")
            events.flush()
        for seed in args.seeds:
            trainers = {"adaptive-ppo": PPOTrainer(replace(config, seed=seed)),
                        "ppo": PPOTrainer(replace(config, seed=seed, adaptive_lr=False, adaptive_entropy=False, entropy_coef=.01))}
            paths = {name: output / f"{name}_seed{seed}" for name in trainers}
            for path in paths.values():
                path.mkdir()
            best = {name: dict(score=-1., round=0, stale=0) for name in trainers}
            stop_reason, completed_round = "round_budget", 0
            for round_id in range(args.rounds + 1):
                target = round_id * args.games_per_round
                for name, trainer in trainers.items():
                    if name == "ppo":
                        target = trainers["adaptive-ppo"].games
                    while trainer.games < target:
                        if name == "adaptive-ppo" and args.max_decisions and trainer.decisions >= args.max_decisions:
                            stop_reason = "decision_budget"
                            break
                        metrics = trainer.train_wave(min(config.num_envs, target-trainer.games))
                        row = dict(algorithm=name, seed=seed, round=round_id,
                                   **{k:v for k,v in metrics.items() if k != "losses"}, **metrics["losses"])
                        training.append(row)
                        event(dict(kind="train", **row))
                    # CPU validation avoids many tiny CUDA inference launches. It cannot mutate training RNG.
                    actor = Policy(name, copy.deepcopy(trainer.model).cpu().eval(), "cpu", "ppo")
                    scores = []
                    for baseline in ("random", "heuristic"):
                        m = evaluate_match(actor, Policy(baseline, kind=baseline), args.eval_games,
                                           config.num_envs, config.workers, protocol["validation_seed"])
                        scores.append(m["win_rate"])
                        row = dict(algorithm=name, seed=seed, round=round_id, baseline=baseline,
                                   games=trainer.games, decisions=trainer.decisions, seconds=trainer.training_seconds, win_rate=m["win_rate"])
                        curves.append(row)
                        event(dict(kind="validation", **row))
                    write_live(output, curves, protocol["learning_analysis"])
                    score = float(np.mean(scores))
                    if score > best[name]["score"]:
                        # Any improvement selects the checkpoint; min delta controls stopping only.
                        meaningful = score >= best[name]["score"] + .005
                        best[name] = dict(score=score, round=round_id, stale=0 if meaningful else best[name]["stale"]+1)
                        trainer.save(paths[name] / "best.pt", export=True)
                    else:
                        best[name]["stale"] += 1
                    trainer.save(paths[name] / "latest.pt")
                    print(json.dumps(dict(stage="training", algorithm=name, seed=seed, round=round_id,
                                          games=trainer.games, decisions=trainer.decisions, updates=trainer.updates,
                                          validation=score, lr=trainer.optimizer.param_groups[0]["lr"])), flush=True)
                completed_round = round_id
                if args.max_decisions and trainers["adaptive-ppo"].decisions >= args.max_decisions:
                    stop_reason = "decision_budget"
                    break
                if round_id >= args.min_rounds and best["adaptive-ppo"]["stale"] >= args.patience:
                    stop_reason = "validation_plateau"
                    break
            actors = {}
            for name, trainer in trainers.items():
                trainer.save(paths[name] / "policy.pt", export=True)
                actors[name] = checkpoint_policies(paths[name] / "best.pt", "cpu", name)[0]
                runs.append(dict(algorithm=name, seed=seed, games=trainer.games, decisions=trainer.decisions,
                                 updates=trainer.updates, waves=trainer.waves, rounds=completed_round,
                                 training_seconds=trainer.training_seconds, stop_reason=stop_reason,
                                 best_round=best[name]["round"], best_validation=best[name]["score"],
                                 config=trainer.config.to_dict()))
            opponents = [Policy("random"), Policy("heuristic", kind="heuristic"), *frozen]
            for name, actor in actors.items():
                for opponent in opponents + ([actors["ppo"]] if name == "adaptive-ppo" else []):
                    m = evaluate_match(actor, opponent, args.test_games, config.num_envs,
                                       config.workers, protocol["test_seed"])
                    m.update(algorithm=name, training_seed=seed)
                    matches.append(m)
                    event(dict(kind="test", **m))
            if not latency:
                batch, _ = trainers["adaptive-ppo"].collect(min(config.num_envs, 128))
                latency = measure_latency(actors["adaptive-ppo"].model, batch["states"], batch["masks"],
                                          (1, config.num_envs), args.latency_repeats, config.torch_threads)
            del trainers, actors
    summary = []
    for algorithm, opponent in sorted({(m["algorithm"], m["opponent"]) for m in matches}):
        values = [m["win_rate"] for m in matches if (m["algorithm"], m["opponent"]) == (algorithm, opponent)]
        summary.append(dict(algorithm=algorithm, opponent=opponent, **seed_summary(values, (0, 1))))
    data = dict(protocol=protocol, config=config.to_dict(), calibration=calibration, curves=curves,
                training=training, runs=runs, matches=matches, summary=summary, latency=latency,
                wall_seconds=perf_counter()-started)
    render(output, data)
    print(json.dumps(dict(stage="complete", report=str((output / "report.html").resolve()),
                          wall_seconds=data["wall_seconds"], summary=summary)), flush=True)
    return data


def main(argv=None):
    args = parser().parse_args(argv)
    if args.render_only:
        render(args.render_only.parent, json.loads(args.render_only.read_text(encoding="utf-8")))
        return
    run(args)


if __name__ == "__main__":
    main()
