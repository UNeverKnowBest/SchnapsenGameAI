"""Budget-matched training curves and cross-play, with training-seed uncertainty."""
import json
from pathlib import Path
import platform

import numpy as np
import torch

from .arena import Policy, checkpoint_policies, evaluate_match
from .config import Config
from .learner import Learner
from .ppo import PPOConfig, PPOTrainer
from .reports import bars, escape, heatmap, percent_interval, table, write_csv, write_report
from .trainer import Trainer, atomic_save, resolve_device
from ._native import FEATURE_VERSION


class DQNLearner(Learner):
    def ingest(self, rl, sl):
        # No average-policy learning or reservoir writes in the DQN baseline.
        super().ingest(rl, {key: value[:0] for key, value in sl.items()})


class DQNTrainer(Trainer):
    learner_type = DQNLearner
    algorithm = "dqn"

    def __init__(self, config):
        if config.eta != 1:
            raise ValueError("DQN self-play requires eta=1")
        super().__init__(config)

    def export(self, path):
        atomic_save(dict(algorithm="dqn", version=1, feature_version=FEATURE_VERSION,
                         config=self.config.to_dict(), games=self.games,
                         policies=[a.q.state_dict() for a in self.learners]), path)


def policies_for(trainer, name):
    if isinstance(trainer, PPOTrainer):
        return [Policy(name, trainer.model, trainer.device, "ppo")]
    dqn = isinstance(trainer, DQNTrainer)
    return [Policy(f"{name}/p{i}", a.q if dqn else a.policy, trainer.device, "dqn" if dqn else "nfsp")
            for i, a in enumerate(trainer.learners)]


def seed_summary(values, bounds=None):
    """Student-t interval across independently trained seeds, never deck pseudo-replication."""
    values = np.asarray(values, float)
    n, mean = len(values), float(values.mean())
    if n < 2:
        return dict(mean=mean, seeds=n, std=None, ci95=None)
    # Exact tabulated 97.5% critical values; conservative nearest lower df above 10.
    critical = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447,
                7:2.365, 8:2.306, 9:2.262, 10:2.228, 20:2.086, 30:2.042,
                60:2.000, 120:1.980}
    df = max(k for k in critical if k <= n-1)
    std = float(values.std(ddof=1))
    radius = critical[df] * std / np.sqrt(n)
    ci = [mean-radius, mean+radius]
    if bounds:
        ci = [max(bounds[0], ci[0]), min(bounds[1], ci[1])]
    return dict(mean=mean, seeds=n, std=std, ci95=ci)


def learning_chart(curves, baseline, axis="games"):
    series = {}
    for row in curves:
        if row["baseline"] == baseline:
            series.setdefault((row["algorithm"], row[axis]), []).append(row["win_rate"])
    names = sorted({k[0] for k in series})
    max_x = max(k[1] for k in series) if series else 1
    palette = ["#55cad2", "#ffa366", "#ae9cff", "#e97baa", "#bcd75e"]
    svg = '<svg viewBox="0 0 1100 360" role="img" aria-label="learning curves"><path d="M60 20V300H1060" stroke="#8295b0" fill="none"/>'
    for value in (0, .25, .5, .75, 1):
        y = 300-value*250
        svg += f'<path d="M60 {y}H1060" stroke="#34435d"/><text x="5" y="{y}" fill="#ddd">{value:.0%}</text>'
    for index, name in enumerate(names):
        points = sorted((x, float(np.mean(vals))) for (a, x), vals in series.items() if a == name)
        color = palette[index % len(palette)]
        xy = ' '.join(f'{60+x/max(max_x,1e-9)*970:.1f},{300-y*250:.1f}' for x, y in points)
        svg += f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="3"><title>{escape(name)}</title></polyline>'
        for x, y in points:
            svg += f'<circle cx="{60+x/max(max_x,1e-9)*970}" cy="{300-y*250}" r="4" fill="{color}"><title>{escape(name)}: {x:.1f}, {y:.2%}</title></circle>'
        svg += f'<text x="{70+index*225}" y="345" fill="{color}">{escape(name)}</text>'
    svg += f'<text x="70" y="325" fill="#ddd">0</text><text x="800" y="325" fill="#ddd">{max_x:.1f} {escape(axis)}</text></svg>'
    return f'<div class="card"><h2>学习曲线 · {escape(baseline)} · {escape(axis)}</h2>{svg}</div>'


def run_comparison(args):
    algorithms = args.algorithms.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    baseline_names = args.baselines.split(",")
    if (not algorithms or len(set(algorithms)) != len(algorithms)
            or not set(algorithms) <= {"he-ppo", "ppo", "nfsp", "dqn"}):
        raise ValueError("unique algorithms must be selected from he-ppo,ppo,nfsp,dqn")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**63 for s in seeds):
        raise ValueError("training seeds must be distinct and in [0, 2**63)")
    if not set(baseline_names) <= {"random", "heuristic"} or len(set(baseline_names)) != len(baseline_names):
        raise ValueError("baselines must be unique random/heuristic names")
    if (args.games <= 0 or args.eval_every <= 0 or args.eval_games <= 0 or args.eval_games % 2
            or args.num_envs <= 0 or args.workers <= 0):
        raise ValueError("positive budgets/interval/envs/workers and even eval-games required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("comparison output must be empty; choose a new directory")
    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    baselines = [Policy(n, kind=n) for n in baseline_names]
    for path in args.baseline_checkpoint:
        baselines.extend(checkpoint_policies(path, device, f"frozen:{Path(path).stem}"))
    if len({p.name for p in baselines}) != len(baselines):
        raise ValueError("baseline names must be unique")
    # Save protocol BEFORE training so budgets/seeds are inspectable, not selected post hoc.
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(feature_version=FEATURE_VERSION, python=platform.python_version(), torch=torch.__version__,
                    device=str(device), frozen_baselines=[p.metadata for p in baselines if p.metadata],
                    evaluation="paired decks, sampled PPO/NFSP, greedy DQN; both NFSP/DQN policies averaged",
                    limits="Finite baseline performance is not exploitability/NashConv or a convergence proof.")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    curves, training, finals, cross = [], [], [], []
    with (output / "events.jsonl").open("w", encoding="utf-8") as events:
        for seed in seeds:
            final_policies = []
            for algorithm in algorithms:
                shared = dict(seed=seed, device=str(device), num_envs=args.num_envs,
                              workers=args.workers, torch_threads=args.torch_threads, hidden=tuple(args.hidden))
                if algorithm in ("he-ppo", "ppo"):
                    trainer = PPOTrainer(PPOConfig(**shared, adaptive_entropy=algorithm == "he-ppo",
                                                  entropy_coef=.05 if algorithm == "he-ppo" else .01))
                else:
                    config = Config(**shared, eta=1. if algorithm == "dqn" else .1,
                                    replay_capacity=args.replay_capacity, reservoir_capacity=args.replay_capacity,
                                    rl_warmup=512, sl_warmup=512, batch_size=256)
                    trainer = (DQNTrainer if algorithm == "dqn" else Trainer)(config)
                run = output / f"{algorithm}_seed{seed}"
                run.mkdir()
                targets = sorted({0, args.games, *range(args.eval_every, args.games, args.eval_every)})
                for target in targets:
                    while trainer.games < target:
                        metrics = trainer.train_wave(min(args.num_envs, target-trainer.games))
                        event = dict(kind="train", algorithm=algorithm, seed=seed, **metrics)
                        events.write(json.dumps(event) + "\n")
                        training.append(event)
                    actors = policies_for(trainer, algorithm)
                    for baseline in baselines:
                        # Different held-out seeds at each checkpoint; final arena has a third stream.
                        matches = [evaluate_match(p, baseline, args.eval_games, args.num_envs, args.workers,
                                                  args.eval_seed + target * 101) for p in actors]
                        row = dict(algorithm=algorithm, training_seed=seed, baseline=baseline.name,
                                   games=target, decisions=trainer.decisions, seconds=trainer.training_seconds,
                                   win_rate=float(np.mean([m["win_rate"] for m in matches])),
                                   mean_game_point_difference=float(np.mean([m["mean_game_point_difference"] for m in matches])),
                                   normalized_entropy=float(np.mean([m["normalized_entropy"] or 0 for m in matches])),
                                   matches=matches)
                        curves.append(row)
                        events.write(json.dumps(dict(kind="evaluation", **row)) + "\n")
                    events.flush()
                    print(json.dumps(dict(algorithm=algorithm, seed=seed, games=target,
                                          seconds=round(trainer.training_seconds, 2))), flush=True)
                trainer.save(run / "latest.pt")
                if isinstance(trainer, PPOTrainer):
                    trainer.save(run / "policy.pt", export=True)
                else:
                    trainer.export(run / "policy.pt")
                finals.append(dict(algorithm=algorithm, seed=seed, games=trainer.games,
                                   decisions=trainer.decisions, seconds=trainer.training_seconds,
                                   config=trainer.config.to_dict(),
                                   updates=trainer.updates if isinstance(trainer, PPOTrainer) else
                                   [{"rl": a.rl_updates, "sl": a.sl_updates} for a in trainer.learners]))
                final_policies.extend(actors)
                del trainer
            if not args.no_cross_play:
                entrants = baselines + final_policies
                for i, policy in enumerate(entrants):
                    for opponent in entrants[i+1:]:
                        match = evaluate_match(policy, opponent, args.eval_games, args.num_envs,
                                               args.workers, args.eval_seed + 50_000_003)
                        cross.append(dict(training_seed=seed, **match))
                        events.write(json.dumps(dict(kind="cross_play", **cross[-1])) + "\n")
                events.flush()
    summary = []
    for algorithm in algorithms:
        for baseline in baselines:
            rows = [r for r in curves if r["algorithm"] == algorithm and r["baseline"] == baseline.name and r["games"] == args.games]
            summary.append(dict(algorithm=algorithm, baseline=baseline.name,
                                win=seed_summary([r["win_rate"] for r in rows], (0, 1)),
                                points=seed_summary([r["mean_game_point_difference"] for r in rows], (-3, 3)),
                                entropy=seed_summary([r["normalized_entropy"] for r in rows], (0, 1))))
    deltas = []
    if {"he-ppo", "ppo"} <= set(algorithms):
        for baseline in baselines:
            values = {(r["algorithm"], r["training_seed"]): r["win_rate"] for r in curves
                      if r["baseline"] == baseline.name and r["games"] == args.games}
            deltas.append(dict(baseline=baseline.name,
                               **seed_summary([values["he-ppo", s]-values["ppo", s] for s in seeds], (-1, 1))))
    data = dict(protocol=protocol, summary=summary, he_ppo_minus_ppo=deltas, curves=curves,
                training=training, final_runs=finals, cross_play=cross)
    render_comparison(output, data)
    return data


def render_comparison(output, data):
    summary, curves, cross = data["summary"], data["curves"], data["cross_play"]
    deltas = data["he_ppo_minus_ppo"]
    algorithms = data["protocol"]["algorithms"].split(",")
    baselines = [Policy(name) for name in dict.fromkeys(r["baseline"] for r in summary)]
    body = '<p>相同训练局数预算；所有算法使用同一 465 维公开观测和 28 维合法动作空间。胜率同时报告先后手、牌组不确定性与训练种子差异。可在下拉框切换 baseline。</p>'
    body += '<p>这些结果只描述本次预算下的表现。不同算法的更新量与参数总数不同；多重比较未经显著性校正。高熵自博弈没有纳什收敛保证。</p>'
    body += '<label>Baseline <select onchange="document.querySelectorAll(\'[data-baseline]\').forEach(e=>e.hidden=e.dataset.baseline!==this.value)">'
    body += ''.join(f'<option value="{i}">{escape(p.name)}</option>' for i, p in enumerate(baselines)) + '</select></label>'
    for i, baseline in enumerate(baselines):
        rows = [r for r in summary if r["baseline"] == baseline.name]
        body += f'<section data-baseline="{i}" {"hidden" if i else ""}>'
        body += bars(f'最终胜率 vs {baseline.name}（95% 种子 t 区间）', [r["algorithm"] for r in rows],
                     [100*r["win"]["mean"] for r in rows], "%",
                     [[100*x for x in (r["win"]["ci95"] or [r["win"]["mean"]]*2)] for r in rows])
        body += table(["算法", "胜率", "种子数", "平均游戏点差", "归一化熵"],
                      [[r["algorithm"], f'{r["win"]["mean"]:.2%}', r["win"]["seeds"],
                        f'{r["points"]["mean"]:.3f}', f'{r["entropy"]["mean"]:.3f}'] for r in rows])
        body += learning_chart(curves, baseline.name)
        # Wall-clock points use mean time per algorithm/checkpoint across seeds.
        clock_rows = []
        for algorithm in algorithms:
            for target in sorted({r["games"] for r in curves}):
                group = [r for r in curves if r["algorithm"] == algorithm and r["games"] == target and r["baseline"] == baseline.name]
                clock_rows.append(dict(algorithm=algorithm, baseline=baseline.name,
                                       seconds=float(np.mean([r["seconds"] for r in group])),
                                       win_rate=float(np.mean([r["win_rate"] for r in group]))))
        body += learning_chart(clock_rows, baseline.name, "seconds") + '</section>'
    if deltas:
        body += table(["HE-PPO − PPO", "胜率差", "95% 配对种子 t 区间"],
                      [[r["baseline"], f'{r["mean"]:+.2%}', percent_interval(r["ci95"])] for r in deltas])
    if cross:
        names = list(dict.fromkeys([r["policy"] for r in cross] + [r["opponent"] for r in cross]))
        aggregate = []
        for a in names:
            for b in names:
                values = [r["win_rate"] for r in cross if r["policy"] == a and r["opponent"] == b]
                if values:
                    aggregate.append(dict(policy=a, opponent=b, win_rate=float(np.mean(values))))
        body += heatmap(names, aggregate)
    body += '<p>单一种子不能估计训练稳定性；t 区间在少量种子下也很不稳定。JSON 保留每副牌的成对结果、bootstrap 区间和非退化 Hoeffding 区间。学习曲线未用于选择最佳测试检查点。</p>'
    write_report(output, data, "Schnapsen · 算法对比实验", body)
    write_csv(output, [{k: v for k, v in r.items() if k != "matches"} for r in curves])
    return data
