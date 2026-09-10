"""Budget-matched training curves and cross-play, with training-seed uncertainty."""
import json
from dataclasses import replace
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
from .learning import (seed_summary, learning_chart, learning_protocol, export_learning,
                       plateau_table, write_live)


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


def load_algorithm_configs(specifications, algorithms):
    configs = {}
    for specification in specifications:
        algorithm, separator, filename = specification.partition("=")
        if not separator or algorithm not in algorithms or algorithm in configs:
            raise ValueError("algorithm-config must be unique ALGORITHM=JSON for a selected algorithm")
        cls = PPOConfig if algorithm in ("ppo", "he-ppo") else Config
        config = cls(**json.loads(Path(filename).read_text(encoding="utf-8"))).validate()
        if algorithm == "dqn" and config.eta != 1:
            raise ValueError("DQN requires eta=1")
        if algorithm == "ppo" and (config.adaptive_entropy or config.adaptive_lr):
            raise ValueError("standard PPO requires fixed entropy and learning rate")
        if algorithm == "he-ppo" and not config.adaptive_entropy:
            raise ValueError("HE-PPO requires adaptive entropy")
        configs[algorithm] = config
    return configs


def run_comparison(args):
    analysis_protocol = learning_protocol(args)
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
            or args.test_games <= 0 or args.test_games % 2
            or args.num_envs <= 0 or args.workers <= 0):
        raise ValueError("positive budgets/interval/envs/workers and even eval-games required")
    algorithm_configs = load_algorithm_configs(getattr(args, "algorithm_config", []), algorithms)
    targets = sorted({0, args.games, *range(args.eval_every, args.games, args.eval_every)})
    test_seed = args.eval_seed + (args.games + 1) * 101
    cross_seed = test_seed + 1
    evaluation_seeds = {args.eval_seed + target * 101 for target in targets} | {test_seed, cross_seed}
    if args.eval_seed < 0 or cross_seed >= 2**63 or set(seeds) & evaluation_seeds:
        raise ValueError('training and evaluation seed streams must be distinct and in range')
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
    protocol.update(algorithm_configs={a: c.to_dict() for a, c in algorithm_configs.items()}, test_seed=test_seed, cross_seed=cross_seed, learning_analysis=analysis_protocol, feature_version=FEATURE_VERSION, python=platform.python_version(), torch=torch.__version__,
                    device=str(device), frozen_baselines=[p.metadata for p in baselines if p.metadata],
                    evaluation="paired decks, sampled PPO/NFSP, greedy DQN; both NFSP/DQN policies averaged",
                    limits="Finite baseline performance is not exploitability/NashConv or a convergence proof.")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    curves, training, finals, cross, final_tests = [], [], [], [], []
    write_live(output, curves, analysis_protocol)
    with (output / "events.jsonl").open("w", encoding="utf-8") as events:
        for seed in seeds:
            final_policies = []
            for algorithm in algorithms:
                shared = dict(seed=seed, device=str(device), num_envs=args.num_envs,
                              workers=args.workers, torch_threads=args.torch_threads, hidden=tuple(args.hidden))
                if algorithm in ("he-ppo", "ppo"):
                    config = (replace(algorithm_configs[algorithm], **shared) if algorithm in algorithm_configs else
                              PPOConfig(**shared, adaptive_entropy=algorithm == "he-ppo",
                                        entropy_coef=.05 if algorithm == "he-ppo" else .01))
                    trainer = PPOTrainer(config)
                else:
                    config = Config(**shared, eta=1. if algorithm == "dqn" else .1,
                                    replay_capacity=args.replay_capacity, reservoir_capacity=args.replay_capacity,
                                    rl_warmup=512, sl_warmup=512, batch_size=256)
                    if algorithm in algorithm_configs:
                        config = replace(algorithm_configs[algorithm], **shared)
                    trainer = (DQNTrainer if algorithm == "dqn" else Trainer)(config)
                run = output / f"{algorithm}_seed{seed}"
                run.mkdir()
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
                    if isinstance(trainer, PPOTrainer):
                        trainer.save(run / f"policy_{target}.pt", export=True)
                    else:
                        trainer.export(run / f"policy_{target}.pt")
                    events.flush()
                    write_live(output, curves, analysis_protocol)
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
                for baseline in baselines:
                    matches = [evaluate_match(p, baseline, args.test_games, args.num_envs, args.workers,
                                              test_seed) for p in actors]
                    row = dict(algorithm=algorithm, training_seed=seed, baseline=baseline.name,
                               games=trainer.games, win_rate=float(np.mean([m["win_rate"] for m in matches])),
                               mean_game_point_difference=float(np.mean([m["mean_game_point_difference"] for m in matches])),
                               normalized_entropy=float(np.mean([m["normalized_entropy"] or 0 for m in matches])),
                               matches=matches)
                    final_tests.append(row)
                    events.write(json.dumps(dict(kind="final_test", **row)) + "\n")
                events.flush()
                final_policies.extend(actors)
                del trainer
            if not args.no_cross_play:
                entrants = baselines + final_policies
                for i, policy in enumerate(entrants):
                    for opponent in entrants[i+1:]:
                        match = evaluate_match(policy, opponent, args.test_games, args.num_envs,
                                               args.workers, cross_seed)
                        cross.append(dict(training_seed=seed, **match))
                        events.write(json.dumps(dict(kind="cross_play", **cross[-1])) + "\n")
                events.flush()
    summary = []
    for algorithm in algorithms:
        for baseline in baselines:
            rows = [r for r in final_tests if r["algorithm"] == algorithm and r["baseline"] == baseline.name]
            summary.append(dict(algorithm=algorithm, baseline=baseline.name,
                                win=seed_summary([r["win_rate"] for r in rows], (0, 1)),
                                points=seed_summary([r["mean_game_point_difference"] for r in rows], (-3, 3)),
                                entropy=seed_summary([r["normalized_entropy"] for r in rows], (0, 1))))
    deltas = []
    if {"he-ppo", "ppo"} <= set(algorithms):
        for baseline in baselines:
            values = {(r["algorithm"], r["training_seed"]): r["win_rate"] for r in final_tests
                      if r["baseline"] == baseline.name}
            deltas.append(dict(baseline=baseline.name,
                               **seed_summary([values["he-ppo", s]-values["ppo", s] for s in seeds], (-1, 1))))
    data = dict(protocol=protocol, summary=summary, he_ppo_minus_ppo=deltas, curves=curves,
                training=training, final_runs=finals, cross_play=cross, final_tests=final_tests)
    render_comparison(output, data)
    return data


def render_comparison(output, data):
    summary, curves, cross = data["summary"], data["curves"], data["cross_play"]
    deltas = data["he_ppo_minus_ppo"]
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
        body += learning_chart(curves, baseline.name, "decisions")
        body += learning_chart(curves, baseline.name, "seconds") + '</section>'
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
    if 'final_tests' in data:
        body += '<p>最终胜率使用独立测试种子评估预算末尾模型；训练曲线只用于观察学习过程。</p>'
    analysis_protocol = data["protocol"].get("learning_analysis", dict(window=5, delta=.02, min_rounds=10))
    data["plateau"] = export_learning(output, curves, analysis_protocol)
    body += plateau_table(data["plateau"], analysis_protocol)
    body += '<p>原始统计：learning_statistics.csv / learning_statistics.json；平台期协议和判断：plateau.json；可导出图表：figures/learning_编号_轴.svg（baseline 按名称排序）。</p>'
    write_report(output, data, "Schnapsen · 算法对比实验", body)
    write_live(output, curves, analysis_protocol, complete=True)
    write_csv(output, [{k: v for k, v in r.items() if k != "matches"} for r in curves])
    return data
