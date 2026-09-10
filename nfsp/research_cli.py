import json
from pathlib import Path
import torch
from .arena import Policy, checkpoint_policies, evaluate_match
from .experiments import run_comparison
from .performance import run_performance
from .ppo import PPOConfig, PPOTrainer
from .reports import heatmap, percent_interval, table, write_csv, write_report
from .trainer import resolve_device


def register(sub):
    audit = sub.add_parser("feature-audit", help="measure feature sparsity and redundancy without changing observations")
    audit.add_argument("--games", type=int, default=4096)
    audit.add_argument("--num-envs", type=int, default=256)
    audit.add_argument("--workers", type=int, default=1)
    audit.add_argument("--seed", type=int, default=70)
    audit.add_argument("--output", type=Path, default=Path("runs/feature_audit"))
    train = sub.add_parser("ppo-train", help="High-Entropy PPO self-play or standard PPO ablation")
    train.add_argument("--variant", choices=("he-ppo", "ppo"), default="he-ppo")
    train.add_argument("--config", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--games", type=int, default=100_000)
    train.add_argument("--output", type=Path, default=Path("runs/he-ppo"))
    train.add_argument("--device")
    train.add_argument("--workers", type=int)
    train.add_argument("--num-envs", type=int)
    train.add_argument("--save-every", type=int, default=10_000)
    train.add_argument("--eval-every", type=int, default=10_000)
    train.add_argument("--eval-games", type=int, default=1000)

    arena = sub.add_parser("arena", help="paired-seat round-robin of mixed algorithm checkpoints and bots")
    arena.add_argument("checkpoints", type=Path, nargs="*")
    arena.add_argument("--baselines", default="random,heuristic")
    arena.add_argument("--games", type=int, default=2000)
    arena.add_argument("--seed", type=int, default=1_000_042)
    arena.add_argument("--device", default="auto")
    arena.add_argument("--workers", type=int, default=1)
    arena.add_argument("--num-envs", type=int, default=256)
    arena.add_argument("--output", type=Path, default=Path("runs/arena"))

    compare = sub.add_parser("compare", help="train/evaluate algorithms at matched game budgets")
    compare.add_argument("--algorithms", default="he-ppo,ppo,nfsp,dqn")
    compare.add_argument("--baselines", default="random,heuristic")
    compare.add_argument("--baseline-checkpoint", action="append", default=[])
    compare.add_argument("--seeds", default="42,43,44")
    compare.add_argument("--eval-seed", type=int, default=2_000_042)
    compare.add_argument("--games", type=int, default=100_000)
    compare.add_argument("--eval-every", type=int, default=10_000)
    compare.add_argument("--eval-games", type=int, default=2000)
    compare.add_argument("--device", default="auto")
    compare.add_argument("--workers", type=int, default=1)
    compare.add_argument("--num-envs", type=int, default=256)
    compare.add_argument("--torch-threads", type=int, default=4)
    compare.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    compare.add_argument("--replay-capacity", type=int, default=100_000)
    compare.add_argument("--no-cross-play", action="store_true")
    compare.add_argument("--output", type=Path, default=Path("runs/comparison"))

    bench = sub.add_parser("performance", help="controlled parallel benchmark with offline HTML visualization")
    bench.add_argument("--games", type=int, default=1024)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--env-counts", default="1,64,256")
    bench.add_argument("--worker-counts", default="1,4,8")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--torch-threads", type=int, default=4)
    bench.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--no-training", action="store_true")
    bench.add_argument("--output", type=Path, default=Path("runs/performance"))


def dispatch(args):
    if args.command == "feature-audit":
        from .feature_audit import audit_features
        audit_features(args.games, args.num_envs, args.workers, args.seed, args.output)
    elif args.command == "ppo-train":
        if args.games <= 0:
            raise ValueError("games must be positive")
        if args.resume:
            if args.config or args.num_envs is not None:
                raise ValueError("resume restores configuration; only device/workers may change")
            trainer = PPOTrainer.load(args.resume, args.device, args.workers)
        else:
            if (args.output / "latest.pt").exists():
                raise FileExistsError("output checkpoint exists; use --resume or a new directory")
            config = PPOConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else PPOConfig()
            if args.variant == "ppo":
                config.adaptive_entropy, config.entropy_coef = False, .01
            for key in ("device", "workers", "num_envs"):
                if getattr(args, key) is not None:
                    setattr(config, key, getattr(args, key))
            trainer = PPOTrainer(config)
        print(json.dumps(trainer.run(args.games, args.output, args.save_every, args.eval_every, args.eval_games)))
    elif args.command == "compare":
        run_comparison(args)
    elif args.command == "performance":
        run_performance(args)
    elif args.command == "arena":
        device = resolve_device(args.device)
        torch.set_num_threads(4)
        names = [n for n in args.baselines.split(",") if n]
        if not set(names) <= {"random", "heuristic"}:
            raise ValueError("unknown baseline")
        policies = [Policy(n, kind=n) for n in names]
        for i, path in enumerate(args.checkpoints):
            policies.extend(checkpoint_policies(path, device, f"{path.parent.name}:{path.stem}:{i}"))
        if len(policies) < 2 or len({p.name for p in policies}) != len(policies):
            raise ValueError("at least two distinct policies required")
        matches = []
        for i, policy in enumerate(policies):
            for opponent in policies[i+1:]:
                match = evaluate_match(policy, opponent, args.games, args.num_envs, args.workers, args.seed)
                matches.append(match)
                print(json.dumps({k: match[k] for k in ("policy", "opponent", "win_rate", "mean_game_point_difference")}), flush=True)
        body = '<p>相同牌组交换先后手。PPO/NFSP 按分布采样，DQN 贪心执行；单次检查点评估不代表多种子训练稳定性。</p>'
        body += heatmap([p.name for p in policies], matches)
        body += table(["策略", "对手", "胜率", "95% 成对 bootstrap 区间", "游戏点差", "先手 / 后手胜率"],
                      [[m["policy"], m["opponent"], f'{m["win_rate"]:.2%}', percent_interval(m["win_uncertainty"]["ci95_bootstrap"]),
                        f'{m["mean_game_point_difference"]:.3f}', m["seat_win_rates"]] for m in matches])
        write_report(args.output, dict(policies=[dict(name=p.name, kind=p.kind, **p.metadata) for p in policies], matches=matches),
                     "Schnapsen · 检查点对战", body)
        write_csv(args.output, [{k: m[k] for k in ("policy", "opponent", "games", "wins", "win_rate", "mean_game_point_difference", "seconds")} for m in matches])
    else:
        return False
    return True
