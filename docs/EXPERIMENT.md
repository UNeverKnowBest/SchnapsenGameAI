# 一键实验

更新：三轴学习曲线、逐点跨种子置信区间、实时页面、平台期分析及推荐轮数见 [学习曲线与训练预算](LEARNING_CURVES.md)。

使用项目虚拟环境运行；系统 Python 不一定装有相同的 CUDA PyTorch 和 Rust 扩展。

```powershell
# 小 demo：自动测 CPU/GPU，训练两种 PPO，评估并生成图表
.venv/Scripts/python.exe experiment.py --demo

# 正式实验：3 个 seed，每个算法每 seed 最多 100,000 局
.venv/Scripts/python.exe experiment.py

# 明确指定预算；达到 decision 上限后完成当前 wave
.venv/Scripts/python.exe experiment.py --rounds 12 --games-per-round 5000 --max-decisions 1000000 --output runs/my_experiment

# 快速使用保守 CPU 配置，不执行训练性能调优
.venv/Scripts/python.exe experiment.py --demo --skip-tuning --device cpu --num-envs 256

# 添加已冻结的 PPO/NFSP/DQN 检查点作为额外对手
.venv/Scripts/python.exe experiment.py --baseline-checkpoint runs/old/policy.pt

# 仅重新生成图表，无需重新训练
.venv/Scripts/python.exe experiment.py --render-only runs/my_experiment/results.json
```

初次安装见 [NFSP.md](NFSP.md)；Windows 构建扩展可运行 `scripts/build_native.ps1`。入口只依赖现有 NumPy、PyTorch 和 Rust 扩展，SVG 无需 matplotlib 或网络。导入 `experiment` 不会启动训练。已有输出目录非空时拒绝覆盖，默认目录带时间戳。`latest.pt` 可通过原有 `python -m nfsp ppo-train --resume ...` 继续训练；一键实验本身目前不支持中断后续跑整个对比流程。

## 默认预算与调整

| 项目 | 正式实验 | `--demo` |
| --- | --- | --- |
| 独立训练 seed | 42、43、44 | 42 |
| 最大评估轮次 | 20 | 4 |
| 每轮完整对局 | 5,000 | 512 |
| 每 wave 最大优化 epochs | 4 | 4 |
| 每 baseline 验证对局 | 1,000 | 256 |
| 每 baseline 最终测试对局 | 4,000 | 512 |
| 硬件计时重复次数 | 3 | 3 |
| 单种子每算法最大训练局数 | 100,000 | 2,048 |

默认正式实验共训练主算法与标准 PPO 各 3 次，最多 600,000 局。训练 seed 不允许使用预留的验证/测试 seed 2000042、7000042。每局包含多个动作，实际 decisions 和 optimizer steps 写入报告，不能把 games 当训练步数。`--rounds` 指评估轮次，`--epochs` 指同一批样本优化的最大遍数。每个 wave 采集完整对局、固定行为策略并沿同一玩家轨迹计算 GAE。可用 `--max-decisions` 限制主算法环境动作预算；最多超出一个完整 wave，随后标准 PPO 补齐相同训练局数。两种策略的轨迹长度和 KL 提前停止不同，梯度步数并非严格相等。

这是一组有上限的起始预算，不是收敛保证。至少训练 8 轮后，主算法连续 5 轮没有至少 0.5 个百分点的验证成绩改善则停止两种算法；`--min-rounds`、`--patience` 可改。最佳检查点按自身验证集对随机/规则策略胜率的均值选取，包含未训练的第 0 轮；不强行把最后模型当最好模型。

主算法从 lr=3e-4 开始，按每 wave 更新后的实际 KL 调节：超过 target_kl=0.02 或出现 KL 提前停止则除以 1.5，低于目标一半则乘以 1.1，并限制在 [1e-5, 1e-3]。各 minibatch 保留 KL 提前停止和梯度裁剪。合法动作归一化熵目标为 0.8，熵系数在 [0.001, 0.5] 内自适应，强制单动作状态排除。学习率、实际 KL、熵、loss、梯度更新次数均落盘。

## 设备和效率

校准对 CPU 的 1/4 个 PyTorch 线程、1/4 个 Rust 环境线程、不同环境批量，以及 CUDA 采样+更新、CPU 采样+CUDA 更新做有限候选比较；候选上限由本机逻辑核心与 `--num-envs` 控制。默认候选批量 256/1024，demo 为 128/256。每候选预热后从相同初始化重新训练，随机测量顺序，按重复计时的完整训练吞吐量中位数选配置。接近最快配置 5% 内时优先 CPU 和较少线程。强制 `--device cpu/cuda` 限制最终选择，仍保留其他可用设备的测量结果。无 CUDA 时明确记录不可用并继续 CPU。

校准预算是最大候选批量的两倍局数，各候选 epochs 相同；批量改变 wave 数量，可能改变实际优化次数，因此校准图是配置比较，不把全部差异归因于硬件。`calibration.jsonl` 保存逐次 games、decisions、optimizer steps、采样/环境/更新耗时及 GPU 显存峰值。CUDA OOM 候选剔除，其他错误直接暴露。小校准不能保证长时间温度或系统负载改变后的性能。

PPO 采样将概率、log probability、value 合并一次传回 CPU，并在 CPU 采样动作，避免动作再次送 GPU 获取 log probability。更新只传必要张量，减少逐 minibatch 指标同步。可选的 CPU actor 每 wave 从最新 GPU 模型同步，行为概率和采样权重一致。

训练完成后用同一个最佳模型和同一组观测在 CPU/CUDA 测 latency，包含 CPU 输入→模型→合法动作概率→CPU 动作的真实调用路径。每种配置预热 10 次，测量前显式同步 GPU，随后测量 100 次；每个请求通过返回 CPU 数组的同步拷贝确保推理完成，计时中不叠加多余 CUDA barrier，输出 P50/P95/P99、actions/s 和原始样本。batch=1 是单次请求；批量总耗时除以动作数只是摊销成本，不是用户请求 latency。不包括 HTTP、排队或整局时间；这是本机策略能力测量，不是线上服务 SLA。

## Baseline 与测试隔离

- 均匀随机合法动作策略。
- 只用公开信息的规则策略：换将、报婚、用廉价牌赢墩、保留高点牌。
- 标准 PPO：相同网络、游戏预算和最大 epochs，固定 lr=3e-4、entropy=0.01。它是主算法自适应学习率与熵的组合消融，不能单独归因于其中一个改动。
- 可通过 `--baseline-checkpoint` 添加已有 DQN、NFSP 或 PPO 导出；记录路径、SHA-256 和训练预算。

保留当前 `nfsp/experiments.py` 中的 DQN/NFSP 实现用于额外研究对照，原有 `python -m nfsp compare` 仍可用。一键入口默认两种 PPO，避免默认再训练不必要的算法。

训练、验证、最终测试使用独立的 seed 流；不同训练 seed 共用相同验证/测试牌组便于比较。每副牌交换座位，记录分座位胜率、游戏点差、成对 bootstrap 95% 区间和保守 Hoeffding 区间。最终测试只运行一次，不用于超参数调整或选最佳 checkpoint。报告跨训练 seed 的区间使用 t 区间；一个 seed 显示无跨训练 seed 区间。多次复用验证集本身可能导致选择偏差，因此最终测试独立。有限 baseline 胜率不代表 Nash 收敛或 exploitability。

## 输出

`report.html` 直接给出本次 CUDA 相对 CPU 调优配置是否更慢、各 batch 的低延迟设备选择；可直接离线打开，所有图也单独保存为 `figures/*.svg`，包括学习曲线、baseline 胜率、lr、熵、KL、value loss、训练吞吐量、校准和 latency。SVG 可以直接用于文档或进一步转换成 PNG/PDF。

`protocol.json` 在训练前保存预算与评估协议，`config.json` 保存硬件选择，`events.jsonl` 持续刷新过程记录。`results.json` 包含完整可重绘数据，另有 `results.csv`、`training.csv`、`learning.csv`、`latency.csv`。各算法/seed 目录包括最佳策略 `best.pt`、最终策略 `policy.pt` 和含优化器/RNG 的最终 `latest.pt`。

## 清理范围

删除无人引用、依赖旧 Python 环境的 `ActionRepresentation.py`、`DQN_bot.py`、`DQNMultithread.py`、`feacture.py`、`storage.py`、`common/utils.py`、`model/DQN.py`。历史内容仍可从 Git 获取。保留正在使用的 Rust 引擎、原生扩展、当前算法、虚拟环境和工具链；`compatibility/` 的独立旧引擎 oracle 是规则差分验证依据，保留其验证用途。
