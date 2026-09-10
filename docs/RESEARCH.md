# High-Entropy PPO、自博弈对比与工程基准

新功能复用已验证的 Rust `BatchEnv`，不改变规则或 v2 的 465 维公开观测。入口仍是 `python -m nfsp`。已有 NFSP 命令保持兼容。生成的 HTML 内嵌 SVG，无 CDN、绘图库或网络依赖，可直接打开；算法对比页面支持切换 baseline，图中数据点可悬停查看。

## 算法与定义

这里的 **High-Entropy PPO Self-Play** 是本项目明确定义的 PPO 变体：共享 actor-critic 自博弈、合法动作掩码、按合法动作数量归一化的熵奖励，以及有界自适应熵系数。它不是对某篇同名算法的未注明复现，也不宣称有纳什收敛保证。

依据是 [PPO 原始论文](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438) 和 [熵正则化分析](https://arxiv.org/abs/1811.11214)。归一化熵与控制器是本项目的工程选择，应通过消融验证。

令 `k(s)` 为合法动作数，`r=π_new(a|s)/π_old(a|s)`。本实现最小化：

```text
Lpolicy = -mean(min(r A, clip(r, 1-ε, 1+ε) A))
Lvalue  = 0.5 mean((V - return)²)
Hnorm   = H(π(.|s)) / log(k(s)), k(s)>1
L       = Lpolicy + value_coef Lvalue - α mean_choice_states(Hnorm)
α_next  = clamp(α exp(entropy_lr (target_entropy - Hnorm_final)), α_min, α_max)
```

默认 `ε=.2`、`γ=1`、`λ=.95`、Adam `3e-4`、4 epochs、512 minibatch、梯度裁剪 `.5`、KL 阈值 `.02`。actor-critic 为 `465→256→128→64` Tanh 共享主干，两个输出头分别为 28 logits 和 1 value。默认熵系数 `.05`、目标归一化熵 `.8`、更新率 `.05`、上下限 `.001/.5`。每波完整更新后调节一次 α。

只有一个合法动作时熵为零，并排除出熵目标的分母。非法动作概率严格为零；计算熵前将非法 log probability 置零，避免强制行动时 float32 梯度溢出。归一化熵 1 代表合法动作上的均匀分布；高熵不是高胜率的替代指标，也不要求终局强制行动保持探索。

完整一波自博弈期间权重固定。经验按 `(game, player)` 分组：GAE 沿同一玩家的决策链计算；对手行动跨越该转换，换将可使同一玩家连续行动。双方最后一次决策分别接收终局 `+1/-1`，terminal bootstrap 为零。`reward=game_points` 可改为实际游戏点乘胜负号再除以 3，改变了优化目标，不能与 win 奖励混为同一实验。

优势在整波样本上标准化；minibatch 多轮更新使用冻结的旧 log probability。当当前 minibatch 的近似 KL 超过阈值时，停止剩余更新。日志含 policy/value loss、归一化熵、熵系数、KL、clip fraction、梯度范数、解释方差、优化次数与分项耗时。数据只用一次 PPO 更新阶段，没有跨波 replay。

```powershell
# 高熵 PPO
.venv/Scripts/python.exe -m nfsp ppo-train --config configs/he_ppo.json --games 100000 --device cuda --output runs/he-ppo

# 普通 PPO 消融：同一实现，固定熵系数 .01
.venv/Scripts/python.exe -m nfsp ppo-train --variant ppo --games 100000 --device cuda --output runs/ppo

# 完整恢复模型、优化器、RNG、熵系数、计数；games 为绝对目标
.venv/Scripts/python.exe -m nfsp ppo-train --resume runs/he-ppo/latest.pt --games 200000 --output runs/he-ppo
```

检查点原子保存于完整波边界；`latest.pt` 可续训，`policy.pt` 为轻量导出。恢复仅允许更换 device/workers，不允许悄悄改变 batch 或特征版本。跨设备不承诺逐位一致。`--variant` 只用于新建训练，恢复以保存配置为准。完整文件含 Python/NumPy 类型，只加载可信本地检查点。

## 学术对比协议

```powershell
.venv/Scripts/python.exe -m nfsp compare --algorithms he-ppo,ppo,nfsp,dqn --baselines random,heuristic --seeds 42,43,44 --games 100000 --eval-every 10000 --eval-games 2000 --device cuda --output runs/comparison

# 加入冻结历史模型或外部训练模型作为额外 baseline
.venv/Scripts/python.exe -m nfsp compare --baseline-checkpoint runs/he-ppo/policy.pt --output runs/comparison_with_frozen

# 只评估已有模型，支持 NFSP/PPO/DQN 混合，NFSP/DQN 保留两个玩家策略
.venv/Scripts/python.exe -m nfsp arena runs/he-ppo/policy.pt runs/nfsp/policy.pt --baselines random,heuristic --games 2000 --output runs/arena
```

| 对比项 | 实验意义与边界 |
| --- | --- |
| Uniform random | 检查学习是否超过无知识合法行动；不能证明对强对手有效 |
| Public heuristic | 换将/报婚、跟牌时选最便宜的获胜牌，否则低代价出牌；只用公开特征，没有隐藏信息搜索 |
| DQN self-play | 与 NFSP 相同的 Double DQN、探索和 replay，`eta=1`；不写 SL reservoir、不训练平均策略；评估 Q 的合法贪心动作 |
| NFSP | `eta=.1`，独立 Q 和平均策略、全历史 reservoir；评估两个平均策略并等权平均，不挑表现好的玩家 |
| Ordinary PPO | 相同 PPO 代码、网络与奖励，固定归一化熵系数 `.01`；是受控低熵消融，不代表所有 PPO 调参方案 |
| High-Entropy PPO | 默认 `.05` 起始熵系数与自适应控制器；检验保留探索是否改善学习/稳定性 |
| Frozen checkpoint | 对历史策略的泛化与遗忘检查；固定文件 SHA256 和配置写入协议 |
| Cross-play matrix | 呈现策略间的克制和循环；不强行压成单一 Elo 分数 |

`compare` 在训练前保存 `protocol.json`，各算法使用相同游戏预算、隐藏层宽度、观测和动作空间。NFSP/DQN 使用 ReLU、多网络及 replay，PPO 使用 Tanh 共享 actor-critic；总参数量、梯度更新量和训练速度并不相同。记录实际决策数、完整配置、优化次数和纯训练秒数，同时展示胜率随局数和训练时间的曲线。按时间曲线只是资源效率参考，不是重新实施的等墙钟预算实验。

每个评估牌组执行两局，候选分别占初始先手和后手。牌组、候选与对手均冻结；动作采样用游戏编号与决策步数驱动的无状态 RNG，避免换线程/批次改变随机序列。浮点推理后端差异仍可能改变临界采样。评估不写训练缓冲区，也不消耗训练 RNG。不同检查点评估使用不同的独立测试种子；最终交叉对战使用第三个种子流。相同检查点/训练种子间共享测试牌组，便于配对比较。

输出指标：胜率、先/后手胜率、平均 Schnapsen 游戏点差（范围 -3 至 +3）、选择状态上的策略熵、训练样本量与耗时。**置信区间的单位必须区分**：

- 单次对战以“交换座位的牌组对”为独立样本，保存 pair scores、标准误差、成对 bootstrap 95% 区间；另给出 Hoeffding 有界区间，避免全胜/全败时 bootstrap 退化成假精确结论。
- 算法汇总以独立训练种子为样本，用 Student-t 区间。NFSP/DQN 两个玩家先在种子内平均，再跨种子统计。不能把数千副牌当成数千次独立训练。
- HE-PPO − PPO 报告相同训练种子配对的胜率差及 t 区间。区间覆盖 0 时，本次预算无法支持明确优势。多 baseline 比较没有做多重检验校正，应作为探索性结果。
- 三个种子只够初步观察；单个种子不给跨训练种子的区间。没有 exact best response，因此不报告伪造的 exploitability/NashConv，不将胜率称为均衡证明。可将未来近似最佳响应攻击器作为额外 baseline，但它也只能给下界证据。

完整实验输出包括 `events.jsonl`、`results.json`、CSV、HTML、每个算法/种子的完整快照和导出策略。输出目录需为空，避免混淆两次实验。长训练的各算法快照可独立恢复；当前 `compare` 调度器不支持恢复半途完成的整个实验套件。

## 工程性能协议

```powershell
.venv/Scripts/python.exe -m nfsp performance --games 1024 --repeats 3 --env-counts 1,64,256 --worker-counts 1,4,8 --device cuda --output runs/performance
```

三个层次分别测量：随机动作下的批量环境执行（包含 NumPy/Python 传输）、固定神经网络的完整采样、固定 rollout 大小的完整 PPO 更新。前两个层次分别比较单局单线程、批量单线程、相同批量多线程；第三个层次保持模型、批次、对局、更新预算不变，只改变 Rust workers。不会把更少的梯度更新当作并行加速。

每种配置先预热，计时前后同步 CUDA，至少重复两次，重复内随机打乱配置顺序。预热、检查点 I/O 和评估不计入耗时；计时/哈希轨迹记录本身有成本。报告均值、标准差、最小/最大吞吐、整体加速、同批次线程加速、线程效率、批次延迟 p50/p95、环境/推理/更新/其他耗时。完整训练还记录优化次数及 CUDA 峰值张量分配内存。

环境和冻结采样比对完整动作与结果的 SHA256；训练比较最终模型 SHA256。如不一致，保留报告并报错，明确不将该结果称为受控工作量加速。单机、单次系统负载与温度影响仍存在。小牌局的环境成本很低，线程池调度和数组传输可能超过计算收益；如实展示降速是工程分析的一部分。

这套基准比较当前 Rust 系统内部的串行/批量/线程配置，没有把旧 Python 引擎作为同负载参照。旧 `benchmark` 命令仍保留，用于 NFSP 端到端吞吐探索；其改变 wave 大小时会改变学习动态，适用边界见 NFSP 文档。

特征深入分析见 [FEATURES.md](FEATURES.md)，本机实测记录见 [RESEARCH_VALIDATION.md](RESEARCH_VALIDATION.md)。
