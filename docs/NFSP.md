# Rust 批量 NFSP

训练实现位于 `nfsp/`，Rust 绑定位于 `native/`。根目录 `main.py` 是命令入口。未使用的历史 Python DQN、特征、缓冲区实现已删除；当前 NFSP/DQN 对比实现仍在 `nfsp/`。一键训练、基线评估与性能测试见 [EXPERIMENT.md](EXPERIMENT.md)。

## 算法定义

这是一套 NFSP 实现及可配置的 DQN 工程增强，不宣称是新的最先进算法。

依据：

- [Heinrich & Silver, 2016: NFSP](https://arxiv.org/abs/1603.01121)：每个玩家各自学习最佳响应 Q 网络和平均策略网络；每局选择行为模式，RL 使用所有行为经验，SL 使用最佳响应行为的全历史 reservoir。
- [OpenSpiel PyTorch NFSP](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/nfsp.py)（2026-09-10 查阅）：按局采样模式、合法动作约束、用最佳响应动作分布作为监督目标。此项目独立实现，没有引入 OpenSpiel 运行时依赖。
- [Double DQN](https://arxiv.org/abs/1509.06461)：可配置的 Q 学习增强，在线网络选择下一合法动作，目标网络估计其价值。

每个玩家维护 `Q / target_Q / average_policy`，以及各自的 RL 环形经验池和 SL reservoir。默认配置：

| 参数 | 默认值与含义 |
| --- | --- |
| eta | 0.1；每个玩家开局独立选择最佳响应模式，否则使用平均策略 |
| epsilon | 按该玩家累计决策数，从 0.2 线性降到 0.01，跨度 1,000,000 次决策 |
| reward / gamma | 终局胜负 +1/-1，其他为 0；gamma=1，对应最大化胜率 |
| 网络 | 两种网络均为独立 465→256→128→64→28 MLP；ReLU；策略输出 logits |
| RL | 默认 masked Double DQN、Huber loss、Adam 3e-4、梯度范数裁剪 5 |
| SL | 默认合法动作上的软目标交叉熵、Adam 3e-4 |
| batch / learn_every | 每批 256；每个玩家每新增 128 条 RL 经验，分别获得一次合格的 RL/SL 更新 |
| target_every | 每个玩家每 1,000 次 RL 优化更新，同步其目标网络 |
| warmup | RL 2,000 条；SL 1,000 条；必须同时满足 batch 大小 |
| 容量 | 每个玩家 RL 100,000，SL 200,000，数据在 CPU 内存 |
| 并行 | 默认 256 局，8 个常驻 Rust 工作线程；PyTorch CPU 线程默认 4 |

另提供 `configs/throughput.json`：batch=1024、learn_every=512、target_every=250。稳定阶段每条新 RL 经验对应的 RL/SL 抽样量仍各为 2，目标同步对应的新经验数量仍约为 128,000；但大批次减少优化器更新次数，会改变学习动态，不承诺等同的策略质量。warmup 与轮次取整也影响短程实际更新量。

最佳响应分布：合法动作共享 epsilon 概率质量，在线 Q 最大的合法动作额外获得 1-epsilon。只有最佳响应模式产生的决策进入 SL。软目标保存整个分布；`soft_sl_targets=false` 时保存实际采样动作的 one-hot 标签，与原始动作模仿形式对应。平均策略在合法动作 logits 上 softmax 后采样。

RL 样本是同一玩家从一次决策到其下一次决策的转换，并非从本方动作直接连接到对手观察。换将可能使同一玩家连续行动；终局为双方最后一个动作分别补齐奖励。终局不进行 bootstrap，也不对全非法掩码求最大值。

Reservoir 使用 Algorithm R，每条符合条件的历史样本具有相同保留概率。批量写入遇到同一槽位多次替换时，严格保留最后一次替换；保存总接收计数和独立随机数状态。没有优先经验回放，避免改变平均策略的全历史抽样分布。

## 并行与调度

每轮固定一批完整对局，采样期间模型权重不变，整批完成后集中更新模型。Rust 游戏对象各自独立，线程池复用；Python/Rust 边界传输连续 NumPy 数组，Rust 计算期间释放 GIL。每个推理批次按玩家和行为模式分组。双方网络轮流训练，GPU 由一个进程管理。

初始先手按全局游戏编号交替分配给两个学习者。种子由运行种子和游戏编号确定，工作线程数不会改变发牌。固定并行局数、配置和计算后端时，改变 Rust 线程数应保留采样顺序。改变并行局数会改变权重冻结时长、模式抽样顺序和更新时机，不承诺同一训练轨迹。warmup 期间不积累跨轮的待更新债务；达到 warmup 的当前轮开始累计更新额度。

当前实现不重叠采样与梯度训练。未来若增加异步流水线，需要显式限定模型版本滞后、队列容量和采样/训练比。

## 特征与兼容性

特征版本为 2。保留 465 维结构及 28 个历史动作索引，明确映射 Rust 的 HEARTS/CLUBS/SPADES/DIAMONDS 与旧动作的 HEARTS/SPADES/CLUBS/DIAMONDS 顺序。历史最近五条记录中的赢家现在按观察者编码为“我/对手”；换将记录仍为零历史槽位。当前领先动作单独编码。

特征仅从玩家可见观察生成。第一阶段不读取对手隐藏手牌和牌堆顺序。第二阶段可按规则使用对手手牌信息。Rust 特征入口不接收特权 snapshot。有限五条历史和现有特征仍是一种状态抽象，并不满足通用的完美记忆表示；不能据此承诺收敛到纳什均衡。

旧检查点的特征含义、网络激活和监督经验历史不匹配，新入口明确拒绝它们。此次重构以重新训练为入口，没有静默导入旧权重。

## 安装

需要 Python 3.10+、Rust/Cargo 和相应平台的链接器。首次构建会下载 Cargo 依赖；GPU 训练需要 CUDA 版 PyTorch。项目不更改全局 Python 包。

Windows PowerShell：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install "numpy>=2,<3" "pytest>=8,<10" "maturin>=1.9,<2"
./scripts/build_native.ps1
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

构建脚本会识别此项目已有的 `.tools/` Rust 工具链和本机 MinGW 路径，临时消除继承的 Conda/venv 标识冲突。其他机器可使用正常安装的 Rust 工具链。

Linux/macOS：

```sh
python -m venv .venv
. .venv/bin/activate
# GPU/CPU wheel 按目标平台安装；之后安装本项目。
pip install -e '.[dev]'
```

## 运行

```powershell
# 小规模验证：短训练、评估、完整保存
.venv/Scripts/python.exe -m nfsp train --config configs/smoke.json --games 256 --output runs/smoke --eval-every 128 --eval-games 64 --save-every 128

# 使用吞吐配置训练；games 是总对局目标
.venv/Scripts/python.exe -m nfsp train --config configs/throughput.json --games 100000 --device cuda --output runs/nfsp

# 从完整检查点续训，保留模型、经验池和所有训练计数
.venv/Scripts/python.exe -m nfsp train --resume runs/nfsp/latest.pt --games 200000 --output runs/nfsp

# 使用较小的导出策略文件评估
.venv/Scripts/python.exe -m nfsp evaluate runs/nfsp/policy.pt --games 2000

# 与冻结的历史策略比较，双方都使用平均策略
.venv/Scripts/python.exe -m nfsp evaluate runs/nfsp/policy.pt --opponent runs/nfsp/policy_10240.pt --games 2000

# 测量完整训练而非仅测随机规则模拟
.venv/Scripts/python.exe -m nfsp benchmark --games 1024 --env-counts 1,64,256 --worker-counts 1,8 --device cuda --output runs/benchmark.json

.venv/Scripts/python.exe -m pytest
```

`python main.py train ...` 是等价入口；导入 `main` 不产生训练副作用。自定义参数通过 JSON 配置指定，字段见 `nfsp/config.py`。恢复时只允许覆盖工作线程数和运行设备；跨设备不承诺逐位一致。任意总局数截止时最后一批可能小于 num_envs；比较续训与不中断运行时，也需要使用相同的分批边界。

## 评估、保存与性能解释

评估只读取平均策略，使用独立随机数，不写经验池。相同牌组各评估两局，策略分别先手和后手。报告胜率、按牌组对计算的标准误差，以及实际 Schnapsen 游戏点差。后者是额外指标，训练奖励仍然是胜负。对随机策略胜率和对历史策略胜率都不能代替 exploitability/NashConv。

`latest.pt` 是完整训练快照，`policy.pt` 和周期性策略文件是轻量导出。完整快照包含双方 Q/目标/平均策略、优化器、两个经验池及其采样随机数、reservoir 总计数、采样器随机数、Python/NumPy/PyTorch 随机数和调度计数。仅在完整采样和更新轮次结束后原子替换文件，因此无需丢弃未结束的牌局。保存/评估周期会向上对齐至轮次边界；长任务中断可从上次成功快照恢复。只加载可信的本地完整快照。

`metrics.jsonl` 记录环境、推理、样本汇总、经验池写入、优化更新耗时及吞吐。基准不包含检查点 I/O 和评估，不能直接等同于长期任务平均速度。并行局数、线程数和 batch 大小都应依据端到端测量调节；只增加 Rust 线程不保证更快。本机短基准中 GPU 配置使用 1 个 Rust 线程比 8 个更快，因环境成本占比很小；需要极致吞吐时可用 `--workers 1` 实测，多线程能力仍可用于更大的环境批次。这里的同步版本支持 CPU 或 GPU 上的集中推理和训练，尚未实现独立的 CPU 推理副本。

验证结果与本机测量见 [VALIDATION.md](VALIDATION.md)。
