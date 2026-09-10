# NFSP 验证记录

2026-09-10，本地 Windows，Intel Core i9-14900HX（24 核 / 32 逻辑处理器）、约 32 GB 内存、RTX 4080 Laptop 12 GB。Python 3.13.5、PyTorch 2.9.1+cu128、NumPy 2.5.3。Rust 1.98.1 GNU Windows 工具链，native release 构建。实际版本以机器环境和 JSON 记录为准。

## 正确性

- `.venv/Scripts/python.exe -m pytest -q`：**16 passed**。
- `cargo test --locked --manifest-path native/Cargo.toml`：**3 passed**。
- `cargo test --locked --manifest-path engine/Cargo.toml`：**12 passed**，保留原规则回归。
- Rust fmt、native 全 target Clippy `-D warnings`、`git diff --check` 通过。
- Windows 构建脚本和安装后的 `schnapsen-nfsp` 命令入口已实际运行。

覆盖：完整轨迹在线程数 1/4 下相同、批量非法动作原子拒绝、配对牌组、输出数组所有权、28 动作语义映射、第一阶段隐藏信息隔离、有限历史与完整观察一致、真实历史赢家、RL 环形覆盖、reservoir 批量与逐条 Algorithm R 一致、抽样统计无明显近期偏差、reservoir 恢复、Double DQN 非法动作/终局处理、零概率动作不会被选取、按局模式、双方终局奖励、换将后的同玩家转换、评估隔离、CPU 精确恢复（含改变 Rust 线程数）、旧检查点拒绝、导入无副作用。

CPU 精确恢复测试比较下一轮损失和所有模型参数，要求逐位一致。未承诺跨 GPU、驱动、平台或不同并行局数的逐位一致。

## GPU 集成训练

先完成 256 局 smoke（周期评估/保存），再用吞吐配置训练 20,000 局，恢复完整 GPU 检查点并切换到 1 个 Rust 线程继续至 20,256 局。

[机器可读摘要](training_validation.json)：

- 280,836 条双方决策经验。
- 双方各 274 次 RL、258 次 SL 更新，均超过目标网络 250 次更新同步周期。
- 双方 RL 池均达到 100,000 条并发生循环覆盖。
- SL 池分别为 13,386 / 13,633 条；较小容量下的替换、均匀性和恢复由单元测试覆盖。
- 累计训练计算耗时 14.771 秒，约 **1,371 局/秒**；排除评估、保存和加载。
- 模型、优化器、经验池及 RNG 已保存至本机 `runs/validation/latest.pt`，平均策略导出至 `runs/validation/policy.pt`。这些训练产物被 Git 忽略。

20,000 局时，双方平均策略对随机对手各评估 1,000 局，胜率 54.8% / 53.4%，配对标准误差约 1.43% / 1.41%。20,256 局策略对 10,240 局历史策略的结果见 [历史策略评估](evaluation_vs_history.json)。这些小规模、单种子结果用于验证训练和评估流程，不证明充分训练、均衡收敛或稳定的策略优势。

## 端到端基准

以下均包含环境推进、特征、神经网络推理、经验池操作和 RL/SL 优化；不包含检查点 I/O 与评估。每组是单次本机测量，GPU 动态频率、系统负载和短时启动效应会影响结果。

1,024 局，标准网络、batch=256、learn_every=128；为在短基准中覆盖更新，经验池容量设为 20,000、warmup=256：

| 计算设备 | 同时运行局数 | Rust 线程 | 局/秒 |
| --- | ---: | ---: | ---: |
| CPU | 1 | 1 | 138 |
| CPU | 256 | 8 | 764 |
| GPU | 1 | 1 | 62 |
| GPU | 256 | 1 | 1,214 |
| GPU | 256 | 8 | 1,074 |

原始数据：[CPU](benchmark_cpu.json)、[GPU](benchmark_cuda.json)。这里的单局基线同样使用新的正确 NFSP 和 Rust 引擎，**不是旧 main.py 的实测结果**。旧程序存在回调签名和训练语义问题，没有以其结果作为公平速度基线。

4,096 局，GPU，256 个环境：

| 配置 | Rust 线程 | 局/秒 |
| --- | ---: | ---: |
| 标准 batch=256 | 1 | 834 |
| 标准 batch=256 | 8 | 721 |
| 吞吐 batch=1024 | 1 | 1,598 |
| 吞吐 batch=1024 | 4 | 1,400 |
| 吞吐 batch=1024 | 8 | 1,349 |

原始数据：[标准](benchmark_cuda_4096.json)、[吞吐配置](benchmark_throughput_4096.json)。

吞吐配置把 learn_every 同比例调为 512，稳定阶段抽样量/新经验量保持为 2；但优化器更新节奏、warmup、短程实际更新次数和最终轨迹有差异。这是吞吐比较，不是等策略质量比较。模型更新已经占主要时间，当前小环境任务中 8 个 Rust 线程没有比 1 个带来更高整体速度。架构保留多线程能力，线程数应按目标负载测量选择。

复现：

```powershell
.venv/Scripts/python.exe -m nfsp benchmark --games 1024 --env-counts 1,64,256 --worker-counts 1,8 --device cuda --output runs/benchmark.json
.venv/Scripts/python.exe -m nfsp benchmark --config configs/throughput.json --games 4096 --env-counts 256 --worker-counts 1,4,8 --device cuda --output runs/throughput.json
```

尚未验证：其他平台构建、长期多随机种子学习曲线、精确 exploitability/NashConv、异步采样与训练重叠、独立 CPU 推理副本、多 GPU。
