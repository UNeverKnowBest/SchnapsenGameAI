# 实测后的推荐配置与 TensorBoard

本次只设置参数、安装监控依赖和导入历史日志；没有启动新训练，也没有启动 TensorBoard 服务。

## 选择依据

上一轮完整完成了四算法 × 三训练种子 × 每种子 100,000 局，共 1,200,000 训练局。以下为独立最终测试结果；95% 区间按三个训练种子计算。

| 算法 | 对随机策略胜率 | 对规则策略胜率 | 对规则策略的 95% 区间 |
| --- | ---: | ---: | --- |
| DQN | 79.54% | 58.47% | 55.14%–61.80% |
| 标准 PPO | 79.15% | 56.30% | 54.81%–57.79% |
| HE-PPO | 74.16% | 49.17% | 45.66%–52.69% |
| NFSP | 64.65% | 34.10% | 33.44%–34.76% |

优先使用 DQN，并保留标准 PPO 对照。DQN 直接对战 PPO 的种子平均胜率为 51.89%，95% 区间 49.62%–54.16%，尚不能宣称它明确强于 PPO。这轮进行了算法比较，没有做超参数搜索，以下是目前有实测依据的推荐设置，不是全局最佳参数证明。

数据来源：[原始结果](../runs/academic_comparison_20260910/results.json)、[完整报告](../runs/academic_comparison_20260910/report.html)。

## 已保存的参数

- [DQN 参数](../configs/recommended_dqn.json)：复制本次实测 DQN 配置。
- [PPO 对照参数](../configs/recommended_ppo.json)：复制本次实测标准 PPO 配置。
- [下一轮研究预算](../configs/recommended_research.json)：轮数、评估量及多种子设置的说明性配置。

| DQN 参数 | 推荐值 | 含义 |
| --- | --- | --- |
| hidden | 256, 128, 64 | ReLU 网络隐藏层 |
| rl_lr | 0.0003 | Adam 学习率 |
| gamma | 1.0 | 终局奖励、不折扣 |
| batch_size | 256 | 每次学习的经验条数 |
| replay_capacity | 100000 | 每个玩家的 RL 经验池 |
| rl_warmup | 512 | 开始更新前的最小经验量 |
| learn_every | 128 | 每玩家每新增 128 条经验获得一次更新额度 |
| target_every | 1000 | 每玩家每 1,000 次 RL 更新同步目标网络 |
| epsilon_start / end | 0.2 / 0.01 | 探索率起点/下限 |
| epsilon_decay_steps | 1000000 | 按每个玩家自己的决策数衰减 |
| double_dqn / eta | true / 1.0 | Double DQN、自博弈最佳响应策略；DQN 实现不训练 NFSP 平均策略 |
| grad_clip | 5.0 | 梯度裁剪 |
| num_envs / workers | 256 / 1 | 环境批量 / Rust 线程数 |
| device / torch_threads | cpu / 4 | 复用本机已完成实验的运行配置，非所有硬件的最优保证 |

标准 PPO 对照采用：lr=3e-4、gamma=1、GAE lambda=.95、clip=.2、epochs=4、minibatch=512、target_KL=.02、entropy_coef=.01、grad_clip=.5；关闭自适应熵和自适应学习率。不建议根据本次结果继续默认高熵目标 .8。

DQN 的 JSON 保留了共享 Config 中的 SL/reservoir 字段，它们在 DQN 学习器中不参与平均策略学习。请用 `compare --algorithms dqn`，不要把 DQN 配置交给默认 NFSP 的 `train` 入口。

`compare` 新增 `--algorithm-config 算法=JSON`，可以分别读取两份参数。种子、设备、环境并行数、线程数和网络宽度由共享 CLI 参数覆盖，以维持共同实验设置；算法自己的学习率、经验池、探索和更新参数从文件读取。真实采用的完整配置写入协议和每次运行结果。

## 推荐训练轮数

**下一次建议 60 轮，每轮 5,000 局，即每算法每种子 300,000 局。正式比较使用 5 个种子 42–46。**

依据：DQN 在末尾五个评估点中，后两个点相比前两个点，对规则策略的胜率提升约 7.50 个百分点，95% 区间约 +3.86 至 +11.15 个百分点；20 轮时仍明显改善。训练到 100,000 局时，每玩家 epsilon 约 .063，尚未达到 .01 的下限。扩大到三倍训练局数是合理的下一阶段预算，但未实测 60 轮，不能称为最佳或已收敛轮数。

保持每 5,000 局一次评估，每对手评估 4,000 局，独立最终测试 10,000 局。DQN 与 PPO 使用相同固定预算；不按最终测试结果提前挑选某个算法的停止点。平台期规则沿用 5 点窗口、2 个百分点阈值。如果 60 轮的验证曲线仍改善，可以预先安排上限 100 轮的独立后续实验；若曲线波动主导判断，应提高评估样本量，而不是只加训练。

这里一轮是评估区间。PPO 的 4 epochs 是每批样本的优化遍数，不应改成 60。两算法、5 种子、60 轮合计 3,000,000 训练局；本次没有运行该预算。

以下命令仅供以后手动运行，当前未执行：

```powershell
.venv/Scripts/python.exe -m nfsp compare `
  --algorithms dqn,ppo `
  --algorithm-config dqn=configs/recommended_dqn.json `
  --algorithm-config ppo=configs/recommended_ppo.json `
  --games 300000 --eval-every 5000 --eval-games 4000 --test-games 10000 `
  --seeds 42,43,44,45,46 --device cpu --num-envs 256 --workers 1 --torch-threads 4 `
  --output runs/recommended_research
```

## TensorBoard 已配置

项目虚拟环境已安装 TensorBoard。复现安装可以使用 `python -m pip install "tensorboard>=2.18,<3"`；`pyproject.toml` 的可选依赖组为 `monitor`。

已经把历史 `events.jsonl` 转换到了 `runs/tensorboard/academic_comparison_20260910`。只查看已有结果，手动执行下面命令即可，**不会启动训练**：

```powershell
.venv/Scripts/python.exe -m tensorboard.main --logdir runs/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 2
```

浏览器打开 `http://127.0.0.1:6006`。未来训练时，在另一个终端启动日志跟随器：

```powershell
.venv/Scripts/python.exe -m nfsp.tensorboard `
  --events runs/recommended_research/events.jsonl `
  --logdir runs/tensorboard/recommended_research --follow
```

跟随器本身不训练；目标文件尚未生成时会等待，默认每秒检查追加记录。支持 `compare` 与 `experiment.py` 的 `events.jsonl`，并按算法/训练种子分开显示。普通 `train` 的 `metrics.jsonl` 不使用此事件格式。

- `eval/heuristic/win_rate`：对规则策略的评估胜率，Step 为累计训练局数。
- `eval/random/win_rate`：对随机策略的评估胜率。
- `eval_by_decisions/...`：Step 为累计环境动作数。
- `eval_by_training_ms/...`：Step 为累计纯训练毫秒数。
- `train/losses/player_0/rl` 等：DQN 两个玩家各自的学习损失，实际叶标签沿用事件文件字段。
- `train/epsilon/player_0`、`train/rl_updates/player_0`：探索率与 RL 更新次数。
- PPO 的 `train/losses/policy_loss`、`value_loss`、`final_kl`、`normalized_entropy`；一键实验中的 loss 字段在 `train/` 下直接展开。
- `test/...`：独立最终测试，和周期评估分开显示。

TensorBoard 选择 Step 轴可查看明确的训练预算；其 Wall/Relative 时间反映事件写入时间，历史日志导入后不能把它当作原始训练用时。按实际训练时间比较请使用 `eval_by_training_ms`。TensorBoard 的每个 run 是一个种子，不会自动变成统计置信区间；正式报告仍使用原有 HTML/SVG 的跨种子区间。

跟随器保存偏移，正常退出再启动不会重复导入。文件被截断或替换时应使用新 logdir；同一 logdir 不要同时运行两个跟随器。进程在写入与保存偏移之间崩溃时，重启可能重复最后一小批事件。Ctrl+C 只停止监控，不会结束训练。

实现使用 PyTorch 官方的 [SummaryWriter](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)；界面启动与日志目录机制见 [TensorBoard 官方说明](https://www.tensorflow.org/tensorboard/get_started)。
