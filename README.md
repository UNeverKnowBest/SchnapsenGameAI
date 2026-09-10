# SchnapsenGameAI

三种子实测后的 DQN/PPO 参数、60 轮预算建议及 TensorBoard 实时监控见 [推荐设置](docs/RECOMMENDED_SETTINGS.md)。

学术学习曲线现支持训练局数/决策数/时间三个横轴、逐点跨种子 95% 区间、离线实时页面和预设平台期诊断。四算法对比新增独立最终测试与每轮策略导出；运行方法与 20/50 轮预算说明见 [学习曲线与训练预算](docs/LEARNING_CURVES.md)。

一键训练、设备调优、baseline 对比、SVG 图表和 latency 测量：

```powershell
.venv/Scripts/python.exe experiment.py --demo
.venv/Scripts/python.exe experiment.py
```

默认正式实验为 3 个训练种子，每个算法每种子最多 20 × 5,000 局、每批最多 4 个 PPO epochs；支持验证集早停和 `--max-decisions`。先测 CPU/CUDA 完整训练吞吐量，再选设备、环境批量与线程数。详见 [一键实验说明](docs/EXPERIMENT.md) 和 [本机实测结果与图表](docs/EXPERIMENT_VALIDATION.md)。

High-Entropy PPO self-play, standard PPO ablations, DQN/NFSP comparisons, paired-seat tournaments, and controlled parallel benchmarks are available. See [the research guide](docs/RESEARCH.md), [feature representation analysis](docs/FEATURES.md), and [measured results](docs/RESEARCH_VALIDATION.md).

```sh
python -m nfsp ppo-train --config configs/he_ppo.json --games 100000 --output runs/he-ppo
python -m nfsp compare --games 100000 --seeds 42,43,44 --output runs/comparison
python -m nfsp performance --games 1024 --repeats 3 --output runs/performance
python -m nfsp feature-audit --output runs/feature_audit
```

The comparison, performance, arena, and feature audit commands generate standalone `report.html`, `results.json`, and `results.csv` files in their output directories.

NFSP training now uses a Rust multithreaded batch environment and PyTorch batched inference/learning. See [the training guide](docs/NFSP.md) for installation, algorithm details, training, checkpoints and evaluation. Run `python -m nfsp --help` after building the extension. Unused historical Python bot/buffer/model implementations have been removed; active entry points are `experiment.py`, `main.py` and `python -m nfsp`.

The independent deterministic Rust Schnapsen engine lives in `engine/`. The following sections document its rules compatibility and standalone tooling.

The engine targets the external [VU course Schnapsen engine at commit ca0b3d9](https://github.com/intelligent-systems-course/schnapsen/tree/ca0b3d9cd9c3922a10303536e28f1266fe3a2c0d) (declared package version 0.0.5). **The original project's installed version cannot be established.** This is a deliberate pinned compatibility target selected from upstream history before Rust implementation. See [the manifest](compatibility/manifest.json) for evidence and the comparison with current upstream.

## Build and use

Requires Rust/Cargo (tested with Rust 1.98.1) and Python 3.10+ for oracle checks only (tested with Python 3.13.5). The compiled engine has no Python runtime dependency.

```sh
cargo test --locked --manifest-path engine/Cargo.toml
cargo build --release --locked --manifest-path engine/Cargo.toml
cargo run --release --locked --manifest-path engine/Cargo.toml -- evaluate 100000 42 4
cargo bench --locked --manifest-path engine/Cargo.toml --bench independent_games
```

Evaluation runs uniform-random policies for both seats, reporting wins, game points, award distribution, actions, tricks, marriages, exchanges and throughput. Game seeds are independent of worker count, and statistics are identical for the same seed/game count across worker counts. Seat 0 leads initially, so these are seat-specific results, not a balanced comparison between different policies. Set `SCHNAPSEN_BENCH_GAMES` to change benchmark size.

On this Windows workspace a local toolchain was installed in ignored `.tools/`, without changing the user's persistent PATH. To use it in PowerShell:

```powershell
$env:RUSTUP_HOME = "$PWD/.tools/rustup"
$env:CARGO_HOME = "$PWD/.tools/cargo"
$env:PATH = "$PWD/.tools/cargo/bin;$env:PATH"
```

A GNU Windows toolchain also needs a working MinGW GCC linker (this machine uses `C:/msys64/ucrt64/bin`). A normal installed MSVC or Unix Rust toolchain can build the crate too; those platforms were not locally validated.

## Differential validation

```sh
python compatibility/python_oracle/bootstrap.py
python compatibility/python_oracle/test_adapter.py
python compatibility/differential/run.py --games 10000 --report compatibility/differential/report.json
```

The adapter imports only the pinned reference's standard-library engine/deck modules. No ML dependencies are needed for these checks. The ignored reference checkout stays under `compatibility/python_oracle/upstream/` and is verified against the manifest before use.

`--mode states` runs privileged engine-state comparison; `--mode observations` runs player-visible comparison. Default `all` runs both. Each request supplies an explicit deck and semantic actions, so Python/Rust RNG differences cannot mask or create rule mismatches. Tests preserve legal-move order as well as compare normalized sets. Failures save the exact input and both traces for replay. See [the protocol](compatibility/PROTOCOL.md).

To run upstream core tests unchanged, install its dependencies in a separate environment:

```sh
python -m venv compatibility/.venv
# Activate the environment, or invoke its Python executable directly.
python -m pip install -e "compatibility/python_oracle/upstream[test]"
python -m pytest compatibility/python_oracle/upstream/tests/test_schnapsen_implementation.py compatibility/python_oracle/upstream/tests/test_game.py compatibility/python_oracle/upstream/tests/test_deck.py compatibility/python_oracle/upstream/tests/test_repr.py compatibility/python_oracle/upstream/tests/bots/test_randbot.py
```

On Windows that interpreter is `compatibility/.venv/Scripts/python.exe`; on Unix it is `compatibility/.venv/bin/python`.

## API and boundaries

`Game::from_deck` takes an explicit 20-card permutation. `legal_moves` and `step` implement deterministic single-action progression; `outcome` returns winner, game points and winning score. A marriage/exchange is a semantic move variant, not a Python class clone. `Game::clone` supports independent simulations; `from_position` supports structurally validated constructed scenarios with history reset.

`PlayerObservation` contains owned public values and perspective history. `Position`, `EngineSnapshot` and `Record` are privileged diagnostics and must not be passed to agents. The Rust `Bot` trait receives only observations and supports exchange/game-end notifications. The JSON Lines executable and Python adapter share the [documented canonical format](compatibility/PROTOCOL.md).

[Legacy API notes](compatibility/LEGACY_API.md) document the old bot contract, action-index mapping and the unexplained required `eta` callback argument. The subsequent NFSP implementation adds a separate Python extension in `native/` and training package in `nfsp/`; it does not recreate the old runtime bot callback API. See [NFSP feature and checkpoint compatibility](docs/NFSP.md).

## Recorded validation

* 24 handwritten oracle-verified scenarios, checked into golden fixtures.
* 12 Rust tests, including 1,520 distinct card-pair/trump winner cases, 2,000 full invariant games, hidden-information checks, callbacks and worker determinism.
* 40 upstream tests passed unchanged.
* Adapter verified against actual reference bot callbacks in 100 complete games.
* **10,000 complete differential trajectories, zero unexplained mismatches:** 161,510 state records; 171,517 observations; 881,781 history views.
* Rust formatting and Clippy checks pass.

See [the machine-readable report](compatibility/differential/report.json), [coverage and exclusions](compatibility/COVERAGE.md), and [benchmark record](engine/benches/RESULTS.md). This is finite compatibility evidence for the pinned standard-game contract, not a proof of all trajectories or confirmation of the unknown historical installation.
