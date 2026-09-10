# Independent-game benchmark

Measured 2026-09-10 on Windows, Intel Core i9-14900HX (24 physical cores / 32 logical processors), Rust 1.98.1, x86_64-pc-windows-gnu, optimized release build.

Command: `cargo bench --locked --manifest-path engine/Cargo.toml --bench independent_games`

| Games | Workers | Seconds | Games/second | Actions/second |
| --- | --- | --- | --- | --- |
| 100,000 | 1 | 0.485 | 206,299 | 3,269,789 |
| 100,000 | 32 | 0.090 | 1,112,467 | 17,632,325 |

These are single local measurements, including thread startup, deck shuffling, legal-move selection, state/history transitions and statistics. They exclude JSON serialization and observation construction for bot inference. Short wall times and machine load affect the numbers; increase `SCHNAPSEN_BENCH_GAMES` for more stable measurements. No comparison to Python throughput is claimed.

Separate CLI evaluation with seed 42 and 100,000 games gave identical statistics on 1 and 4 workers: wins [57,549, 42,451], game points [102,726, 65,603], 1,584,975 actions, 784,883 regular tricks, 22,149 marriages, 15,209 exchanges. These describe uniform-random play by seat and are not evidence of semantic compatibility.
