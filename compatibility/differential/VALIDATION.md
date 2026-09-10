# Recorded checks

Reference: `ca0b3d9cd9c3922a10303536e28f1266fe3a2c0d`; package metadata 0.0.5; Windows; Python 3.13.5; Rust 1.98.1 GNU; checked 2026-09-10.

* `cargo test --manifest-path engine/Cargo.toml`: 12 passing integration tests.
* `cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings`: passed.
* `cargo fmt --manifest-path engine/Cargo.toml --check`: passed.
* `python compatibility/python_oracle/test_adapter.py`: 2 passing tests, including 100 complete actual-callback comparisons.
* Pinned upstream core modules plus random-bot smoke test: 40 passed in 1.41 seconds, run unchanged in the isolated virtual environment.
* `python compatibility/differential/run.py --games 10000 --report compatibility/differential/report.json`: 10,000 complete games, 24 handwritten cases, both state and observation modes, zero unexplained mismatches.

See `report.json` for counters/seed and `../COVERAGE.md` for exact scope. Golden cases are reused by Rust tests without requiring Python. Rule-equivalence trajectories use explicit Python-shuffled deck orders and identical semantic actions; they do not require RNG equivalence.

The initial adapter bring-up exposed the reference Talon constructor consuming a generator twice. The adapter now always supplies materialized lists. This was an adapter setup issue, resolved before the recorded acceptance run; no reference source was modified.
