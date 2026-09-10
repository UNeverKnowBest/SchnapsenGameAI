use schnapsen_engine::evaluation::random_games;
fn main() {
    let games = std::env::var("SCHNAPSEN_BENCH_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    for workers in [
        1,
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    ] {
        let start = std::time::Instant::now();
        let stats = std::hint::black_box(random_games(games, 42, workers).unwrap());
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "games={} workers={} seconds={:.3} games/s={:.0} actions/s={:.0}",
            stats.games,
            workers,
            elapsed,
            stats.games as f64 / elapsed,
            stats.actions as f64 / elapsed
        );
    }
}
