use schnapsen_engine::{Card, Game, Move, Position, Record, evaluation};
use serde::Deserialize;
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    deck: Option<[Card; 20]>,
    position: Option<Position>,
    #[serde(default)]
    actions: Vec<Move>,
}
fn replay(request: Request) -> Result<Vec<Record>, String> {
    let mut game = match (request.deck, request.position) {
        (Some(deck), None) => Game::from_deck(deck)?,
        (None, Some(position)) => Game::from_position(position)?,
        _ => return Err("provide exactly one of deck or position".into()),
    };
    let mut records = vec![game.record()];
    for action in request.actions {
        game.step(action)?;
        records.push(game.record());
    }
    Ok(records)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "evaluate") {
        let games = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(10000);
        let seed = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(42);
        let workers = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(1);
        let start = std::time::Instant::now();
        let stats = evaluation::random_games(games, seed, workers)?;
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({
                "policy": "uniform_random_both_seats", "seed": seed, "workers": workers,
                "statistics": stats, "seconds": seconds, "games_per_second": games as f64 / seconds
            })
        );
        return Ok(());
    }
    if args.len() > 1 && args[1] != "replay" {
        return Err("usage: schnapsen-engine [replay | evaluate [games seed workers]]".into());
    }
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let response = serde_json::from_str::<Request>(&line?)
            .map_err(|e| e.to_string())
            .and_then(replay);
        let value = match response {
            Ok(records) => serde_json::json!({"records": records}),
            Err(error) => serde_json::json!({"error": error}),
        };
        serde_json::to_writer(&mut stdout, &value)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
