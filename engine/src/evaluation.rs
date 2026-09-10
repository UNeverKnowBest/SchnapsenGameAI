//! Reproducible independent-game execution; seed mapping does not depend on worker count.
use crate::{Card, Game, Move, Trick};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct Random(u64);
impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn index(&mut self, len: usize) -> usize {
        assert!(len > 0);
        let bound = len as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
    pub fn deck(&mut self) -> [Card; 20] {
        let mut deck = Card::deck();
        for i in (1..20).rev() {
            let j = self.index(i + 1);
            deck.swap(i, j);
        }
        deck
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Statistics {
    pub games: u64,
    pub wins: [u64; 2],
    pub game_points: [u64; 2],
    pub awards_1_2_3: [u64; 3],
    pub actions: u64,
    pub regular_tricks: u64,
    pub marriages: u64,
    pub exchanges: u64,
    pub phase_two_finishes: u64,
    pub winner_direct_points_sum: u64,
}
impl Statistics {
    fn merge(&mut self, rhs: Self) {
        self.games += rhs.games;
        for i in 0..2 {
            self.wins[i] += rhs.wins[i];
            self.game_points[i] += rhs.game_points[i];
        }
        for i in 0..3 {
            self.awards_1_2_3[i] += rhs.awards_1_2_3[i];
        }
        self.actions += rhs.actions;
        self.regular_tricks += rhs.regular_tricks;
        self.marriages += rhs.marriages;
        self.exchanges += rhs.exchanges;
        self.phase_two_finishes += rhs.phase_two_finishes;
        self.winner_direct_points_sum += rhs.winner_direct_points_sum;
    }
}
pub fn random_games(games: u64, seed: u64, workers: usize) -> Result<Statistics, String> {
    if workers == 0 {
        return Err("workers must be positive".into());
    }
    let workers = workers.min(usize::try_from(games).unwrap_or(usize::MAX).max(1));
    let handles: Vec<_> = (0..workers)
        .map(|worker| {
            std::thread::spawn(move || {
                let mut stats = Statistics::default();
                for id in (worker as u64..games).step_by(workers) {
                    let mut rng =
                        Random::new(seed.wrapping_add(id.wrapping_mul(0x9e3779b97f4a7c15)));
                    let mut game = Game::from_deck(rng.deck()).unwrap();
                    while game.current_player().is_some() {
                        let moves = game.legal_moves();
                        let action = moves[rng.index(moves.len())];
                        stats.actions += 1;
                        if matches!(action, Move::Marriage { .. }) {
                            stats.marriages += 1;
                        }
                        match game.step(action).unwrap() {
                            Some(Trick::Exchange { .. }) => stats.exchanges += 1,
                            Some(Trick::Regular { .. }) => stats.regular_tricks += 1,
                            None => (),
                        }
                    }
                    let outcome = game.outcome().unwrap();
                    stats.games += 1;
                    stats.wins[outcome.winner.index()] += 1;
                    stats.game_points[outcome.winner.index()] += outcome.game_points as u64;
                    stats.awards_1_2_3[outcome.game_points as usize - 1] += 1;
                    stats.winner_direct_points_sum += outcome.winner_score.direct_points as u64;
                    if game.snapshot().phase == 2 {
                        stats.phase_two_finishes += 1;
                    }
                }
                stats
            })
        })
        .collect();
    let mut result = Statistics::default();
    for handle in handles {
        result.merge(handle.join().map_err(|_| "worker panicked")?);
    }
    Ok(result)
}
