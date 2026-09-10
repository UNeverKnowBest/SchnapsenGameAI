//! Versioned public-information encoder. Legacy dimensions/action indices are retained.
//! Version 2 fills historical winners correctly; version 1 weights are not equivalent.
use schnapsen_engine::{Card, Move, PlayerObservation, Rank, Suit, Trick};

pub const STATE_DIM: usize = 465;
pub const ACTION_DIM: usize = 28;
pub const FEATURE_VERSION: u32 = 2;
const ACTION_SUITS: [u8; 4] = [0, 2, 1, 3];

pub fn action_id(action: Move) -> usize {
    match action {
        Move::Play { card } => {
            ACTION_SUITS[card.suit().id() as usize] as usize * 5 + card.id() as usize % 5
        }
        Move::Marriage { suit } => 20 + ACTION_SUITS[suit.id() as usize] as usize,
        Move::Exchange { suit } => 24 + ACTION_SUITS[suit.id() as usize] as usize,
    }
}
pub fn decode_action(id: usize) -> Result<Move, String> {
    if id >= ACTION_DIM {
        return Err("action must be in 0..28".into());
    }
    let suit_id = ACTION_SUITS[if id < 20 { id / 5 } else { (id - 20) % 4 }];
    let suit = Suit::try_from(suit_id)?;
    Ok(if id < 20 {
        Move::Play {
            card: Card::try_from(suit_id * 5 + (id % 5) as u8)?,
        }
    } else if id < 24 {
        Move::Marriage { suit }
    } else {
        Move::Exchange { suit }
    })
}
fn move_feature(action: Option<Move>, out: &mut [f32]) {
    let Some(action) = action else {
        return;
    };
    let (kind, card) = match action {
        Move::Play { card } => (0, card),
        Move::Marriage { suit } => (2, Card::new(suit, Rank::Queen)),
        Move::Exchange { suit } => (1, Card::new(suit, Rank::Jack)),
    };
    out[kind] = 1.;
    let rank = match card.rank() {
        Rank::King => 0,
        Rank::Queen => 1,
        Rank::Jack => 2,
        Rank::Ten => 3,
        Rank::Ace => 12,
    };
    out[3 + rank] = 1.;
    out[16 + (3 - card.suit().id()) as usize] = 1.;
}
pub fn encode(obs: &PlayerObservation) -> ([f32; STATE_DIM], [u8; ACTION_DIM]) {
    let v = &obs.view;
    let mut state = [0.; STATE_DIM];
    state[..5].copy_from_slice(&[
        v.my_score.direct_points as f32 / 66.,
        v.my_score.pending_points as f32 / 66.,
        v.opponent_score.direct_points as f32 / 66.,
        v.opponent_score.pending_points as f32 / 66.,
        v.talon_size as f32 / 20.,
    ]);
    state[5 + (3 - v.trump.id()) as usize] = 1.;
    state[9 + usize::from(v.phase != 2)] = 1.;
    state[11 + usize::from(v.am_i_leader)] = 1.;
    for card in Card::deck() {
        let category = if v.hand.contains(&card) {
            5
        } else if v.won_cards.contains(&card) {
            4
        } else if v.known_opponent_cards.contains(&card) {
            3
        } else if v.opponent_won_cards.contains(&card) {
            2
        } else if v.trump_card == Some(card) {
            1
        } else {
            0
        };
        state[13 + card.id() as usize * 6 + category] = 1.;
    }
    // The last history element is the current observation, without a completed trick.
    for (i, entry) in obs.history.iter().rev().skip(1).take(5).enumerate() {
        if let Some(Trick::Regular {
            leader_move,
            winner,
            ..
        }) = &entry.trick
        {
            let offset = 133 + 22 * i;
            move_feature(Some(*leader_move), &mut state[offset..offset + 20]);
            state[offset + 20 + usize::from(*winner != v.player)] = 1.;
        }
    }
    let mut mask = [0; ACTION_DIM];
    assert!(
        v.legal_moves.len() <= 10,
        "feature schema supports at most ten legal moves"
    );
    for (i, &action) in v.legal_moves.iter().enumerate() {
        mask[action_id(action)] = 1;
        move_feature(Some(action), &mut state[243 + 20 * i..263 + 20 * i]);
    }
    state[443] = f32::from(
        v.legal_moves
            .iter()
            .any(|m| matches!(m, Move::Exchange { .. })),
    );
    state[444] = f32::from(v.hand.iter().any(|c| c.suit() == v.trump));
    move_feature(v.leader_move, &mut state[445..465]);
    (state, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use schnapsen_engine::{Game, Player};
    #[test]
    fn mapping_round_trip_and_suit_order() {
        for id in 0..28 {
            assert_eq!(action_id(decode_action(id).unwrap()), id);
        }
        assert_eq!(
            action_id(Move::Play {
                card: Card::new(Suit::SPADES, Rank::Jack)
            }),
            5
        );
        assert_eq!(
            action_id(Move::Play {
                card: Card::new(Suit::CLUBS, Rank::Jack)
            }),
            10
        );
        assert!(decode_action(28).is_err());
    }
    #[test]
    fn hidden_cards_do_not_change_features() {
        let deck = Card::deck();
        let mut other = deck;
        other.swap(1, 3); // Two cards in opponent hand.
        other.swap(12, 14); // Two unseen talon cards.
        let a = Game::from_deck(deck).unwrap();
        let b = Game::from_deck(other).unwrap();
        assert_eq!(
            encode(&a.observation(Player::ZERO)),
            encode(&b.observation(Player::ZERO))
        );
    }
    #[test]
    fn recent_history_matches_full_and_records_winner() {
        let mut game = Game::from_deck(Card::deck()).unwrap();
        let mut saw_winner = false;
        while let Some(player) = game.current_player() {
            let full = encode(&game.observation(player));
            assert_eq!(full, encode(&game.observation_recent(player, 5)));
            saw_winner |= full.0[153] + full.0[154] == 1.;
            game.step(game.legal_moves()[0]).unwrap();
        }
        assert!(saw_winner);
    }
}
