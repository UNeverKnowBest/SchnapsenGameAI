//! Independent deterministic engine. Python is used only by external compatibility tests.
//! Cards and legal moves retain reference order. Players have stable identities across tricks.
mod domain;
pub mod evaluation;
pub use domain::*;
use serde::{Deserialize, Serialize};

/// Privileged position: never pass this or Game to a bot.
/// Restoring a position deliberately starts with no public history.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub hands: [Vec<Card>; 2],
    pub talon: Vec<Card>,
    pub trump: Suit,
    pub scores: [Score; 2],
    pub won_cards: [Vec<Card>; 2],
    pub leader: Player,
}
impl Position {
    pub fn phase(&self) -> u8 {
        if self.talon.is_empty() { 2 } else { 1 }
    }
    fn validate(&self) -> Result<(), String> {
        if self.hands[0].len() != self.hands[1].len() || self.hands[0].len() > 5 {
            return Err("hands must have equal size, at most five".into());
        }
        if self.talon.len() > 10
            || !self.talon.len().is_multiple_of(2)
            || (!self.talon.is_empty() && self.hands[0].len() != 5)
        {
            return Err("invalid talon/hand sizes".into());
        }
        if self.talon.last().is_some_and(|c| c.suit() != self.trump) {
            return Err("bottom card must match trump".into());
        }
        let mut mask = 0;
        let mut count = 0;
        for c in self
            .hands
            .iter()
            .flatten()
            .chain(&self.talon)
            .chain(self.won_cards.iter().flatten())
        {
            if mask & c.bit() != 0 {
                return Err("duplicate card".into());
            }
            mask |= c.bit();
            count += 1;
        }
        if count != 20 {
            return Err("position must account for all 20 cards".into());
        }
        if self
            .won_cards
            .iter()
            .any(|cards| !cards.len().is_multiple_of(2))
        {
            return Err("won cards must form complete tricks".into());
        }
        if self
            .scores
            .iter()
            .any(|s| s.direct_points > 240 || s.pending_points > 120)
            || self.scores[self.leader.other().index()].direct_points >= 66
        {
            return Err("invalid score or terminal follower".into());
        }
        Ok(())
    }
    fn outcome(&self) -> Option<Outcome> {
        let leader = self.leader.index();
        let loser_score = self.scores[self.leader.other().index()].direct_points;
        let points = if self.scores[leader].direct_points >= 66 {
            if loser_score == 0 {
                3
            } else if loser_score < 33 {
                2
            } else {
                1
            }
        } else if self.hands[0].is_empty() && self.talon.is_empty() {
            1
        } else {
            return None;
        };
        Some(Outcome {
            winner: self.leader,
            game_points: points,
            winner_score: self.scores[leader],
        })
    }
    fn leader_moves(&self) -> Vec<Move> {
        let hand = &self.hands[self.leader.index()];
        let mut moves: Vec<_> = hand.iter().map(|&card| Move::Play { card }).collect();
        if !self.talon.is_empty() && hand.contains(&Card::new(self.trump, Rank::Jack)) {
            moves.push(Move::Exchange { suit: self.trump });
        }
        for card in hand {
            if card.rank() == Rank::Queen && hand.contains(&Card::new(card.suit(), Rank::King)) {
                moves.push(Move::Marriage { suit: card.suit() });
            }
        }
        moves
    }
    fn follower_moves(&self, lead: Move) -> Vec<Move> {
        let hand = &self.hands[self.leader.other().index()];
        let target = lead
            .follow_target()
            .expect("follower cannot respond to exchange");
        let mut candidates: Vec<Card> = hand.clone();
        if self.phase() == 2 {
            let same: Vec<_> = hand
                .iter()
                .copied()
                .filter(|c| c.suit() == target.suit())
                .collect();
            if !same.is_empty() {
                let higher: Vec<_> = same
                    .iter()
                    .copied()
                    .filter(|c| c.points() > target.points())
                    .collect();
                candidates = if higher.is_empty() { same } else { higher };
            } else {
                let trumps: Vec<_> = hand
                    .iter()
                    .copied()
                    .filter(|c| c.suit() == self.trump)
                    .collect();
                if !trumps.is_empty() {
                    candidates = trumps;
                }
            }
        }
        candidates
            .into_iter()
            .map(|card| Move::Play { card })
            .collect()
    }
}

/// Flat public view. There is no hidden state pointer, opponent hand in phase one, or talon order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerView {
    pub player: Player,
    pub am_i_leader: bool,
    pub hand: Vec<Card>,
    pub my_score: Score,
    pub opponent_score: Score,
    pub won_cards: Vec<Card>,
    pub opponent_won_cards: Vec<Card>,
    pub trump: Suit,
    pub trump_card: Option<Card>,
    pub talon_size: usize,
    pub phase: u8,
    pub known_opponent_cards: Vec<Card>,
    pub opponent_hand: Option<Vec<Card>>,
    pub seen_cards: Vec<Card>,
    pub leader_move: Option<Move>,
    pub legal_moves: Vec<Move>,
    pub legal_moves_status: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub view: PlayerView,
    pub trick: Option<Trick>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerObservation {
    pub view: PlayerView,
    pub history: Vec<HistoryEntry>,
}

#[derive(Clone, Debug)]
struct Frame {
    position: Position,
    revealed: u32,
    trick: Trick,
}

#[derive(Clone, Debug)]
pub struct Game {
    position: Position,
    pending: Option<Move>,
    revealed: u32,
    history: Vec<Frame>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineSnapshot {
    #[serde(flatten)]
    pub position: Position,
    pub current_player: Option<Player>,
    pub leader_move: Option<Move>,
    pub legal_moves: Vec<Move>,
    pub phase: u8,
    pub terminal: bool,
    pub winner: Option<Outcome>,
    pub tricks: Vec<Trick>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    pub state: EngineSnapshot,
    pub observations: Vec<PlayerObservation>,
}

impl Game {
    pub fn from_deck(deck: [Card; 20]) -> Result<Self, String> {
        let position = Position {
            hands: [
                deck[..10].iter().step_by(2).copied().collect(),
                deck[1..10].iter().step_by(2).copied().collect(),
            ],
            talon: deck[10..].to_vec(),
            trump: deck[19].suit(),
            leader: Player::ZERO,
            scores: [Score::default(); 2],
            won_cards: [vec![], vec![]],
        };
        Self::from_position(position)
    }
    pub fn from_position(position: Position) -> Result<Self, String> {
        position.validate()?;
        Ok(Self {
            position,
            pending: None,
            revealed: 0,
            history: vec![],
        })
    }
    pub fn outcome(&self) -> Option<Outcome> {
        self.position.outcome()
    }
    pub fn current_player(&self) -> Option<Player> {
        if self.outcome().is_some() {
            None
        } else {
            Some(if self.pending.is_some() {
                self.position.leader.other()
            } else {
                self.position.leader
            })
        }
    }
    pub fn legal_moves(&self) -> Vec<Move> {
        if self.outcome().is_some() {
            vec![]
        } else if let Some(lead) = self.pending {
            self.position.follower_moves(lead)
        } else {
            self.position.leader_moves()
        }
    }
    /// One semantic action. An invalid action leaves the entire game unchanged.
    /// A lead is pending until the follower responds, exactly as in reference callbacks.
    pub fn step(&mut self, action: Move) -> Result<Option<Trick>, String> {
        if !self.legal_moves().contains(&action) {
            return Err("illegal move or finished game".into());
        }
        let leader = self.position.leader;
        let li = leader.index();
        let fi = leader.other().index();
        if self.pending.is_none() && !matches!(action, Move::Exchange { .. }) {
            self.pending = Some(action);
            return Ok(None);
        }
        let before = self.position.clone();
        let trick = if let Move::Exchange { suit } = action {
            let jack = Card::new(suit, Rank::Jack);
            let old = *self.position.talon.last().expect("validated exchange");
            remove(&mut self.position.hands[li], jack);
            self.position.hands[li].push(old);
            *self.position.talon.last_mut().unwrap() = jack;
            Trick::Exchange {
                leader,
                exchange: action,
                trump_card: old,
            }
        } else {
            let lead = self.pending.take().unwrap();
            let lc = lead.played_card().unwrap();
            let fc = action.played_card().unwrap();
            if let Move::Marriage { suit } = lead {
                self.position.scores[li].pending_points +=
                    if suit == self.position.trump { 40 } else { 20 };
            }
            remove(&mut self.position.hands[li], lc);
            remove(&mut self.position.hands[fi], fc);
            let follower_wins = if lc.suit() == fc.suit() {
                fc.points() > lc.points()
            } else {
                fc.suit() == self.position.trump
            };
            let winner = if follower_wins {
                leader.other()
            } else {
                leader
            };
            let wi = winner.index();
            self.position.won_cards[wi].extend([lc, fc]);
            self.position.scores[wi].direct_points += lc.points() + fc.points();
            self.position.scores[wi] = self.position.scores[wi].redeemed();
            self.position.leader = winner;
            // Drawing precedes terminal detection, even on a winning trick.
            if !self.position.talon.is_empty() {
                let drawn: Vec<_> = self.position.talon.drain(..2).collect();
                self.position.hands[wi].push(drawn[0]);
                self.position.hands[winner.other().index()].push(drawn[1]);
            }
            Trick::Regular {
                leader,
                leader_move: lead,
                follower_move: action,
                winner,
            }
        };
        self.history.push(Frame {
            position: before,
            revealed: self.revealed,
            trick: trick.clone(),
        });
        for card in trick.cards() {
            self.revealed |= card.bit();
        }
        Ok(Some(trick))
    }
    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            position: self.position.clone(),
            current_player: self.current_player(),
            leader_move: self.pending,
            legal_moves: self.legal_moves(),
            phase: self.position.phase(),
            terminal: self.outcome().is_some(),
            winner: self.outcome(),
            tricks: self.history.iter().map(|f| f.trick.clone()).collect(),
        }
    }
    pub fn observation(&self, player: Player) -> PlayerObservation {
        self.observation_recent(player, usize::MAX)
    }
    /// Public observation retaining only the most recent completed history entries.
    pub fn observation_recent(&self, player: Player, history_limit: usize) -> PlayerObservation {
        let terminal = self.outcome().is_some();
        let lead = if player != self.position.leader {
            self.pending
        } else {
            None
        };
        let view = make_view(&self.position, self.revealed, player, lead, terminal, false);
        let mut history: Vec<_> = self
            .history
            .iter()
            .skip(self.history.len().saturating_sub(history_limit))
            .map(|frame| {
                let (lead, exchange) = match &frame.trick {
                    Trick::Regular { leader_move, .. } => (Some(*leader_move), false),
                    Trick::Exchange { .. } => (None, true),
                };
                HistoryEntry {
                    view: make_view(
                        &frame.position,
                        frame.revealed,
                        player,
                        if player == frame.position.leader {
                            None
                        } else {
                            lead
                        },
                        false,
                        exchange,
                    ),
                    trick: Some(frame.trick.clone()),
                }
            })
            .collect();
        history.push(HistoryEntry {
            view: view.clone(),
            trick: None,
        });
        PlayerObservation { view, history }
    }
    /// Privileged testing/diagnostics record; never a bot input.
    pub fn record(&self) -> Record {
        let players = match self.current_player() {
            Some(player) => vec![player],
            None => vec![Player::ZERO, Player::ONE],
        };
        Record {
            state: self.snapshot(),
            observations: players.into_iter().map(|p| self.observation(p)).collect(),
        }
    }
}
fn remove(cards: &mut Vec<Card>, card: Card) {
    let i = cards
        .iter()
        .position(|c| *c == card)
        .expect("legal action card");
    cards.remove(i);
}
fn make_view(
    pos: &Position,
    revealed: u32,
    player: Player,
    lead: Option<Move>,
    terminal: bool,
    exchange: bool,
) -> PlayerView {
    let mine = player.index();
    let opp = player.other().index();
    let is_leader = player == pos.leader;
    let (legal_moves, status) = if terminal {
        (vec![], "game_over")
    } else if is_leader {
        (pos.leader_moves(), "ok")
    } else if exchange {
        (vec![], "ok")
    } else if let Some(lead) = lead {
        (pos.follower_moves(lead), "ok")
    } else {
        (vec![], "no_leader_move")
    };
    let mut seen = revealed;
    for card in &pos.hands[mine] {
        seen |= card.bit();
    }
    if let Some(card) = pos.talon.last() {
        seen |= card.bit();
    }
    if let Some(lead) = lead {
        for card in lead.cards() {
            seen |= card.bit();
        }
    }
    PlayerView {
        player,
        am_i_leader: is_leader,
        hand: pos.hands[mine].clone(),
        my_score: pos.scores[mine],
        opponent_score: pos.scores[opp],
        won_cards: pos.won_cards[mine].clone(),
        opponent_won_cards: pos.won_cards[opp].clone(),
        trump: pos.trump,
        trump_card: pos.talon.last().copied(),
        talon_size: pos.talon.len(),
        phase: pos.phase(),
        known_opponent_cards: pos.hands[opp]
            .iter()
            .copied()
            .filter(|c| pos.phase() == 2 || revealed & c.bit() != 0)
            .collect(),
        opponent_hand: if pos.phase() == 2 {
            Some(pos.hands[opp].clone())
        } else {
            None
        },
        seen_cards: Card::deck()
            .into_iter()
            .filter(|c| seen & c.bit() != 0)
            .collect(),
        leader_move: lead,
        legal_moves,
        legal_moves_status: status.into(),
    }
}

/// Bots receive owned, public observations only.
pub trait Bot {
    fn select_move(&mut self, observation: &PlayerObservation) -> Move;
    fn notify_trump_exchange(&mut self, _exchange: Move) {}
    fn notify_game_end(&mut self, _won: bool, _observation: &PlayerObservation) {}
}
pub fn play_game(deck: [Card; 20], bots: &mut [&mut dyn Bot; 2]) -> Result<Outcome, String> {
    let mut game = Game::from_deck(deck)?;
    while let Some(player) = game.current_player() {
        let action = bots[player.index()].select_move(&game.observation(player));
        if let Some(Trick::Exchange {
            exchange, leader, ..
        }) = game.step(action)?
        {
            bots[leader.index()].notify_trump_exchange(exchange);
            bots[leader.other().index()].notify_trump_exchange(exchange);
        }
    }
    let result = game.outcome().unwrap();
    for player in [result.winner, result.winner.other()] {
        bots[player.index()].notify_game_end(player == result.winner, &game.observation(player));
    }
    Ok(result)
}
