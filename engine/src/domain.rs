use serde::{Deserialize, Serialize};

/// Wire suit order follows the pinned reference deck, not the legacy action tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct Suit(u8);
impl Suit {
    pub const HEARTS: Self = Self(0);
    pub const CLUBS: Self = Self(1);
    pub const SPADES: Self = Self(2);
    pub const DIAMONDS: Self = Self(3);
    pub fn id(self) -> u8 {
        self.0
    }
}
impl TryFrom<u8> for Suit {
    type Error = String;
    fn try_from(v: u8) -> Result<Self, String> {
        if v < 4 {
            Ok(Self(v))
        } else {
            Err("suit must be in 0..4".into())
        }
    }
}
impl From<Suit> for u8 {
    fn from(s: Suit) -> Self {
        s.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rank {
    Jack,
    Queen,
    King,
    Ten,
    Ace,
}
impl Rank {
    fn index(self) -> u8 {
        self as u8
    }
    pub fn points(self) -> u16 {
        [2, 3, 4, 10, 11][self.index() as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct Card(u8);
impl Card {
    pub fn new(suit: Suit, rank: Rank) -> Self {
        Self(suit.0 * 5 + rank.index())
    }
    pub fn id(self) -> u8 {
        self.0
    }
    pub fn suit(self) -> Suit {
        Suit(self.0 / 5)
    }
    pub fn rank(self) -> Rank {
        [Rank::Jack, Rank::Queen, Rank::King, Rank::Ten, Rank::Ace][(self.0 % 5) as usize]
    }
    pub fn points(self) -> u16 {
        self.rank().points()
    }
    pub fn deck() -> [Card; 20] {
        std::array::from_fn(|i| Card(i as u8))
    }
    pub(crate) fn bit(self) -> u32 {
        1 << self.0
    }
}
impl TryFrom<u8> for Card {
    type Error = String;
    fn try_from(v: u8) -> Result<Self, String> {
        if v < 20 {
            Ok(Self(v))
        } else {
            Err("card must be in 0..20".into())
        }
    }
}
impl From<Card> for u8 {
    fn from(c: Card) -> Self {
        c.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct Player(u8);
impl Player {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub fn index(self) -> usize {
        self.0 as usize
    }
    pub fn other(self) -> Self {
        Self(1 - self.0)
    }
}
impl TryFrom<u8> for Player {
    type Error = String;
    fn try_from(v: u8) -> Result<Self, String> {
        if v < 2 {
            Ok(Self(v))
        } else {
            Err("player must be 0 or 1".into())
        }
    }
}
impl From<Player> for u8 {
    fn from(p: Player) -> Self {
        p.0
    }
}

/// A marriage implies a queen/king pair; an exchange always implies a jack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Move {
    Play { card: Card },
    Marriage { suit: Suit },
    Exchange { suit: Suit },
}
impl Move {
    pub fn cards(self) -> Vec<Card> {
        match self {
            Self::Play { card } => vec![card],
            Self::Marriage { suit } => {
                vec![Card::new(suit, Rank::Queen), Card::new(suit, Rank::King)]
            }
            Self::Exchange { suit } => vec![Card::new(suit, Rank::Jack)],
        }
    }
    pub fn played_card(self) -> Option<Card> {
        match self {
            Self::Play { card } => Some(card),
            Self::Marriage { suit } => Some(Card::new(suit, Rank::King)),
            Self::Exchange { .. } => None,
        }
    }
    // The reference follower validator uses the QUEEN, while scoring plays the KING.
    pub(crate) fn follow_target(self) -> Option<Card> {
        match self {
            Self::Marriage { suit } => Some(Card::new(suit, Rank::Queen)),
            _ => self.played_card(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Score {
    pub direct_points: u16,
    pub pending_points: u16,
}
impl Score {
    pub fn redeemed(self) -> Self {
        Self {
            direct_points: self.direct_points + self.pending_points,
            pending_points: 0,
        }
    }
}
impl std::ops::Add for Score {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            direct_points: self.direct_points + rhs.direct_points,
            pending_points: self.pending_points + rhs.pending_points,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub winner: Player,
    pub game_points: u8,
    pub winner_score: Score,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Trick {
    Exchange {
        leader: Player,
        exchange: Move,
        trump_card: Card,
    },
    Regular {
        leader: Player,
        leader_move: Move,
        follower_move: Move,
        winner: Player,
    },
}
impl Trick {
    pub fn cards(&self) -> Vec<Card> {
        match self {
            Self::Exchange {
                exchange,
                trump_card,
                ..
            } => {
                let mut cards = exchange.cards();
                cards.push(*trump_card);
                cards
            }
            Self::Regular {
                leader_move,
                follower_move,
                ..
            } => {
                let mut cards = leader_move.cards();
                cards.extend(follower_move.cards());
                cards
            }
        }
    }
}
