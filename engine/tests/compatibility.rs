use schnapsen_engine::evaluation::{Random, random_games};
use schnapsen_engine::*;
use serde_json::Value;

fn card(id: u8) -> Card {
    Card::try_from(id).unwrap()
}
fn play(id: u8) -> Move {
    Move::Play { card: card(id) }
}

#[test]
fn handwritten_reference_fixtures() {
    let golden: Value = serde_json::from_str(include_str!(
        "../../compatibility/fixtures/handwritten.json"
    ))
    .unwrap();
    assert_eq!(
        golden["reference"],
        "ca0b3d9cd9c3922a10303536e28f1266fe3a2c0d"
    );
    for fixture in golden["fixtures"].as_array().unwrap() {
        let req = &fixture["request"];
        let mut game = if req.get("deck").is_some() {
            Game::from_deck(serde_json::from_value(req["deck"].clone()).unwrap()).unwrap()
        } else {
            Game::from_position(serde_json::from_value(req["position"].clone()).unwrap()).unwrap()
        };
        let mut records = vec![game.record()];
        for action in req["actions"].as_array().unwrap() {
            game.step(serde_json::from_value(action.clone()).unwrap())
                .unwrap();
            records.push(game.record());
        }
        assert_eq!(
            serde_json::to_value(records).unwrap(),
            fixture["records"],
            "fixture {}",
            fixture["name"]
        );
    }
}
#[test]
fn deck_card_and_move_domain() {
    let deck = Card::deck();
    for (id, c) in deck.into_iter().enumerate() {
        assert_eq!(c.id() as usize, id);
        assert_eq!(Card::new(c.suit(), c.rank()), c);
        assert_eq!(c.points(), [2, 3, 4, 10, 11][id % 5]);
        assert_eq!(serde_json::from_str::<Card>(&id.to_string()).unwrap(), c);
    }
    for suit in [Suit::HEARTS, Suit::CLUBS, Suit::SPADES, Suit::DIAMONDS] {
        let marriage = Move::Marriage { suit };
        assert_eq!(
            marriage.cards(),
            vec![Card::new(suit, Rank::Queen), Card::new(suit, Rank::King)]
        );
        assert_eq!(marriage.played_card(), Some(Card::new(suit, Rank::King)));
        let exchange = Move::Exchange { suit };
        assert_eq!(exchange.cards(), vec![Card::new(suit, Rank::Jack)]);
        assert_eq!(exchange.played_card(), None);
    }
    assert!(serde_json::from_str::<Card>("20").is_err());
    assert!(serde_json::from_str::<Card>("-1").is_err());
    assert!(serde_json::from_str::<Move>(r#"{"kind":"marriage","suit":4}"#).is_err());
    assert!(serde_json::from_str::<Move>(r#"{"kind":"exchange","card":3}"#).is_err());
    assert!(serde_json::from_str::<Move>(r#"{"kind":"play","card":1,"extra":2}"#).is_err());
    assert!(Player::try_from(2).is_err());
}
#[test]
fn alternating_deal_order_and_trump() {
    let game = Game::from_deck(Card::deck()).unwrap();
    let pos = game.snapshot().position;
    assert_eq!(pos.hands[0], [0, 2, 4, 6, 8].map(card));
    assert_eq!(pos.hands[1], [1, 3, 5, 7, 9].map(card));
    assert_eq!(pos.talon, (10..20).map(card).collect::<Vec<_>>());
    assert_eq!(pos.trump, Suit::DIAMONDS);
}
#[test]
fn invalid_positions_are_rejected() {
    let mut deck = Card::deck();
    deck[19] = deck[0];
    assert!(Game::from_deck(deck).is_err());
    let original = Game::from_deck(Card::deck()).unwrap().snapshot().position;
    let mut p = original.clone();
    p.trump = Suit::HEARTS;
    assert!(Game::from_position(p).is_err());
    let mut p = original.clone();
    p.talon.pop();
    assert!(Game::from_position(p).is_err());
    let mut p = original.clone();
    p.hands[0].push(card(10));
    assert!(Game::from_position(p).is_err());
    let mut p = original.clone();
    p.scores[1].direct_points = 66;
    assert!(Game::from_position(p).is_err());
    let mut p = original;
    p.scores[0].pending_points = u16::MAX;
    assert!(Game::from_position(p).is_err());
}
#[test]
fn invalid_actions_are_atomic() {
    let mut game = Game::from_deck(Card::deck()).unwrap();
    for action in [
        play(19),
        Move::Marriage { suit: Suit::HEARTS },
        Move::Exchange { suit: Suit::HEARTS },
    ] {
        let before = game.record();
        assert!(game.step(action).is_err());
        assert_eq!(game.record(), before);
    }
    game.step(play(0)).unwrap();
    let before = game.record();
    assert!(game.step(Move::Marriage { suit: Suit::CLUBS }).is_err());
    assert_eq!(game.record(), before);
}
#[test]
fn observations_and_clones_are_owned() {
    let game = Game::from_deck(Card::deck()).unwrap();
    let before = game.record();
    let mut view = game.observation(Player::ZERO);
    view.view.hand.clear();
    view.history.clear();
    let mut copied = game.clone();
    copied.step(play(0)).unwrap();
    copied.step(play(1)).unwrap();
    assert_eq!(game.record(), before);
    assert_ne!(copied.record(), before);
}
#[test]
fn phase_one_hidden_permutations_do_not_change_observations() {
    let a = Card::deck();
    // Opponent card and hidden talon card change, player's hand and face-up card stay fixed.
    let mut b = a;
    b.swap(1, 12);
    let ga = Game::from_deck(a).unwrap();
    let gb = Game::from_deck(b).unwrap();
    assert_ne!(ga.snapshot(), gb.snapshot());
    assert_eq!(ga.observation(Player::ZERO), gb.observation(Player::ZERO));
    let mut b = a;
    b.swap(2, 12);
    let mut ga = Game::from_deck(a).unwrap();
    let mut gb = Game::from_deck(b).unwrap();
    ga.step(play(0)).unwrap();
    gb.step(play(0)).unwrap();
    assert_eq!(ga.observation(Player::ONE), gb.observation(Player::ONE));
    let json = serde_json::to_value(ga.observation(Player::ONE)).unwrap();
    assert!(json["view"].get("talon").is_none());
    assert!(json["view"]["opponent_hand"].is_null());
}
#[test]
fn scores_add_and_redeem_nonnegative_domain() {
    for direct in 0..66 {
        for pending in [0, 20, 40, 60, 80, 120] {
            let a = Score {
                direct_points: direct,
                pending_points: pending,
            };
            let b = Score {
                direct_points: 3,
                pending_points: 20,
            };
            assert_eq!(a + b, b + a);
            assert_eq!((a + b).direct_points, direct + 3);
            assert_eq!((a + b).pending_points, pending + 20);
            assert_eq!(
                a.redeemed(),
                Score {
                    direct_points: direct + pending,
                    pending_points: 0
                }
            );
        }
    }
}
#[test]
fn all_distinct_card_pair_winners_and_points() {
    for trump in [Suit::HEARTS, Suit::CLUBS, Suit::SPADES, Suit::DIAMONDS] {
        for a in Card::deck() {
            for b in Card::deck() {
                if a == b {
                    continue;
                }
                let p = Position {
                    hands: [vec![a], vec![b]],
                    talon: vec![],
                    trump,
                    leader: Player::ZERO,
                    scores: [Score::default(); 2],
                    won_cards: [
                        Card::deck()
                            .into_iter()
                            .filter(|c| *c != a && *c != b)
                            .collect(),
                        vec![],
                    ],
                };
                let mut game = Game::from_position(p).unwrap();
                game.step(Move::Play { card: a }).unwrap();
                game.step(Move::Play { card: b }).unwrap();
                let outcome = game.outcome().unwrap();
                let winner = if a.suit() == b.suit() {
                    if a.id() % 5 > b.id() % 5 {
                        Player::ZERO
                    } else {
                        Player::ONE
                    }
                } else if b.suit() == trump {
                    Player::ONE
                } else {
                    Player::ZERO
                };
                assert_eq!(outcome.winner, winner);
                assert_eq!(outcome.winner_score.direct_points, a.points() + b.points());
                assert_eq!(outcome.game_points, 1);
                assert!(game.step(Move::Play { card: a }).is_err());
            }
        }
    }
}
#[test]
fn generated_games_conserve_cards_and_terminate() {
    for seed in 0..2000 {
        let mut rng = Random::new(seed);
        let mut game = Game::from_deck(rng.deck()).unwrap();
        let mut steps = 0;
        while game.current_player().is_some() {
            let legal = game.legal_moves();
            game.step(legal[rng.index(legal.len())]).unwrap();
            let p = game.snapshot().position;
            let mut all: Vec<_> = p
                .hands
                .iter()
                .flatten()
                .chain(&p.talon)
                .chain(p.won_cards.iter().flatten())
                .copied()
                .collect();
            all.sort();
            assert_eq!(all, Card::deck());
            assert!(p.hands.iter().all(|h| h.len() <= 5));
            assert!(p.talon.len().is_multiple_of(2));
            let card_points: u16 = p.won_cards.iter().flatten().map(|c| c.points()).sum();
            let direct: u16 = p.scores.iter().map(|s| s.direct_points).sum();
            assert!(direct >= card_points);
            assert_eq!((direct - card_points) % 20, 0);
            steps += 1;
            assert!(steps <= 22);
        }
        for player in [Player::ZERO, Player::ONE] {
            assert_eq!(
                game.observation(player).view.legal_moves_status,
                "game_over"
            );
        }
    }
}
#[test]
fn independent_workers_produce_identical_statistics() {
    assert_eq!(
        random_games(2000, 42, 1).unwrap(),
        random_games(2000, 42, 4).unwrap()
    );
    assert_eq!(random_games(0, 42, 4).unwrap().games, 0);
    assert!(random_games(1, 42, 0).is_err());
}
#[test]
fn callbacks_receive_only_public_views_and_fire_once() {
    use std::{cell::RefCell, rc::Rc};
    struct TestBot {
        id: usize,
        log: Rc<RefCell<Vec<String>>>,
    }
    impl Bot for TestBot {
        fn select_move(&mut self, o: &PlayerObservation) -> Move {
            assert_eq!(o.view.player.index(), self.id);
            assert_eq!(o.view.legal_moves_status, "ok");
            if o.view.phase == 1 {
                assert!(o.view.opponent_hand.is_none());
            }
            *o.view
                .legal_moves
                .iter()
                .find(|m| matches!(m, Move::Exchange { .. }))
                .unwrap_or(&o.view.legal_moves[0])
        }
        fn notify_trump_exchange(&mut self, _: Move) {
            self.log.borrow_mut().push(format!("exchange-{}", self.id));
        }
        fn notify_game_end(&mut self, won: bool, o: &PlayerObservation) {
            assert_eq!(o.view.legal_moves_status, "game_over");
            self.log.borrow_mut().push(format!("end-{}-{won}", self.id));
        }
    }
    let log = Rc::new(RefCell::new(vec![]));
    let mut b0 = TestBot {
        id: 0,
        log: log.clone(),
    };
    let mut b1 = TestBot {
        id: 1,
        log: log.clone(),
    };
    let mut deck = Card::deck();
    deck.swap(0, 15);
    let outcome = play_game(deck, &mut [&mut b0, &mut b1]).unwrap();
    let entries = log.borrow();
    assert_eq!(&entries[..2], ["exchange-0", "exchange-1"]);
    let ends: Vec<_> = entries.iter().filter(|s| s.starts_with("end")).collect();
    assert_eq!(ends.len(), 2);
    assert_eq!(*ends[0], format!("end-{}-true", outcome.winner.index()));
    assert_eq!(
        *ends[1],
        format!("end-{}-false", outcome.winner.other().index())
    );
}
