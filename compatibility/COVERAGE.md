# Compatibility coverage and boundaries

The authoritative target is the immutable commit in [manifest.json](manifest.json), selected before Rust implementation. Historical identity remains unconfirmed. The reference's executable behavior takes priority over comments/docstrings.

## Rules that deserve explicit attention

* Marriage plays the **king**, leaving the queen. Follower legality compares against the queen's value. The latter distinction makes no legal-set difference in a unique standard deck (the leader holds the king), but both implementation choices are preserved.
* A marriage always adds pending points, even if its player won earlier tricks. Only winning a later/current completed trick redeems pending points.
* Follower callbacks see the pre-trick hands and scores, with the proposed leader move supplied separately.
* Exchange removes the jack, appends the old bottom card to the leader's hand, preserves leader and talon length, notifies both bots, and constitutes its own history entry.
* Completed tricks remove cards in place, append won cards in leader/follower order, then draw winner first, loser second. Drawing happens before terminal checks.
* At 66 direct points, awards are 3 for an opponent with zero direct points, 2 for 1–32, 1 for 33+. Pending points do not win. Otherwise exhausting all cards awards the last-trick winner 1.
* `seen_cards` is a normalized set. Own/opponent hands, known cards, won cards and legal moves preserve order. Known opponent cards in phase one use **completed** public tricks, not the pending leader move.
* History contains each completed exchange/trick with the viewer's historical perspective and a final current-view/null-trick entry. A historical exchange follower has an empty legal list. End callbacks expose game-over legal-query status.

## Upstream test mapping

All game-relevant assertions from the four core test modules have equivalents below. Tests about Python-specific infrastructure or non-Schnapsen cards are explicitly excluded rather than claimed as ports.

| Pinned upstream tests | Rust equivalent / decision |
| --- | --- |
| `test_schnapsen_implementation.py::DeckGenerationTest` | `deck_card_and_move_domain`, `alternating_deal_order_and_trump` |
| `DealingTest.test_dealing` | `alternating_deal_order_and_trump`, `alternating_deal` golden fixture, every generated initial state |
| `test_game.py::MoveTest` creation/type/card assertions | `deck_card_and_move_domain`: typed variants make mismatched marriage partners and non-jack exchanges unrepresentable; malformed wire inputs rejected |
| `HandTest` creation/capacity/empty | `invalid_positions_are_rejected`, full-game conservation and terminal tests |
| `HandTest` remove/add/order/has-cards | Golden marriage/exchange/draw cases, `invalid_actions_are_atomic`, generated state comparisons |
| `HandTest.test_copy`, `GameTest.test_BotState` copy data | `observations_and_clones_are_owned` |
| `HandTest` suit/rank filters | `must_beat`, `must_follow_lower`, trump fixtures, marriage legal enumeration and ordered differential comparisons |
| `TalonTest` bottom/trump/exchange/draw | Deal, exchange and last-draw fixtures; invalid-position checks |
| `TalonTest.test_overdraw_cards` | No arbitrary draw API: only validated two-card draws; size checks and generated conservation cover engine behavior |
| `ScoreTest` add/redeem | `scores_add_and_redeem_nonnegative_domain`, pending-marriage fixtures. Negative/arbitrary signed scores are outside valid games |
| `GameTest.test_GameState` unfinished state | Initial/partial/terminal fixture assertions and generated trajectories |
| `GameTest.test_LeaderGameState`, `test_FollowerGameState` | Ordered legal moves and own-hand golden views; separate observation differential checks |
| `GameTest.test_marriage_point` | Upstream is a **pass-only placeholder**; replaced by losing/redeeming/prior-win/royal marriage fixtures |
| `test_deck.py::CardTest` identity/rank/suit | `deck_card_and_move_domain` for the 20 Schnapsen cards. Unicode glyphs and other 32 cards excluded |
| `CollectionTest` empty/iteration/filter/membership | Standard Rust owned vectors, deal/order/filter/conservation coverage above. Generic 52-card/duplicate multiset API excluded |
| `test_repr.py` eval(repr(...)) | Python repr/eval is excluded. Typed JSON golden record serialization and protocol tests cover the replacement interchange contract |
| `tests/bots/test_randbot.py` 1000 complete games | 2000 Rust invariant games plus 10000 differential complete games |
| Other `tests/bots/` search algorithms | Out of scope: Rdeep, minimax and alpha-beta implementations, not engine rules |

One upstream marriage-invalid-construction test contains a tuple in its condition and therefore does not exercise the intended cases. Rust's typed move representation and invalid-wire tests supply real coverage.

## Separate checks

`differential/run.py --mode states` compares privileged engine records, including semantic legal sets **and** ordered legal lists. `--mode observations` compares public observations and every historical public view. Default `all` executes both comparisons independently and counts them separately.

The oracle adapter serializes privileged fields only in `Oracle.record`. `public_view` and `public_observation` use public perspective methods exclusively. `test_adapter.py` verifies adapter traces against actual upstream `play_game_from_state` bot callbacks for 100 complete games, including exchanges/endings.

Golden fixture expectations are checked in. `fixtures/freeze.py` is an explicit update command and never runs automatically during tests. It verifies handwritten expectations before writing. Constructed fixture positions may inject scores/won zones with history reset; these isolate rules and are not claims that every fixture is reachable from a deal. Complete generated trajectories supply that separate evidence.

The frozen contract covers standard, valid game behavior. Rust deliberately rejects duplicate/missing cards, wrong trump, malformed moves and unsupported deck sizes. It does not reproduce Python exception strings, mutable aliases, arbitrary invalid states, debug repr leaks, RNG sequences, or Python class hierarchy.

Zero mismatches in generated runs is strong finite evidence, not a formal proof over every trajectory. Aggregate win rate is never used as evidence of equivalence.
