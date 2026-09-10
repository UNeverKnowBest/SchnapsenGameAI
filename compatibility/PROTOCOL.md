# JSON Lines replay protocol (schema 1)

Both commands accept one request per line and produce one response per line:

```text
python compatibility/python_oracle/adapter.py
engine/target/release/schnapsen-engine replay
```

On Windows the executable has an `.exe` suffix. A request contains exactly one `deck` or `position`, plus `actions` (default empty). A deck is a permutation of integers 0–19. Card IDs are `5*suit + rank`: suits HEARTS=0, CLUBS=1, SPADES=2, DIAMONDS=3; ranks JACK=0, QUEEN=1, KING=2, TEN=3, ACE=4. IDs deliberately differ from the legacy tensor's suit order.

```json
{"deck":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],"actions":[{"kind":"play","card":0},{"kind":"play","card":1}]}
```

Moves are `{"kind":"play","card":0}`, `{"kind":"marriage","suit":0}`, or `{"kind":"exchange","suit":3}`. A complete regular trick consumes two actions; exchange consumes one and leaves the same player to act. A pending lead leaves the privileged position unchanged until the follower responds.

A response contains `records`: the initial record and one after every action. Each record has:

* `state`: privileged hands, talon draw order, permanent trump suit, scores and won cards by stable player ID, leader, current player (null at end), pending leader move, ordered legal moves, phase, terminal flag, optional winner/game-points/winner-score, and public completed tricks.
* `observations`: the acting player's public observation, or both players' terminal observations ordered by stable ID. Each has a flat `view` and chronological `history`. History views contain no recursive history or privileged references.

A view contains own hand, both scores, both won-card lists, trump suit/card, talon size, phase, known opponent cards, optional phase-two opponent hand, normalized seen-card set, supplied leader move, ordered legal moves and legal-query status. Status is `ok`, `game_over`, or (Rust's optional nonacting view) `no_leader_move`. Game-over normalizes the reference exception into an explicit status, not a playable empty action set. History's exchange follower has `ok` with zero moves.

Constructed `position` requests use the Position fields in `state` (hands, talon, trump, scores, won_cards, leader); pending moves and histories must be built by replaying actions. Structural validity is enforced; reachability and score-to-won-card consistency are not certified by this constructor. Python and Rust errors have a top-level `error`; exception wording is not compared.

This protocol is privileged testing infrastructure. Send only `PlayerObservation` to an agent. In Rust, `play_game` enforces that boundary through the `Bot` trait.

Replay is deterministic from the supplied deck. Rust evaluation uses its own SplitMix64/Fisher–Yates stream with rejection sampling, with game index mapped to its own stream; it does not claim Python RNG equivalence.
