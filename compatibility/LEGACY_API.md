# Historical consumer audit

The Python agent files remain in their original locations. They are consumers, not the rules specification.

* `main.py` creates `SchnapsenGamePlayEngine`, calls `play_game(bot1, bot2, Random())`, and consumes `(winner_bot, game_points, winner_score)`. Evaluation plays 100 games against `RandBot`, with the tested bot always initially leading, and counts bot identity wins. This is policy evaluation, not a compatibility test.
* `DQN_bot.py` and `DQNMultithread.py` subclass `Bot`; `get_move(perspective, leader_move, eta)` requires an extra argument absent from the reference's two-argument callback. Nothing in this repository proves a modified engine existed. A future adapter must handle this explicitly.
* `notify_game_end(won, perspective)` gives reward +1/-1 and resets replay state. The reference also notifies both bots of trump exchanges. Rust offers `Bot` callbacks and a standalone `Game`; bots are not embedded in states.
* `ActionRepresentation.py` uses 28 actions: 20 regular cards, four queen/king marriages, four jack exchanges. Its suit order HEARTS, SPADES, CLUBS, DIAMONDS differs from the reference deck order. An adapter must map semantically, never reuse IDs blindly.
* `feacture.py` consumes own hand, both direct/pending scores, trump suit/card, talon size, phase, leader status, own/opponent won cards, known opponent cards, ordered legal moves, and perspective/trick history. It encodes the supplied leader move separately. Its history winner encoding is a placeholder returning `[0, 0]`.
* Opponent hand is available in phase two. Phase-one known opponent cards derive from completed public tricks, including marriage partners/exchanged cards. Current leader cards are exposed separately through the supplied move; they are not yet incorporated into past-trick knowledge.

Rust `PlayerObservation` covers those observable data through owned values. It does not expose privileged state pointers. Python's `get_engine`, `make_assumption`, `get_state_in_phase_two` and bot-search convenience wrappers are not reproduced as a Python API; consumers can build search/determinization separately. No legacy runtime adapter or Python extension is implemented in this phase.
