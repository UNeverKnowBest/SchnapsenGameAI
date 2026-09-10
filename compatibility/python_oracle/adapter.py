"""Serialization adapter only: all legality, scoring and completed transitions call the pinned engine."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).resolve().parent / "upstream"
MANIFEST = json.loads((ROOT / "compatibility/manifest.json").read_text(encoding="utf-8-sig"))


def verify_reference():
    actual = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if actual != MANIFEST["reference"]["commit"]:
        raise RuntimeError(f"oracle commit mismatch: {actual}")
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--exit-code", "HEAD", "--", "src", "tests", "setup.cfg"],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


verify_reference()
sys.path.insert(0, str(UPSTREAM / "src"))
from schnapsen.deck import Card, Rank, Suit, OrderedCardCollection
from schnapsen.game import (
    Bot, BotState, GameState, Hand, Talon, Score, RegularMove, Marriage, TrumpExchange,
    SchnapsenGamePlayEngine, LeaderPerspective, FollowerPerspective, WinnerPerspective,
    LoserPerspective, GamePhase,
)

# Fail if an already-imported or shadowing distribution displaced the pinned modules.
for name in ('schnapsen.game', 'schnapsen.deck'):
    module_path = Path(sys.modules[name].__file__).resolve()
    expected_path = (UPSTREAM / 'src' / (name.replace('.', '/') + '.py')).resolve()
    if module_path != expected_path:
        raise RuntimeError(f"oracle module shadowed: {name} from {module_path}")

ENGINE = SchnapsenGamePlayEngine()
CARDS = list(ENGINE.deck_generator.get_initial_deck())
CARD_IDS = {card: i for i, card in enumerate(CARDS)}
SUITS = list(Suit)


def cards(collection):
    return [CARD_IDS[c] for c in collection]


def encode_move(move):
    if move is None:
        return None
    if move.is_marriage():
        return {"kind": "marriage", "suit": SUITS.index(move.suit)}
    if move.is_trump_exchange():
        return {"kind": "exchange", "suit": SUITS.index(move.jack.suit)}
    return {"kind": "play", "card": CARD_IDS[move.card]}


def decode_move(value):
    if value["kind"] == "play":
        return RegularMove(CARDS[value["card"]])
    suit = SUITS[value["suit"]]
    if value["kind"] == "marriage":
        return Marriage(Card.get_card(Rank.QUEEN, suit), Card.get_card(Rank.KING, suit))
    if value["kind"] == "exchange":
        return TrumpExchange(Card.get_card(Rank.JACK, suit))
    raise ValueError("unknown move")


def score(value):
    return {"direct_points": value.direct_points, "pending_points": value.pending_points}


def encode_trick(trick, leader, winner):
    if trick.is_trump_exchange():
        return {"kind": "exchange", "leader": leader, "exchange": encode_move(trick.exchange),
                "trump_card": CARD_IDS[trick.trump_card]}
    return {"kind": "regular", "leader": leader, "leader_move": encode_move(trick.leader_move),
            "follower_move": encode_move(trick.follower_move), "winner": winner}


def public_view(perspective, player, leader_move=None):
    # Use public PlayerPerspective methods exclusively here.
    try:
        moves = [encode_move(m) for m in perspective.valid_moves()]
        status = "ok"
    except Exception:
        if not isinstance(perspective, (WinnerPerspective, LoserPerspective)):
            raise
        moves, status = [], "game_over"
    trump_card = perspective.get_trump_card()
    phase_two = perspective.get_phase() == GamePhase.TWO
    return {
        "player": player, "am_i_leader": perspective.am_i_leader(),
        "hand": cards(perspective.get_hand()),
        "my_score": score(perspective.get_my_score()), "opponent_score": score(perspective.get_opponent_score()),
        "won_cards": cards(perspective.get_won_cards()),
        "opponent_won_cards": cards(perspective.get_opponent_won_cards()),
        "trump": SUITS.index(perspective.get_trump_suit()),
        "trump_card": None if trump_card is None else CARD_IDS[trump_card],
        "talon_size": perspective.get_talon_size(), "phase": 2 if phase_two else 1,
        "known_opponent_cards": cards(perspective.get_known_cards_of_opponent_hand()),
        "opponent_hand": cards(perspective.get_opponent_hand_in_phase_two()) if phase_two else None,
        "seen_cards": sorted(cards(perspective.seen_cards(leader_move))),
        "leader_move": encode_move(leader_move), "legal_moves": moves, "legal_moves_status": status,
    }


def public_observation(perspective, player, leader_move=None):
    view = public_view(perspective, player, leader_move)
    history = []
    pairs = perspective.get_game_history()
    for i, (past, trick) in enumerate(pairs):
        if trick is None:
            history.append({"view": view, "trick": None})
            continue
        lead = player if past.am_i_leader() else 1 - player
        next_perspective = pairs[i + 1][0]
        winner = player if next_perspective.am_i_leader() else 1 - player
        move = None if past.am_i_leader() or trick.is_trump_exchange() else trick.leader_move
        history.append({"view": public_view(past, player, move), "trick": encode_trick(trick, lead, winner)})
    return {"view": view, "history": history}


class FixedBot(Bot):
    def __init__(self, player):
        super().__init__(f"seat-{player}")
        self.player = player
        self.move = None
        self.exchanges = []
        self.endings = []

    def get_move(self, perspective, leader_move):
        assert self.move is not None
        return self.move

    def notify_trump_exchange(self, move):
        self.exchanges.append(encode_move(move))

    def notify_game_end(self, won, perspective):
        self.endings.append((won, public_observation(perspective, self.player)))


class Oracle:
    def __init__(self, request):
        self.bots = [FixedBot(0), FixedBot(1)]
        self.pending = None
        if "deck" in request:
            deck = request["deck"]
            if sorted(deck) != list(range(20)):
                raise ValueError("deck must be a 20-card permutation")
            h0, h1, talon = ENGINE.hand_generator.generateHands(OrderedCardCollection(CARDS[c] for c in deck))
            states = [BotState(self.bots[0], h0), BotState(self.bots[1], h1)]
            self.state = GameState(states[0], states[1], talon, None)
        else:
            p = request["position"]
            states = [BotState(self.bots[i], Hand([CARDS[c] for c in p["hands"][i]]),
                               Score(**p["scores"][i]), [CARDS[c] for c in p["won_cards"][i]]) for i in range(2)]
            self.state = GameState(states[p["leader"]], states[1-p["leader"]],
                                   Talon([CARDS[c] for c in p["talon"]], SUITS[p["trump"]]), None)

    def outcome(self):
        result = ENGINE.trick_scorer.declare_winner(self.state)
        if result is None:
            return None
        winner, points = result
        return {"winner": winner.implementation.player, "game_points": points, "winner_score": score(winner.score)}

    def perspective(self, player):
        leader = self.state.leader.implementation.player
        if self.outcome() is not None:
            return WinnerPerspective(self.state, ENGINE) if player == leader else LoserPerspective(self.state, ENGINE)
        if player == leader:
            return LeaderPerspective(self.state, ENGINE)
        return FollowerPerspective(self.state, ENGINE, self.pending)

    def legal_moves(self):
        if self.outcome() is not None:
            return []
        player = self.state.follower.implementation.player if self.pending is not None else self.state.leader.implementation.player
        return [encode_move(m) for m in self.perspective(player).valid_moves()]

    def step(self, action):
        if action not in self.legal_moves():
            raise ValueError(f"illegal semantic action: {action}")
        move = decode_move(action)
        if self.pending is None and not move.is_trump_exchange():
            self.pending = move
            return
        if self.pending is None:
            self.state = ENGINE.trick_implementer.play_trick_with_fixed_leader_move(ENGINE, self.state, move)
        else:
            self.state.follower.implementation.move = move
            self.state = ENGINE.trick_implementer.play_trick_with_fixed_leader_move(ENGINE, self.state, self.pending)
            self.pending = None

    def record(self):
        # Privileged state serialization is intentionally separate from public_observation.
        s = self.state
        players = sorted([s.leader, s.follower], key=lambda b: b.implementation.player)
        outcome = self.outcome()
        leader = s.leader.implementation.player
        current = None if outcome is not None else (1-leader if self.pending is not None else leader)
        tricks = []
        node = s
        while node.previous is not None:
            prev = node.previous
            tricks.append(encode_trick(prev.trick, prev.state.leader.implementation.player, node.leader.implementation.player))
            node = prev.state
        state = {
            "hands": [cards(b.hand) for b in players], "talon": cards(s.talon), "trump": SUITS.index(s.trump_suit),
            "scores": [score(b.score) for b in players], "won_cards": [cards(b.won_cards) for b in players],
            "leader": leader, "current_player": current, "leader_move": encode_move(self.pending),
            "legal_moves": self.legal_moves(), "phase": 2 if s.game_phase() == GamePhase.TWO else 1,
            "terminal": outcome is not None, "winner": outcome, "tricks": list(reversed(tricks)),
        }
        observations = [public_observation(self.perspective(p), p,
                        self.pending if p != leader and outcome is None else None)
                        for p in ([0, 1] if current is None else [current])]
        return {"state": state, "observations": observations}


def replay(request):
    oracle = Oracle(request)
    records = [oracle.record()]
    for action in request.get("actions", []):
        oracle.step(action)
        records.append(oracle.record())
    return records


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            print(json.dumps({"records": replay(json.loads(line))}, separators=(",", ":")), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
