"""Check the adapter against actual bot callbacks, not just manually created perspectives."""
from collections import deque
import random
import unittest
from adapter import Oracle, ENGINE, FixedBot, decode_move, public_observation, score


class ScriptBot(FixedBot):
    def __init__(self, player, script):
        super().__init__(player)
        self.script = script

    def get_move(self, perspective, leader_move):
        action, expected = self.script.popleft()
        actual = public_observation(perspective, self.player, leader_move)
        if actual != expected:
            raise AssertionError("adapter observation differs from real engine callback")
        return decode_move(action)


class AdapterTests(unittest.TestCase):
    def test_actual_engine_callbacks_match_adapter_trajectories(self):
        for seed in range(100):
            rng = random.Random(seed)
            deck = list(range(20))
            rng.shuffle(deck)
            oracle = Oracle({"deck":deck})
            script = deque()
            exchanges = []
            while oracle.outcome() is None:
                moves = oracle.legal_moves()
                special = [m for m in moves if m["kind"] != "play"]
                action = rng.choice(special or moves)
                script.append((action, oracle.record()["observations"][0]))
                if action["kind"] == "exchange":
                    exchanges.append(action)
                oracle.step(action)
            initial = Oracle({"deck":deck})
            bots = [ScriptBot(i,script) for i in range(2)]
            initial.state.leader.implementation = bots[0]
            initial.state.follower.implementation = bots[1]
            winner, points, winner_score = ENGINE.play_game_from_state(initial.state,None)
            self.assertFalse(script)
            self.assertEqual({"winner":winner.player,"game_points":points,"winner_score":score(winner_score)},oracle.outcome())
            terminal = oracle.record()["observations"]
            for i, bot in enumerate(bots):
                self.assertEqual(bot.exchanges, exchanges)
                self.assertEqual(bot.endings,[(i==winner.player,terminal[i])])

    def test_hidden_state_permutation_is_not_public(self):
        deck = list(range(20))
        altered = deck[:]
        altered[1],altered[12] = altered[12],altered[1]
        a,b = Oracle({"deck":deck}).record(),Oracle({"deck":altered}).record()
        self.assertNotEqual(a["state"],b["state"])
        self.assertEqual(a["observations"],b["observations"])


if __name__ == "__main__":
    unittest.main()
