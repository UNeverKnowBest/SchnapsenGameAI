"""Handwritten starting positions/actions; expected assertions are independent of either engine."""
def play(card):
    return {"kind": "play", "card": card}


def marriage(suit):
    return {"kind": "marriage", "suit": suit}


def exchange(suit):
    return {"kind": "exchange", "suit": suit}


def position(h0, h1, trump=3, talon=(), direct=(0, 0), pending=(0, 0), leader=0):
    used = h0 + h1 + list(talon)
    assert len(used) == len(set(used))
    return {
        "hands": [h0, h1], "talon": list(talon), "trump": trump, "leader": leader,
        "scores": [{"direct_points": direct[i], "pending_points": pending[i]} for i in range(2)],
        "won_cards": [[c for c in range(20) if c not in used], []],
    }


def deal(h0, h1, bottom=19, direct=(0, 0)):
    used = h0 + h1
    rest = [c for c in range(20) if c not in used and c != bottom] + [bottom]
    return position(h0, h1, bottom // 5, rest, direct)


def scenarios():
    cases = []
    def add(name, p, actions, assertions):
        cases.append({"name": name, "request": {"position": p, "actions": actions}, "assertions": assertions})
    # Assertion path starts at a record index (-1 is the final record).
    add("alternating_deal", None, [], [("0.state.hands", [[0,2,4,6,8],[1,3,5,7,9]])])
    cases[-1]["request"] = {"deck": list(range(20)), "actions": []}
    p = deal([1,2,4,8,10], [3,5,6,11,15])
    add("phase_one_freedom", p, [play(4)], [("-1.state.legal_moves", [play(c) for c in p["hands"][1]])])
    add("marriage_king_pending_then_redeem", p, [marriage(0),play(3),play(5),play(8)],
        [("1.state.scores.0.pending_points",0), ("1.state.hands.0",p["hands"][0]),
         ("2.state.scores.0.pending_points",20), ("2.state.won_cards.1",[2,3]),
         ("-1.state.scores.0.direct_points",32),("-1.state.scores.0.pending_points",0)])
    p_prior = deal([1,2,4,8,10], [3,5,6,11,15], direct=(20,0))
    add("marriage_after_previous_win_still_pending",p_prior,[marriage(0),play(3)],
        [("-1.state.scores.0", {"direct_points":20,"pending_points":20})])
    royal = deal([16,17,4,8,10],[15,0,5,6,11])
    add("royal_marriage_40",royal,[marriage(3),play(15)],
        [("-1.state.scores.0.direct_points",46),("-1.state.scores.0.pending_points",0),
         ("-1.state.won_cards.0",[17,15])])
    ex = deal([15,1,2,4,8],[0,3,5,6,10])
    add("exchange_keeps_leader_and_reveals_card",ex,[exchange(3)],
        [("-1.state.current_player",0),("-1.state.talon.-1",15),
         ("-1.state.hands.0",[1,2,4,8,19]),("-1.state.scores.0.direct_points",0)])
    add("exchange_follower_history",ex,[exchange(3),play(4)],
        [("-1.observations.0.view.known_opponent_cards",[19]),
         ("-1.observations.0.history.0.view.legal_moves",[])])
    p_last = position([4,6,7,10,11],[0,1,2,5,12],talon=[18,19])
    add("last_draw_loser_gets_faceup",p_last,[play(4),play(0)],
        [("-1.state.phase",2),("-1.state.hands.0.-1",18),("-1.state.hands.1.-1",19),
         ("-1.observations.0.view.trump_card",None)])
    add("winning_trick_draws_before_terminal",position([4,6,7,10,11],[0,1,2,5,12],talon=[18,19],direct=(60,0)),
        [play(4),play(0)],[("-1.state.terminal",True),("-1.state.phase",2),
         ("-1.state.winner.game_points",3),("-1.state.hands.1.-1",19),
         ("-1.observations.0.view.legal_moves_status","game_over"),
         ("-1.observations.1.view.legal_moves_status","game_over")])
    for name, lead, h1, legal in [
        ("must_beat",1,[0,2,4],[2,4]), ("must_follow_lower",4,[0,1,15],[0,1]),
        ("must_trump",4,[5,15,19],[15,19]), ("free_discard_without_suit_or_trump",4,[5,6,10],[5,6,10]),
        ("must_overtrump",16,[15,17,19],[17,19]), ("must_undertrump",19,[15,16,5],[15,16]),
        ("void_trump_free_discard",19,[0,5,10],[0,5,10]),
    ]:
        others = [c for c in range(20) if c not in h1 and c != lead][:2]
        add(name,position([lead]+others,h1),[play(lead)],[("-1.state.legal_moves",[play(c) for c in legal])])
    add("phase_two_marriage_plays_king",position([1,2,5],[0,3,6]),[marriage(0),play(3)],
        [("1.state.legal_moves",[play(3)]),("-1.state.won_cards.1",[2,3])])
    for loser_score, award in [(0,3),(1,2),(32,2),(33,1)]:
        add(f"award_threshold_{loser_score}",position([4,6],[0,5],direct=(53,loser_score)),
            [play(4),play(0)],[("-1.state.winner.game_points",award),("-1.state.winner.winner_score.direct_points",66)])
    add("pending_points_do_not_win",position([4,6],[0,5],pending=(80,0)),[],
        [("-1.state.terminal",False)])
    add("last_trick_wins_below_66",position([0],[4],direct=(60,47)),[play(0),play(4)],
        [("-1.state.winner.winner",1),("-1.state.winner.game_points",1),
         ("-1.state.winner.winner_score.direct_points",60)])
    add("terminal_threshold_precedes_last_trick",position([4],[0],direct=(53,0)),[play(4),play(0)],
        [("-1.state.winner.game_points",3)])
    return cases


def assert_expected(records, assertions):
    for path, expected in assertions:
        value = records
        for part in path.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
        assert value == expected, f"{path}: expected {expected!r}, got {value!r}"
