import torch
from torch import tensor
from typing import Optional
from schnapsen.game import *

# 为了避免在每次调用时重复构造，可预先构建常量映射字典（也可直接在函数中构造）
_ONE_HOT_SUITS = {
    Suit.HEARTS: torch.tensor([0, 0, 0, 1], dtype=torch.float32),
    Suit.CLUBS: torch.tensor([0, 0, 1, 0], dtype=torch.float32),
    Suit.SPADES: torch.tensor([0, 1, 0, 0], dtype=torch.float32),
    Suit.DIAMONDS: torch.tensor([1, 0, 0, 0], dtype=torch.float32)
}

_ONE_HOT_RANKS = {
    Rank.ACE:    torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32),
    Rank.TWO:    torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32),
    Rank.THREE:  torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32),
    Rank.FOUR:   torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
    Rank.FIVE:   torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32),
    Rank.SIX:    torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
    Rank.SEVEN:  torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    Rank.EIGHT:  torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    Rank.NINE:   torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    Rank.TEN:    torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    Rank.JACK:   torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    Rank.QUEEN:  torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    Rank.KING:   torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
}


def get_state_feature(perspective: PlayerPerspective,
                      leader_move: Optional[Move]) -> torch.Tensor:
    """
    由于“没有 follower move”，所以只接收 leader_move。
    直接使用原生tensor操作构建状态向量，并返回 float32 类型的张量。
    """
    player_state_tensor = get_state_feature_vector(perspective)  # shape: [445]
    leader_move_tensor = get_move_feature_vector(leader_move)      # shape: [20]
    # 拼接两部分
    full_state = torch.cat([player_state_tensor, leader_move_tensor])
    return full_state  # shape: [445 + 20 = 465]


def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> torch.Tensor:
    """
    返回4维one-hot张量表示花色。
    """
    if card_suit in _ONE_HOT_SUITS:
        return _ONE_HOT_SUITS[card_suit]
    else:
        raise ValueError("Suit of card was not found!")


def get_one_hot_encoding_of_card_rank(card_rank: Rank) -> torch.Tensor:
    """
    返回13维one-hot张量表示牌的级别。
    """
    if card_rank in _ONE_HOT_RANKS:
        return _ONE_HOT_RANKS[card_rank]
    else:
        raise AssertionError("Provided card Rank does not exist!")


def get_move_feature_vector(move: Optional[Move]) -> torch.Tensor:
    """
    如果 move 为 None，则返回20维零张量；
    否则返回 one-hot 表示：move_type (3) + rank (13) + suit (4) = 20 维。
    """
    if move is None:
        return torch.zeros(20, dtype=torch.float32)
    else:
        if move.is_marriage():
            move_type = torch.tensor([0, 0, 1], dtype=torch.float32)
            card = move.queen_card
        elif move.is_trump_exchange():
            move_type = torch.tensor([0, 1, 0], dtype=torch.float32)
            card = move.jack
        else:
            move_type = torch.tensor([1, 0, 0], dtype=torch.float32)
            card = move.card

        rank_tensor = get_one_hot_encoding_of_card_rank(card.rank)  # 13维
        suit_tensor = get_one_hot_encoding_of_card_suit(card.suit)    # 4维
        return torch.cat([move_type, rank_tensor, suit_tensor])  # 共 3+13+4 = 20 维


def normalize_scalar_features(features: torch.Tensor, max_values: torch.Tensor) -> torch.Tensor:
    """
    假设 features 与 max_values 都是 torch.Tensor（形状相同），返回归一化后的张量。
    """
    # 避免除以零，这里用 torch.where 作处理
    normalized = torch.where(max_values > 0, features.float() / max_values.float(), torch.zeros_like(features, dtype=torch.float32))
    return normalized


def get_state_feature_vector(perspective: PlayerPerspective) -> torch.Tensor:
    """
    使用原生 Tensor 操作构造状态向量（不包含 leader_move部分），按照以下部分组合：
      1) 标量特征 (5维)
      2) Trump suit (4维)
      3) 游戏阶段 (2维)
      4) Leader indicator (2维)
      5) Deck knowledge (20张牌，每张6维 → 120维)
      6) 历史回合信息 (5 回合，每回合 leader_move (20维) + winner (2维) → 110维)
      7) 当前合法动作 (最多10个，每个20维 → 200维)
      8) 额外特征 (2维)
    
    总计：5+4+2+2+120+110+200+2 = 445 维
    """
    feats = []

    # (1) 标量特征
    my_score = perspective.get_my_score()
    opp_score = perspective.get_opponent_score()
    # 构造标量：直接放入一个 tensor 中
    scalars = torch.tensor([
        my_score.direct_points, my_score.pending_points,
        opp_score.direct_points, opp_score.pending_points,
        perspective.get_talon_size()
    ], dtype=torch.float32)
    max_vals = torch.tensor([66, 66, 66, 66, 20], dtype=torch.float32)
    feats.append(normalize_scalar_features(scalars, max_vals))  # shape: [5]

    # (2) Trump suit
    feats.append(get_one_hot_encoding_of_card_suit(perspective.get_trump_suit()))  # [4]

    # (3) 游戏阶段
    if perspective.get_phase() == GamePhase.TWO:
        feats.append(torch.tensor([1, 0], dtype=torch.float32))
    else:
        feats.append(torch.tensor([0, 1], dtype=torch.float32))

    # (4) Leader indicator
    if perspective.am_i_leader():
        feats.append(torch.tensor([0, 1], dtype=torch.float32))
    else:
        feats.append(torch.tensor([1, 0], dtype=torch.float32))

    # (5) Deck knowledge: 对20张牌，每张6维 → 120维
    deck_encoding = []
    hand_cards = perspective.get_hand().cards
    trump_card = perspective.get_trump_card()
    my_won = perspective.get_won_cards().get_cards()
    opp_won = perspective.get_opponent_won_cards().get_cards()
    opp_known = perspective.get_known_cards_of_opponent_hand().get_cards()

    for card in SchnapsenDeckGenerator().get_initial_deck():
        if card in hand_cards:
            enc = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32)
        elif card in my_won:
            enc = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32)
        elif card in opp_known:
            enc = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32)
        elif card in opp_won:
            enc = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32)
        elif card == trump_card:
            enc = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)
        else:
            enc = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32)
        deck_encoding.append(enc)
    deck_tensor = torch.cat(deck_encoding)  # shape: [20*6 = 120]
    feats.append(deck_tensor)

    # (6) 历史回合信息：5回合，每回合 leader_move (20) + winner (2) = 22维，共 110维
    history_list = []
    max_history_rounds = 5
    game_history = perspective.get_game_history()
    for i in range(max_history_rounds):
        idx = -(i + 2)
        if len(game_history) + idx >= 0:
            _, trick = game_history[idx]
            if trick and not trick.is_trump_exchange():
                leader_move_tensor = get_move_feature_vector(trick.leader_move)  # [20]
                winner_tensor = torch.tensor(determine_winner_of_trick(trick, perspective), dtype=torch.float32)  # [2]
                history_list.append(torch.cat([leader_move_tensor, winner_tensor]))  # [22]
            else:
                history_list.append(torch.zeros(22, dtype=torch.float32))
        else:
            history_list.append(torch.zeros(22, dtype=torch.float32))
    history_tensor = torch.cat(history_list)  # [5*22 = 110]
    feats.append(history_tensor)

    # (7) 当前合法动作：最多10个，每个20维 → 200维
    legal_moves = perspective.valid_moves()
    actions_list = []
    max_actions = 10
    for i in range(max_actions):
        if i < len(legal_moves):
            actions_list.append(get_move_feature_vector(legal_moves[i]))  # [20]
        else:
            actions_list.append(torch.zeros(20, dtype=torch.float32))
    actions_tensor = torch.cat(actions_list)  # [200]
    feats.append(actions_tensor)

    # (8) 额外特征：能否 exchange jack 与能否打出 trump → 2维
    exch = 1 if can_exchange_jack_now(perspective) else 0
    trump_flag = 1 if can_play_trump_now(perspective) else 0
    extras = torch.tensor([exch, trump_flag], dtype=torch.float32)  # [2]
    feats.append(extras)

    # 将所有部分串联起来
    full_state = torch.cat(feats)  # 5 + 4 + 2 + 2 + 120 + 110 + 200 + 2 = 445 维
    return full_state


def determine_winner_of_trick(trick: Trick, perspective: PlayerPerspective) -> [int]:
    """
    此处仅作为示例。实际需要根据 trick 或游戏引擎返回赢家信息。
    这里固定返回 [0, 0]。
    """
    return [0, 0]


def can_exchange_jack_now(perspective: PlayerPerspective) -> bool:
    if not perspective.am_i_leader():
        return False
    if perspective.get_phase() != GamePhase.ONE:
        return False
    if perspective.get_talon_size() < 2:
        return False
    jack_card = Card.get_card(Rank.JACK, perspective.get_trump_suit())
    return jack_card in perspective.get_hand().cards


def can_play_trump_now(perspective: PlayerPerspective) -> bool:
    tr_suit = perspective.get_trump_suit()
    for c in perspective.get_hand().cards:
        if c.suit == tr_suit:
            return True
    return False
