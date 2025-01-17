from schnapsen.game import *
import torch

class ActionRepresentation:
    def __init__(self):
        """
        Initialize the action space for Schnapsen. This includes:
        - Regular moves for all cards.
        - Special moves like marriages and trump exchanges.
        """
        self.suits = [Suit.HEARTS, Suit.SPADES, Suit.CLUBS, Suit.DIAMONDS]  # All suits
        self.ranks = [Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN, Rank.ACE]  # All ranks
        self.action_map = {}  # Maps actions to unique indices
        self.reverse_action_map = {}  # Maps indices to actions
        self._initialize_actions()

    def _initialize_actions(self):
        """Create the mapping between actions and indices."""
        action_id = 0

        # Regular moves (play any card)
        for suit in self.suits:
            for rank in self.ranks:
                card = Card.get_card(rank, suit)
                move = RegularMove(card)
                self.action_map[move] = action_id
                self.reverse_action_map[action_id] = move
                action_id += 1

        # Marriage moves
        for suit in self.suits:
            queen_card = Card.get_card(Rank.QUEEN, suit)
            king_card = Card.get_card(Rank.KING, suit)
            move = Marriage(queen_card, king_card)
            self.action_map[move] = action_id
            self.reverse_action_map[action_id] = move
            action_id += 1

        # Trump exchange
        for suit in self.suits:
            jack_card = Card.get_card(Rank.JACK, suit)
            move = TrumpExchange(jack_card)
            self.action_map[move] = action_id
            self.reverse_action_map[action_id] = move
            action_id += 1

    def get_action_tensor(self):
        """
        Create an empty action tensor.
        :return: A tensor with zeros, size equal to the total number of actions.
        """
        return torch.zeros(len(self.action_map))

    def encode_action(self, move: Move):
        """
        Encode a move into its corresponding index in the action tensor.
        :param move: The move to encode.
        :return: The index of the move in the tensor.
        """
        return self.action_map.get(move, None)

    def decode_action(self, action_id: int):
        """
        Decode an action index back into a move.
        :param action_id: The index to decode.
        :return: The corresponding move.
        """
        return self.reverse_action_map.get(action_id, None)

    def set_valid_actions(self, valid_moves: list[Move]):
        """
        Create a tensor with valid moves set to 1.
        :param valid_moves: A list of valid moves.
        :return: A tensor indicating valid actions.
        """
        action_tensor = self.get_action_tensor()
        for move in valid_moves:
            action_id = self.encode_action(move)
            if action_id is not None:
                action_tensor[action_id] = 1
        return action_tensor
