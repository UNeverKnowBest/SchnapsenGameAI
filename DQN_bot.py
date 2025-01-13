import random
from collections import deque
from typing import List
import numpy as np

from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase
import torch
from feacture import *

class DQN_bot(Bot):

    def __init__(self, name , dqn1, dqn2, replay_buffer_1, optimizer_1, optimizer_2,epsilon):
        super().__init__(name)
        self.my_last_score = 0
        self.opponent_last_score = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.current_model = dqn1
        self.target_model = dqn2
        self.replay_buffer = replay_buffer_1
        self.last_state = None
        self.last_action = None
        self.rl_optimizer = optimizer_1
        self.sl_optimizer = optimizer_2
        self.epsilon = epsilon


    def get_move(self, perspective: PlayerPerspective, leader_move) -> Move:
        # Get the state representation
        action_represnetation = ActionRepresentation()
        state_representation = get_state_feature_vector(perspective)
        point_get = perspective.get_my_score().direct_points - self.my_last_score
        self.last_score = perspective.get_my_score().direct_points
        opponent_point_get = perspective.get_opponent_score().direct_points - self.opponent_last_score
        self.opponent_last_score = perspective.get_opponent_score().direct_points
        reward = (point_get - opponent_point_get) / 100

        valid_moves = perspective.valid_moves()
        if random.random() > self.epsilon:
            # Calculate Q values, set -inf to illegal moves
            state_tensor = torch.tensor(state_representation, dtype=torch.float32, device=self.device)
            q_values = self.current_model(state_tensor)
            action_tensor = action_represnetation.set_valid_actions(valid_moves).to(self.device)
            q_values_mask = torch.mul(q_values, action_tensor)
            best_index = torch.argmax(q_values_mask).item()
            move = action_represnetation.decode_action(best_index)
        else:
            move = random.choice(valid_moves)
            best_index = action_represnetation.encode_action(move)

        if move not in valid_moves:
            move = random.choice(valid_moves)
            
        self.reward = reward
        self.after_trick(perspective)
        self.last_action = best_index
        self.last_state = state_representation
        return move

    def notify_game_end(self, won, perspective):
        if self.last_state is None:
            return
        
        state = get_state_feature_vector(perspective)
        last_state = self.last_state
        last_action = self.last_action
        last_reward = self.reward
        self.replay_buffer.push(last_state, last_action, last_reward, state , done=False)
        state_final = [0.0] * 133
        reward = 1 if won else -1
        self.replay_buffer.push(state, None, 0, state_final, done=True)
    
    def after_trick(self,perspective: PlayerPerspective):
        if self.last_state is None:
            return
        
        done = False
        state = get_state_feature_vector(perspective)
        last_state = self.last_state
        last_action = self.last_action
        last_reward = self.reward
        self.replay_buffer.push(last_state, last_action, last_reward, state, done)
    
