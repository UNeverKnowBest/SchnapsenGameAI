import random
from collections import deque
from typing import Optional, List, Tuple
import numpy as np

from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase
from typing import Optional, cast, Literal
from schnapsen.deck import Suit, Rank
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenGamePlayEngine
from schnapsen.deck import Rank, Suit
from model.DQN import DQN, Policy
from feacture import *
from common.utils import epsilon_scheduler, update_target, print_log, load_model, save_model
from storage import ReplayBuffer, ReservoirBuffer

class DQN_bot(Bot):

    def __init__(self, name = None):
        super().__init__(name)
        self.update_target = update_target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model = DQN().to()
        self.target_model = DQN().to(self.device)
        self.update_target(self.current_model, self.target_model)
        self.policy = Policy().to(self.device)
        self.replay_buffer = ReplayBuffer(100000)
        self.reservoir_buffer = ReservoirBuffer(100000)
        self.state_deque = deque(maxlen=4)
        self.reward_deque = deque(maxlen=4)
        self.action_deque = deque(maxlen=4)
        self.rl_optimizer = optim.Adam(self.current_model.parameters(), lr=1e-4)
        self.sl_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.reward_list, self.rl_loss_list, self.sl_loss_list = [], [], []
        self.epsilon_reward = 0.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 100000
    
    def get_move(self, perspective: PlayerPerspective, leader_move) -> Move:
        action_represnetation = ActionRepresentation()
        state_representation = get_state_feature_vector(perspective)
        state_tensor = torch.tensor(state_representation, dtype=torch.float32, device=self.device)
        valid_moves = perspective.valid_moves()
        q_values = self.current_model(state_tensor)
        mask = torch.full_like(q_values, -float('inf'))

        if random.random() > self.epsilon:
            action = self.policy.act(state_tensor)
        else:
            action = self.current_model.act(state_tensor)

    def _update_epsilon(self):
        self.num_steps += 1
        fraction = min(self.num_steps / self.epsilon_decay, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

def compute_sl_loss(policy, reservoir_buffer, optimizer, args):
    state, action = reservoir_buffer.sample(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)

    probs = policy(state)
    probs_with_actions = probs.gather(1, action.unsqueeze(1))
    log_probs = probs_with_actions.log()

    loss = -1 * log_probs.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_rl_loss(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    # Q-Learning with target network
    q_values = current_model(state)
    target_next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.max(1)[0]
    expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


