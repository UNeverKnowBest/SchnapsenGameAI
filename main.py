import torch.optim as optim
import numpy as np
import torch
import math
import os

from DQN_bot import DQN_bot
from schnapsen.game import SchnapsenGamePlayEngine
from schnapsen.bots import RandBot, RdeepBot
from random import Random
from model.DQN import DQN, Policy
from storage import ReplayBuffer
from common.utils import update_target
from torch.nn import functional as F

epsilon_start = 1.0
epsilon_end = 0.1
decay_rate = 0.001
min_buffer_size = 50000
gamma = 0.99
batch_size = 32000
update_target = update_target
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 100000
save_path = "./checkpoints"

dqn1 = DQN(state_dim=133, action_dim=28).to(device)
dqn2 = DQN(state_dim=133, action_dim=28).to(device)
replay_buffer_1 = ReplayBuffer(100000)
optimizer_1 = optim.Adam(dqn1.parameters(), lr=1e-4)
optimizer_2 = optim.Adam(dqn2.parameters(), lr=1e-4)

def trian():
    for epoch_num in range(epoch):
        epsilon = update_epsilon(epoch_num)
        rng = Random()
        bot1 = DQN_bot("DQN_bot", dqn1, dqn2, replay_buffer_1, optimizer_1, optimizer_2,epsilon)
        bot2 = RandBot(rng, "rnd")
        engine = SchnapsenGamePlayEngine()
        winner, points, score= engine.play_game(bot1, bot2, rng)
        print(f"Winner: {winner}, Points: {points}, Score: {score}")
        if len(replay_buffer_1) > min_buffer_size:
            loss = compute_rl_loss(dqn1, dqn2, replay_buffer_1, optimizer_1)
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            update_target(dqn1, dqn2)
        
        if epoch_num % 100 == 0:
            save_parameters(epoch_num, dqn1, dqn2, optimizer_1, replay_buffer_1, save_path)

def update_epsilon(steps):
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-decay_rate * steps)
    return epsilon

def compute_rl_loss(current_model, target_model, replay_buffer, optimizer):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        weights = torch.ones(batch_size)

        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(1)  # (batch,1)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1)  # (batch,1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.bool, device=device).unsqueeze(1)

        # Q-Learning with target network
        q_values = current_model(state)
        target_next_q_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.max(1)[0]
        expected_q_value = reward + (gamma ) * next_q_value * (1 - done)

        # Huber Loss
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = (loss * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
def save_parameters(epoch, model, target_model, optimizer, buffer, path):
    """
    Save model, target model, optimizer, and replay buffer states.
    """
    save_dict = {
        "model_state": model.state_dict(),
        "target_model_state": target_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "replay_buffer": buffer,  # Replay buffer object itself (serialized separately if needed)
        "epoch": epoch,
    }
    save_file = os.path.join(path, f"checkpoint_epoch_{epoch}.pth")
    torch.save(save_dict, save_file)
    print(f"Parameters saved to {save_file}")

def load_parameters(path, model, target_model, optimizer):
    """
    Load model, target model, optimizer, and replay buffer states.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    target_model.load_state_dict(checkpoint["target_model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    replay_buffer = checkpoint.get("replay_buffer", None)
    epoch = checkpoint.get("epoch", 0)
    print(f"Parameters loaded from {path}")
    return replay_buffer, epoch

trian()