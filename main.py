import torch.optim as optim
import numpy as np
import torch
import math
import os
import time

from torch import nn
from DQN_bot import DQN_bot
from schnapsen.game import SchnapsenGamePlayEngine
from schnapsen.bots import RandBot
from random import Random
from model.DQN import DQN, Policy
from storage import ReplayBuffer, ReservoirBuffer
from common.utils import update_target
from torch.nn import functional as F
from ActionRepresentation import ActionRepresentation
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
epsilon_start = 0.2
epsilon_end = 0.01
decay_rate = 0.0001
min_buffer_size = 50000
gamma = 0.99
state_dim = 465
action_dim = 28
batch_size = 8192
update_frequency = 1000
reward_clip_value = 1.0 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 1000000000
save_path = "checkpoint"
lr = 1e-5
rl_start = 20000
sl_start = 20000
update_target_num = 10000
evaluation_interval = 20000
save_interval = 10000

# Model Initialization
p1_current_model = DQN(state_dim, action_dim).to(device)
p1_target_model = DQN(state_dim, action_dim).to(device)
update_target(p1_current_model, p1_target_model)

p2_current_model = DQN(state_dim, action_dim).to(device)
p2_target_model = DQN(state_dim, action_dim).to(device)
update_target(p2_current_model, p2_target_model)

p1_replay_buffer = ReplayBuffer(capacity=100000)
p2_replay_buffer = ReplayBuffer(capacity=100000)
p1_reservoir_buffer = ReservoirBuffer(capacity=100000)
p2_reservoir_buffer = ReservoirBuffer(capacity=100000)

p1_policy = Policy(state_dim, action_dim).to(device)
p2_policy = Policy(state_dim, action_dim).to(device)
p1_rl_optimizer = optim.Adam(p1_current_model.parameters(), lr=1e-3)
p2_rl_optimizer = optim.Adam(p2_current_model.parameters(), lr=1e-2)

p1_sl_optimizer = optim.Adam(p1_policy.parameters(), lr=1e-4)
p2_sl_optimizer = optim.Adam(p2_policy.parameters(), lr=1e-3)
action_represnetation = ActionRepresentation()
writer = SummaryWriter(log_dir="runs/rl_experiment")

def train():
    # Load checkpoint if available
    start_epoch, extra_info = load_checkpoint(os.path.join(save_path, "latest_checkpoint.pth"))
    epsilon = extra_info.get('epsilon', epsilon_start)
    for epoch_num in range(start_epoch + 1, epoch + 1):
        epsilon = update_epsilon(epoch_num)
        bot1 = DQN_bot("bot1", p1_current_model, p1_policy, p1_replay_buffer, p1_reservoir_buffer, action_represnetation, epsilon,device)
        bot2 = DQN_bot("bot2", p2_current_model, p2_policy, p2_replay_buffer, p2_reservoir_buffer, action_represnetation, epsilon,device)
        engine = SchnapsenGamePlayEngine()
        win, points, _ = engine.play_game(bot1, bot2, Random())

        if len(p1_replay_buffer) > rl_start:
            # RL Updates
            loss = compute_rl_loss(p1_current_model, p1_target_model, p1_replay_buffer, p1_rl_optimizer)
            if loss is not None:
                writer.add_scalar("p1/rl_loss", loss.item(), epoch_num)

            loss = compute_rl_loss(p2_current_model, p2_target_model, p2_replay_buffer, p2_rl_optimizer)
            if loss is not None:
                writer.add_scalar("p2/rl_loss", loss.item(), epoch_num)

                
        if len(p1_reservoir_buffer) > sl_start:
            # SL Updates
            loss = compute_sl_loss(p1_policy, p1_reservoir_buffer, p1_sl_optimizer)
            if loss is not None:
                writer.add_scalar("p1/sl_loss", loss.item(), epoch_num)

            loss = compute_sl_loss(p2_policy, p2_reservoir_buffer, p2_sl_optimizer)
            if loss is not None:
                writer.add_scalar("p2/sl_loss", loss.item(), epoch_num)

        if epoch_num % update_target_num == 0:
            update_target(p1_current_model, p1_target_model)
            update_target(p2_current_model, p2_target_model)

        if epoch_num % evaluation_interval == 0:
            bot1_win_rate = evaluate_vs_random(bot1, Random(), num_episodes=100)
            bot2_win_rate = evaluate_vs_random(bot2, Random(), num_episodes=100)
            writer.add_scalar("p1/eval_win_rate", bot1_win_rate, epoch_num)
            writer.add_scalar("p2/eval_win_rate", bot2_win_rate, epoch_num)
            print(f"Epoch {epoch_num}: Bot1 Win Rate vs Random: {bot1_win_rate}, Bot2 Win Rate vs Random: {bot2_win_rate}")

        if epoch_num % save_interval == 0:
            save_model(epoch_num, epsilon)
        # Update epsilon
        epsilon = update_epsilon(epoch_num)


def update_epsilon(steps):
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-decay_rate * steps)
    return epsilon

def evaluate_vs_random(bot, rng, num_episodes=100):
    """
    Let the bot play num_episodes games against a random bot and return the win rate.
    """
    engine = SchnapsenGamePlayEngine()
    opponents = [RandBot(rng, "rnd")]  # Additional opponents can be added here
    wins = 0
    for _ in range(num_episodes):
        opponent = rng.choice(opponents)
        winner, _, _ = engine.play_game(bot, opponent, rng)
        if winner == bot:
            wins += 1
    win_rate = wins / num_episodes
    return win_rate

def save_model(epoch_num,epsilon):
    """
    保存训练检查点信息，包括各个模型、优化器、缓冲区状态以及额外的训练指标。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = {
        'epoch': epoch_num,
        # 玩家 1 的状态
        'p1_current_model_state_dict': p1_current_model.state_dict(),
        'p1_target_model_state_dict': p1_target_model.state_dict(),
        'p1_policy_state_dict': p1_policy.state_dict(),
        'p1_rl_optimizer_state_dict': p1_rl_optimizer.state_dict(),
        'p1_sl_optimizer_state_dict': p1_sl_optimizer.state_dict(),
        # 玩家 2 的状态
        'p2_current_model_state_dict': p2_current_model.state_dict(),
        'p2_target_model_state_dict': p2_target_model.state_dict(),
        'p2_policy_state_dict': p2_policy.state_dict(),
        'p2_rl_optimizer_state_dict': p2_rl_optimizer.state_dict(),
        'p2_sl_optimizer_state_dict': p2_sl_optimizer.state_dict(),
        # 缓冲区状态（例如 replay buffer ，你也可以考虑添加其他缓冲区，例如 reservoir buffer）
        'replay_buffer_p1': p1_replay_buffer.state_dict(),
        'replay_buffer_p2': p2_replay_buffer.state_dict(),
        # 保存额外的训练信息
        'epsilon': epsilon,
    }

    checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch_num}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # 同时保存一份 latest checkpoint，便于重启训练时加载最新模型
    latest_path = os.path.join(save_path, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint updated at {latest_path}")

def compute_rl_loss(current_model, target_model, replay_buffer, optimizer):
    # Check if the replay buffer has enough samples for a batch
    if replay_buffer.size() < batch_size:
        print("Not enough samples in buffer. Skipping loss computation.")
        return 0

    # Sample a batch from the replay buffer (already tensors)
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Compute Q-values for the current states using the current model
    q_values = current_model(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1))  # Select Q-value of the taken action

    # Compute target Q-values using the target model
    with torch.no_grad():
        q_next_all = target_model(next_states)
        q_next_max = q_next_all.max(dim=1, keepdim=True)[0]  # Max Q-value for the next state
        target = rewards.unsqueeze(1) + gamma * q_next_max * (1 - dones.unsqueeze(1))

    # Calculate the loss between the current Q-values and the target Q-values
    loss_fn = nn.MSELoss()
    loss = loss_fn(q_sa, target)

    # Perform optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=5)
    optimizer.step()

    return loss

def compute_sl_loss(policy, reservoir_buffer, optimizer):
    # Check if the reservoir buffer has enough samples for a batch
    if reservoir_buffer.size() < batch_size:
        print("Not enough samples in buffer. Skipping loss computation.")
        return 0

    # Sample a batch from the reservoir buffer (already tensors)
    states, actions = reservoir_buffer.sample(batch_size)

    # Compute action probabilities using the policy network
    action_probs = policy(states)
    selected_probs = action_probs.gather(1, actions.unsqueeze(1))  # Extract probabilities of the taken actions
    log_probs = torch.log(selected_probs + 1e-8)  # Avoid log(0) by adding a small value

    # Loss is the negative log-probabilities (cross-entropy loss for action selection)
    loss = -log_probs.mean()

    # Perform optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# Function to load checkpoint
def load_checkpoint(checkpoint_path):
    """
    加载保存的检查点，恢复各个模型、优化器以及缓冲区的状态，同时返回额外的训练信息。
    
    参数:
        checkpoint_path: 检查点文件路径
        
    返回:
        start_epoch: 恢复的训练轮数
        extra_info: 包含 win_rate_p1、win_rate_p2、epsilon、timestamp 等信息的字典
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Starting training from scratch.")
        return 0, {}
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 恢复玩家 1 状态
    p1_current_model.load_state_dict(checkpoint['p1_current_model_state_dict'])
    p1_target_model.load_state_dict(checkpoint['p1_target_model_state_dict'])
    p1_policy.load_state_dict(checkpoint['p1_policy_state_dict'])
    p1_rl_optimizer.load_state_dict(checkpoint['p1_rl_optimizer_state_dict'])
    p1_sl_optimizer.load_state_dict(checkpoint['p1_sl_optimizer_state_dict'])

    # 恢复玩家 2 状态
    p2_current_model.load_state_dict(checkpoint['p2_current_model_state_dict'])
    p2_target_model.load_state_dict(checkpoint['p2_target_model_state_dict'])
    p2_policy.load_state_dict(checkpoint['p2_policy_state_dict'])
    p2_rl_optimizer.load_state_dict(checkpoint['p2_rl_optimizer_state_dict'])
    p2_sl_optimizer.load_state_dict(checkpoint['p2_sl_optimizer_state_dict'])

    # 恢复缓冲区状态
    p1_replay_buffer.load_state_dict(checkpoint['replay_buffer_p1'])
    p2_replay_buffer.load_state_dict(checkpoint['replay_buffer_p2'])

    # 读取额外信息，例如 win rate 和 epsilon
    extra_info = {
        'epsilon': checkpoint.get('epsilon', epsilon_start),
    }

    start_epoch = checkpoint['epoch']
    print(f"Checkpoint {checkpoint_path} loaded successfully. Training will resume from epoch {start_epoch}.")
    print("额外信息:", extra_info)

    return start_epoch, extra_info


train()
