# DQN_bot.py

import random
from schnapsen.game import Bot, PlayerPerspective, Move
from storage import ReplayBuffer, ReservoirBuffer
from feacture import *
from ActionRepresentation import ActionRepresentation
import numpy as np
import torch

class DQN_bot(Bot):
    def __init__(self, name, current_model, policy, replay_buffer: ReplayBuffer, reservoir_buffer: ReservoirBuffer, 
                 action_representation: ActionRepresentation, epsilon, device):
        """
        初始化 DQN_bot。
        
        参数：
          - name: Bot 名称
          - current_model: 当前用于 Q 值计算的神经网络模型
          - policy: 策略网络（例如 softmax 策略）
          - replay_buffer: 用于存储 RL 数据的缓冲区
          - reservoir_buffer: 用于存储 SL 数据的缓冲区
          - action_representation: 用于编码/解码动作的对象
          - epsilon: 探索率
          - device: 设备（例如 "cpu" 或 "cuda"）
        """
        super().__init__(name)
        self.device = device
        self.current_model = current_model.to(device)
        self.policy = policy.to(device)
        self.replay_buffer = replay_buffer
        self.reservoir_buffer = reservoir_buffer
        # 初始化状态和动作（默认初始状态全零，动作设为无效状态 -1）
        self.state = torch.zeros(465, dtype=torch.float32, device=device)
        self.action = -1  # 无效默认动作
        self.epsilon = epsilon
        self.action_representation = action_representation

    def get_move(self, perspective: PlayerPerspective, leader_move, eta) -> Move:
        valid_moves = perspective.valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available!")

        # 设置有效动作掩码，并传到正确设备
        action_mask = self.action_representation.set_valid_actions(valid_moves).to(self.device)
        # 获取状态特征（假设 get_state_feature 已经输出 tensor 形式数据）
        state = get_state_feature(perspective, leader_move).to(self.device)
        
        best_response = False
        # 这里用 90% 概率走策略网络采样，10% 走 best response（可根据需要修改）
        if random.random() > 0.1:
            policy_output = self.policy(state)  # shape: [num_actions]，假定已 softmax 归一化
            action_probs = policy_output.detach().cpu().numpy()
            
            # 构建有效动作的索引列表
            valid_indices = torch.where(action_mask > 0)[0].tolist()
            
            # 选取有效动作对应的概率值
            valid_probs = action_probs[valid_indices]
            
            # 防止 NaN 或者全 0 概率情况
            if np.any(np.isnan(valid_probs)):
                valid_probs = np.nan_to_num(valid_probs, nan=0.0)

            total = valid_probs.sum()
            if total <= 1e-12:
                valid_probs = np.ones_like(valid_probs) / len(valid_probs)
            else:
                valid_probs /= total

            # 根据有效概率采样动作
            selected_idx = np.random.choice(valid_indices, p=valid_probs)
            selected_idx = int(selected_idx)
            move = self.action_representation.decode_action(selected_idx)

        else:
            best_response = True
            # 当走 best response 策略时，按 epsilon 决定是随机动作还是贪婪动作
            if random.random() < self.epsilon:
                move = random.choice(valid_moves)
                selected_idx = self.action_representation.encode_action(move)
            else:
                q_values = self.current_model(state)
                # 将无效动作置为 -infty，确保不会被选中
                q_values[action_mask == 0] = -float('inf')
                action = torch.argmax(q_values).item()
                selected_idx = int(action)
                move = self.action_representation.decode_action(action)

        # 如果采用 best response，则将对应 (state, action) 数据存储到 reservoir_buffer 用于 SL
        if best_response:
            self.reservoir_buffer.push(state, selected_idx)

        # 如果之前已有动作（即 state 已经初始化），则将上一步 transition 存储到 replay_buffer
        if self.action != -1:
            self._after_trick(state)

        self.state = state
        self.action = selected_idx
        return move

    def notify_game_end(self, won, perspective):
        if self.action == -1:  # 如果没有有效动作，则直接返回
            return

        reward = 1 if won else -1 

        # 将终止状态（可设为全零）与最终奖励存入 replay_buffer
        next_state = torch.zeros_like(self.state)  # 终止状态表示
        self.replay_buffer.push(self.state, self.action, reward, next_state, done=True)

        # 游戏结束后清空状态与动作
        self.state = torch.zeros_like(self.state)
        self.action = -1

    def _after_trick(self, next_state):
        if self.action == -1:
            return

        # 在没有终止的情况下存储一步 transition，奖励设为 0
        self.replay_buffer.push(self.state, self.action, 0, next_state, done=False)
