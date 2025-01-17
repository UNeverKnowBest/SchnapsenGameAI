import random
from schnapsen.game import Bot, PlayerPerspective, Move
from storage import ReplayBuffer, ReservoirBuffer
from feacture import *
from ActionRepresentation import ActionRepresentation
import numpy as np
import torch

class DQN_bot(Bot):
    def __init__(self, name, current_model, policy, replay_buffer: ReplayBuffer, reservoir_buffer: ReservoirBuffer, 
                 action_representation: ActionRepresentation, epsilon, device, record_transitions=False):
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
          - record_transitions: 是否记录局中 transition 数据，默认 False
        """
        super().__init__(name)
        self.device = device
        self.current_model = current_model.to(device)
        self.policy = policy.to(device)
        self.replay_buffer = replay_buffer
        self.reservoir_buffer = reservoir_buffer
        self.action_representation = action_representation
        self.epsilon = epsilon
        self.record_transitions = record_transitions
        
        # 初始化状态和动作（状态全 0，动作 -1 表示未初始化）
        self.state = torch.zeros(545, dtype=torch.float32, device=device)
        self.action = -1
        
        # 用于记录局中数据（RL 数据和 SL 数据）
        self.rl_data = []
        self.sl_data = []
    
    def get_move(self, perspective: PlayerPerspective, leader_move, eta) -> Move:
        valid_moves = perspective.valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available!")
        
        action_mask = self.action_representation.set_valid_actions(valid_moves).to(self.device)
        state = get_state_feature(perspective, leader_move)  # 假设返回的是 tensor
        
        best_response = False
        if random.random() > 0.1:
            policy_output = self.policy(state)  # 输出为 softmax 后的概率
            action_probs = policy_output.detach().cpu().numpy()
            
            valid_indices = torch.where(action_mask > 0)[0].tolist()
            valid_probs = action_probs[valid_indices]
            if np.any(np.isnan(valid_probs)):
                valid_probs = np.nan_to_num(valid_probs, nan=0.0)
                
            total = valid_probs.sum()
            if total <= 1e-12:
                valid_probs = np.ones_like(valid_probs) / len(valid_probs)
            else:
                valid_probs /= total
            
            selected_idx = np.random.choice(valid_indices, p=valid_probs)
            action = int(selected_idx)
            move = self.action_representation.decode_action(action)
        else:
            best_response = True
            if random.random() < self.epsilon:
                move = random.choice(valid_moves)
                action = self.action_representation.encode_action(move)
            else:
                q_values = self.current_model(state)
                q_values[action_mask == 0] = -float('inf')
                action = torch.argmax(q_values).item()
                move = self.action_representation.decode_action(action)
        
        if best_response:
            # 记录 best response 数据用于 SL
            if self.reservoir_buffer is not None:
                self.reservoir_buffer.push(state, action)
            if self.record_transitions:
                self.sl_data.append((state, action))
        
        # 如果之前已有动作，则记录 transition 到 RL 数据缓冲
        if self.action != -1:
            self._after_trick(state)
        
        self.state = state
        self.action = action
        return move
    
    def notify_game_end(self, won, perspective):
        if self.action == -1:
            return
        
        reward = 1 if won else -1
        next_state = torch.zeros_like(self.state)
        if self.replay_buffer is not None:
            self.replay_buffer.push(self.state, self.action, reward, next_state, done=True)
        if self.record_transitions:
            self.rl_data.append((self.state, self.action, reward, next_state, True))
        # 重置
        self.state = torch.zeros_like(self.state)
        self.action = -1
        
    def _after_trick(self, next_state):
        if self.action == -1:
            return
        
        if self.replay_buffer is not None:
            self.replay_buffer.push(self.state, self.action, 0, next_state, done=False)
        if self.record_transitions:
            self.rl_data.append((self.state, self.action, 0, next_state, False))
    
    def get_collected_data(self):
        """
        返回记录的 RL 和 SL 数据，并清空内部缓冲
        """
        rl, sl = self.rl_data, self.sl_data
        self.rl_data = []
        self.sl_data = []
        return rl, sl
