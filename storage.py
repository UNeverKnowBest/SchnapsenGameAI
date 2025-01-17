import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        """
        初始化 ReplayBuffer。
        :param capacity: 最大存储容量。
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存储 (state, action, reward, next_state, done)。
        :param state: torch.Tensor，当前状态。
        :param action: int，执行的动作。
        :param reward: float，获得的奖励。
        :param next_state: torch.Tensor，下一个状态。
        :param done: bool，是否完成。
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从缓冲区随机采样，并将张量移动到 GPU（如果可用）。
        :param batch_size: 采样批量大小。
        :return: 各维度张量 (states, actions, rewards, next_states, dones)。
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备
        
        return (torch.stack(states).to(device),
                torch.tensor(actions, dtype=torch.int64).to(device),
                torch.tensor(rewards, dtype=torch.float32).to(device),
                torch.stack(next_states).to(device),
                torch.tensor(dones, dtype=torch.float32).to(device))

    def size(self):
        """
        返回缓冲区中当前存储的样本数量。
        """
        return len(self.buffer)

    def state_dict(self):
        """
        保存缓冲区的内容和相关参数。
        :return: 包含 capacity 和 buffer 的字典。
        """
        return {
            'capacity': self.capacity,
            'buffer': [(state.cpu(), action, reward, next_state.cpu(), done)
                       for state, action, reward, next_state, done in self.buffer]
        }

    def load_state_dict(self, state_dict):
        """
        加载缓冲区的内容和相关参数。
        :param state_dict: 包含缓冲区状态的字典。
        """
        self.capacity = state_dict['capacity']
        self.buffer = deque(
            [(state.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), action, reward,
              next_state.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), done)
             for state, action, reward, next_state, done in state_dict['buffer']],
            maxlen=self.capacity
        )

    def __len__(self):
        """
        返回缓冲区的当前长度。
        """
        return len(self.buffer)

class ReservoirBuffer:
    def __init__(self, capacity):
        """
        初始化 ReservoirBuffer。
        :param capacity: 最大存储容量。
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.n = 0  # 接收到的总样本数

    def push(self, state, action):
        """
        存储 (state, action) 样本。
        :param state: torch.Tensor，状态张量。
        :param action: int，动作标识符。
        """
        self.n += 1
        sample = (state, action)
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.n - 1)
            if idx < self.capacity:
                self.buffer[idx] = sample

    def sample(self, batch_size):
        """
        从缓冲区随机采样，并将张量迁移到可用设备。
        :param batch_size: 采样批量大小。
        :return: (states, actions) -> torch.Tensor, torch.Tensor
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions = zip(*batch)
        
        # 自动检测设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        return torch.stack(states).to(device), torch.tensor(actions, dtype=torch.int64).to(device)

    def size(self):
        """
        返回缓冲区中当前样本数量。
        """
        return len(self.buffer)

    def state_dict(self):
        """
        保存缓冲区状态。
        :return: 包含 capacity 和 buffer 的字典。
        """
        return {
            'capacity': self.capacity,
            'buffer': [(state.cpu(), action) for state, action in self.buffer]
        }

    def load_state_dict(self, state_dict):
        """
        加载缓冲区状态。
        :param state_dict: 包含缓冲区状态的字典。
        """
        self.capacity = state_dict['capacity']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = deque(
            [(state.to(device), action) for state, action in state_dict['buffer']],
            maxlen=self.capacity
        )

    def __len__(self):
        """
        返回缓冲区的当前长度。
        """
        return len(self.buffer)
