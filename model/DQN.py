import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),  
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.ReLU(),  
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),  # Smooth gradients
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)  # Output probabilities
        )

    def forward(self, x):
        return self.network(x)