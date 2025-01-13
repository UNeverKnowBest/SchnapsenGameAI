import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=133, action_dim=28):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)
    

class Policy(nn.Module):
    def __init__(self, state_dim=133, action_dim=28):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  
        )

    def forward(self, state):
        return self.network(state)
    
    