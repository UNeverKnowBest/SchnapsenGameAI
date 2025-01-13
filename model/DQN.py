import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observation=133, n_actions=30): 

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class Policy(DQN):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return action
    
    