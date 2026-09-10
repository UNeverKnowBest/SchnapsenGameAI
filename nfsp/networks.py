import torch
from torch import nn

class Network(nn.Module):
    """Separate Q and average-policy instances; policy returns logits."""
    def __init__(self, state_dim=465, action_dim=28, hidden=(256, 128, 64)):
        super().__init__()
        layers = []
        width = state_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, next_width), nn.ReLU()))
            width = next_width
        layers.append(nn.Linear(width, action_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, states):
        return self.layers(states)

def masked_logits(values, masks):
    return values.masked_fill(~masks, -torch.inf)

@torch.no_grad()
def dqn_targets(online, target, rewards, next_states, next_masks, dones, gamma, double=True):
    """Terminal rows never enter a masked argmax or multiply zero by infinity."""
    result = rewards.clone()
    active = ~dones
    if active.any():
        states, masks = next_states[active], next_masks[active]
        if not masks.any(dim=1).all():
            raise ValueError("nonterminal transition has no legal next action")
        values = target(states)
        selector = online(states) if double else values
        actions = masked_logits(selector, masks).argmax(dim=1, keepdim=True)
        result[active] += gamma * values.gather(1, actions).squeeze(1)
    return result
