import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Simple MLP used to approximate Q(s, a).
    Hidden depth/width are configurable so we can play with capacity.
    """
    def __init__(self, state_dim, num_actions, layer_size, n_hidden_layers=2):
        super(QNetwork, self).__init__()
        layers = []
        in_width = state_dim

        # Stack a few (Linear -> ReLU) blocks
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_width, layer_size))
            layers.append(nn.ReLU())
            in_width = layer_size

        # Final linear to action logits (Q-values)
        layers.append(nn.Linear(in_width, num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)
