import numpy as np
import torch


class ReplayBuffer(object):
    """
    Simple cyclic buffer for (s, a, s', r, not_done) tuples.
    Tensors are moved to the provided device during sampling.
    """
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Pre-allocate arrays
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        """
        Insert one transition.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Random minibatch for DQN update.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(self.device),
            torch.FloatTensor(self.action[idx]).to(self.device),
            torch.FloatTensor(self.next_state[idx]).to(self.device),
            torch.FloatTensor(self.reward[idx]).to(self.device),
            torch.FloatTensor(self.not_done[idx]).to(self.device),
        )
