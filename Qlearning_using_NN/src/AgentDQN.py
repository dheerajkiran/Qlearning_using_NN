import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer


class AgentDQN:
    """
    Vanilla DQN with a target network and soft updates.
    Keeping it straightforward so we can focus on the Gridworld behavior.
    """
    def __init__(self,
                 state_dim,
                 action_dim,     # kept for compatibility with ReplayBuffer signature
                 n_actions,
                 layer_size,
                 lr,
                 gamma,
                 tau,
                 buffer_size,
                 batch_size,
                 use_cuda,
                 n_hidden_layers=2):

        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")

        # Online and target networks
        self.q_network = QNetwork(state_dim, n_actions, layer_size, n_hidden_layers).to(self.device)
        self.q_network_target = copy.deepcopy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay memory for off-policy updates
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, self.device)

        # Training knobs
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

    def select_greedy_action(self, state):
        """
        Argmax over Q(s,·). No exploration here.
        """
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        q_values = self.q_network(state_tensor).detach().cpu().numpy().flatten()
        return int(np.argmax(q_values))

    def select_action(self, state, epsilon):
        """
        Epsilon-greedy policy: random with prob=epsilon, otherwise greedy.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        return self.select_greedy_action(state)

    def store_experience(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def train(self):
        """
        One gradient step if the buffer has enough samples.
        """
        if self.replay_buffer.size < self.batch_size:
            return

        state_b, action_b, next_state_b, reward_b, not_done_b = self.replay_buffer.sample(self.batch_size)

        # Target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.q_network_target(next_state_b)
            max_next_q, _ = next_q.max(dim=1, keepdim=True)
            target_q = reward_b + not_done_b * self.gamma * max_next_q

        # Current Q(s, a)
        current_q = self.q_network(state_b)
        chosen_q = current_q.gather(1, action_b.long())

        loss = F.mse_loss(chosen_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for online_param, target_param in zip(self.q_network.parameters(), self.q_network_target.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def print_greedy_policy(self):
        """
        Quick printout of the greedy action at a few representative states.
        """
        actions = ['up', 'right', 'down', 'left']
        # A little grid of normalized (row, col) samples, matching the original layout.
        rows = [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]
        cols = [0, 0.3333, 0.6667, 1, 0, 0.6667, 0, 0.3333, 0.6667]

        print('Greedy policy:')
        for i in range(9):
            state = [rows[i], cols[i]]
            a = self.select_greedy_action(np.array(state))
            print(f'State {state}: {actions[a]}')
