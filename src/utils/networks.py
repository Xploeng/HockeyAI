import numpy as np
import torch


class DQN(torch.nn.Module):
    def __init__(self, n_actions, n_observations, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_observations, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

    def max_q(self, observations):
        # compute the maximal Q-value
        # Complete this
        observations = torch.from_numpy(observations.astype(np.float32))
        q_values = self.forward(observations)
        return torch.max(q_values, dim=1).values.detach().numpy()
