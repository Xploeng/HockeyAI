import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from torchrl.modules import NoisyLinear


class DQN(nn.Module):
    def __init__(self, n_actions, n_observations, hidden_size, **_):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

    def max_q(self, observations):
        observations = torch.from_numpy(observations.astype(np.float32))
        q_values = self.forward(observations)
        return torch.max(q_values, dim=1).values.detach().numpy()


class DuelingQNetwork(torch.nn.Module):
    def __init__(self, n_actions, n_observations, hidden_size, **_):
        super().__init__()

        # feature layer
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(n_observations, hidden_size),
            torch.nn.ReLU(),
        )

        # value layer
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

        # advantage layer
        self.advantage_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

    def max_q(self, observations):
        observations = torch.from_numpy(observations.astype(np.float32))
        q_values = self.forward(observations)
        return torch.max(q_values, dim=1).values.detach().numpy()


class NoisyDQN(torch.nn.Module):
    def __init__(self, n_actions, n_observations, hidden_size, **_):
        super().__init__()
        self.feature = torch.nn.Linear(n_observations, hidden_size)
        self.noisy1 = NoisyLinear(hidden_size, hidden_size)
        self.noisy2 = NoisyLinear(hidden_size, n_actions)

    def forward(self, x):
        feature = torch.nn.functional.relu(self.feature(x))
        hidden = torch.nn.functional.relu(self.noisy1(feature))
        return self.noisy2(hidden)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    def max_q(self, observations):
        # compute the maximal Q-value
        # Complete this
        observations = torch.from_numpy(observations.astype(np.float32))
        q_values = self.forward(observations)
        return torch.max(q_values, dim=1).values[0].detach().numpy()


class NoisyDueling(torch.nn.Module):
    def __init__(self, n_actions, n_observations, hidden_size, **_):
        super().__init__()
        # feature layer
        self.feature_layer = torch.nn.Sequential(
            NoisyLinear(n_observations, hidden_size),
            torch.nn.ReLU(),
        )

        # value layer
        self.value_layer = torch.nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            NoisyLinear(hidden_size, 1),
        )

        # advantage layer
        self.advantage_layer = torch.nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            NoisyLinear(hidden_size, n_actions),
        )

    def forward(self, x):
        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values

    def reset_noise(self):
        for module in self.children():
            if hasattr(module, "reset_noise"):
                module.reset_noise()

    def max_q(self, observations):
        # compute the maximal Q-value
        # Complete this
        observations = torch.from_numpy(observations.astype(np.float32))
        q_values = self.forward(observations)
        return torch.max(q_values, dim=1).values[0].detach().numpy()


class NoisyCategoricalDueling(torch.nn.Module):
    def __init__(self, n_actions, n_observations, hidden_size, atom_size, support, **_):
        super().__init__()

        self.in_dim = n_observations
        self.out_dim = n_actions
        self.support = support
        self.atom_size = atom_size
        # common feature layer
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, hidden_size),
            torch.nn.ReLU(),
        )

        # value layer
        self.value_hidden_layer = NoisyLinear(hidden_size, hidden_size)
        self.value_layer = NoisyLinear(hidden_size, atom_size)

        # advantage layer
        self.advantage_hidden_layer = NoisyLinear(hidden_size, hidden_size)
        self.advantage_layer = NoisyLinear(hidden_size, self.out_dim * atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32, f"Input must be a float tensor but got {x.dtype}"
        dist = self.dist(x)

        q_values = torch.sum(dist * self.support, dim=2)

        return q_values

    def dist(self, x):
        """Get the distribution of atoms."""
        feature = self.feature_layer(x)
        adv_hid = torch.nn.functional.relu(self.advantage_hidden_layer(feature))
        val_hid = torch.nn.functional.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = torch.nn.functional.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        for module in self.children():
            if hasattr(module, "reset_noise"):
                module.reset_noise()

    def max_q(self, observations):
        # compute the maximal Q-value
        # Complete this
        observations = torch.from_numpy(observations.astype(np.float32))
        q_values = self.forward(observations)
        return torch.max(q_values, dim=1).values[0].detach().numpy()
