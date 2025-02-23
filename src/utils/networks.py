import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from torchrl.modules import NoisyLinear


class SACActor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # Two linear layers for mean and log_std
        self.mean_linear = nn.Linear(hidden_size, output_size)
        self.log_std_linear = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        # Apply tanh squashing
        action = torch.tanh(x_t)
        # Compute log_prob with correction for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean)
        else:
            action, _, _ = self.sample(state)
        return action.detach().cpu().numpy()[0]


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        # Q1 network
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        # Q2 network
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        # Q1
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)
        # Q2
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        q2 = self.linear6(x2)
        return q1, q2

    def min_q(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


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
    def __init__(self, input_size, hidden_size, output_size):
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
    """
    Noisy Duealing and Categorical DQN
    Adapted from:
    https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb
    """
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


class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        self.reward = nn.Sequential(
            nn.Linear(obs_dim + action_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def predict_next_state(self, state, action):
        latent = self.encoder(torch.cat([state, action], dim=-1))
        next_state = self.dynamics(torch.cat([latent, action], dim=-1))
        return next_state

    def predict_reward(self, state, action, next_state):
        return self.reward(torch.cat([state, action, next_state], dim=-1))


class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
