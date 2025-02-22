"""
PPO Agent implementation following the Codebase style.

This agent:
  • Uses an actor network that outputs Gaussian distribution parameters.
  • Samples a raw action then squashes via tanh and finally scales it to the environment's bounds.
  • Maintains an on‐policy buffer for a full rollout.
  • Uses GAE to compute advantages.
  • Performs several PPO update epochs using the clipped surrogate loss,
    critic (value) loss, and an entropy bonus.
"""

import collections
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium.spaces import Box
from PIL import Image

from utils.replay import ReplayMemory

# Append the src/ folder to sys.path for any utilities
sys.path.append("src/")
from agents import Agent


###############################################################################
# PPO Network Definitions
###############################################################################
class PPOActor(nn.Module):
    def __init__(self, input_size, hidden_size, action_dim, log_std_init=-0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        # use a state-independent log_std parameter
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, x):
        # Using tanh activation for hidden layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        # Expand log_std to match mean's shape
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std

    def get_dist(self, x):
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)

    def sample(self, x):
        """
        Returns:
           action: the squashed (tanh) action (to be scaled to env bounds)
           raw_action: the pre-tanh action (used to recompute log_prob later)
           log_prob: log probability with tanh correction
        """
        mean, std = self.forward(x)
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()  # reparameterized sample
        action = torch.tanh(raw_action)
        # Apply correction for the tanh squashing
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, raw_action, log_prob


class PPOCritic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.value(x)


###############################################################################
# PPO Agent Implementation
###############################################################################
class PPO(Agent):
    def __init__(
        self,
        env,
        training,
        memory,
        hidden_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        ppo_epochs=10,
        mini_batch_size=64,
        device=torch.device("cuda:0"),
        mode="train",
        **_,
    ):
        super().__init__()
        self.env = env
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.batch_size = training.batch_size
        self.mode = mode

        # Add memory attribute for compatibility with evaluate.py
        self.memory_cfg = memory
        self.memory = ReplayMemory(self.memory_cfg.capacity)
        # Keep buffer for PPO-specific data
        self.buffer = []

        # Assume continuous action spaces (Box); extend as needed.
        self.num_states = env.observation_space.shape[0]
        if hasattr(env.action_space, "shape"):
            self.action_dim = env.action_space.shape[0]
            self.action_space = env.action_space
        else:
            raise NotImplementedError("PPO agent is implemented only for continuous (Box) action spaces.")

        self.actor = PPOActor(self.num_states, hidden_size, self.action_dim).to(self.device)
        self.critic = PPOCritic(self.num_states, hidden_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def _scale_action(self, action):
        """
        Scale a tanh-squashed action (in [-1, 1]) to the environment's action bounds.
        """
        low = torch.FloatTensor(self.action_space.low).to(self.device)
        high = torch.FloatTensor(self.action_space.high).to(self.device)
        # Scale from [-1, 1] to [low, high]
        scaled_action = low + (action + 1.0) * (high - low) / 2.0
        return scaled_action

    def select_action(self, state, deterministic=False, action_space=None):
        """Select an action using the actor network."""
        scaled_action, _, _, _ = self.get_action_and_value(state, deterministic)
        if action_space is not None:
            return np.clip(scaled_action, action_space.low, action_space.high)
        return scaled_action

    def get_action_and_value(self, state, deterministic=False):
        """
        Returns:
           scaled_action: action to feed to the environment (numpy array)
           raw_action: pre-tanh action (numpy array) used for log_prob recomputation
           log_prob: log probability from the policy (numpy array)
           value: value estimate from the critic (numpy array)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            with torch.no_grad():
                mean, std = self.actor.forward(state_tensor)
                raw_action = mean
                action = torch.tanh(raw_action)
                scaled_action = self._scale_action(action)
                # Compute log_prob for completeness even in deterministic mode
                normal = torch.distributions.Normal(mean, std)
                log_prob = normal.log_prob(raw_action) - torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
                value = self.critic(state_tensor)
            return (
                scaled_action.cpu().numpy()[0],
                raw_action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
                value.cpu().numpy()[0],
            )
        else:
            with torch.no_grad():
                action, raw_action, log_prob = self.actor.sample(state_tensor)
                scaled_action = self._scale_action(action)
                value = self.critic(state_tensor)
            return (
                scaled_action.cpu().numpy()[0],
                raw_action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
                value.cpu().numpy()[0],
            )

    def record(self, state, action, next_state, reward, done, raw_action=None, log_prob=None, value=None):
        """
        Extend the abstract record() to store additional info needed for PPO.
        We ignore next_state for advantage computation (we use a full rollout).
        """
        self.buffer.append((state, raw_action, action, reward, done, log_prob, value))

    def clear_buffer(self):
        """Clear both buffer and memory"""
        self.buffer = []
        self.memory.clear()

    def compute_gae(self, rewards, dones, values, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        rewards, dones, values: lists (length = rollout length)
        next_value: critic value for the final state (0 if terminal)
        Returns the advantages as a list.
        """
        advantages = []
        gae = 0
        # Append next_value to the list of values for bootstrapping.
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def optimize(self):
        """
        Perform PPO policy and value updates using the rollout stored in self.buffer.
        """
        # Unpack the buffer (list of tuples)
        # Each tuple: (state, raw_action, scaled_action, reward, done, log_prob, value)
        states, raw_actions, actions, rewards, dones, old_log_probs, values = zip(*self.buffer)
        # Convert states and raw_actions:
        states = torch.FloatTensor(np.array(states)).to(self.device)
        raw_actions = torch.FloatTensor(np.array(raw_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        # values is a list of scalars (from critic); ensure they are floats
        values = [float(v) for v in values]
        rewards = list(rewards)
        dones = list(dones)

        with torch.no_grad():
            # Compute value for the last state in the rollout.
            last_state = torch.FloatTensor(np.array(self.buffer[-1][0])).unsqueeze(0).to(self.device)
            # If the last step is terminal, there is no bootstrap.
            next_value = 0.0 if self.buffer[-1][4] else self.critic(last_state).cpu().item()

        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, dones, values, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Convert advantages and returns to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        for epoch in range(self.ppo_epochs):
            # Shuffle indices for mini-batch updates
            perm = np.random.permutation(dataset_size)
            for i in range(0, dataset_size, self.mini_batch_size):
                idx = perm[i : i + self.mini_batch_size]
                batch_states = states[idx]
                batch_raw_actions = raw_actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx].unsqueeze(1)  # shape (batch, 1)
                batch_advantages = advantages[idx].unsqueeze(1)  # shape (batch, 1)

                # Recompute new log probabilities for the stored raw actions using current actor parameters
                mean, std = self.actor.forward(batch_states)
                normal = torch.distributions.Normal(mean, std)
                new_log_probs = normal.log_prob(batch_raw_actions) - torch.log(1 - torch.tanh(batch_raw_actions).pow(2) + 1e-6)
                new_log_probs = new_log_probs.sum(dim=-1, keepdim=True)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (using MSE loss)
                values_pred = self.critic(batch_states)
                critic_loss = nn.MSELoss()(values_pred, batch_returns)

                # Entropy bonus for exploration
                entropy = normal.entropy().sum(dim=-1).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        # After update, clear the buffer.
        self.clear_buffer()

    def train_episode(self) -> None:
        """
        Collect a rollout (one episode) then do PPO update.
        """
        state, _ = self.env.reset()
        done = False
        while not done:
            # Get action info from policy
            scaled_action, raw_action, log_prob, value = self.get_action_and_value(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = self.env.step(scaled_action)
            done = terminated or truncated
            # Record the rollout data. Note: we pass state, the executed (scaled) action,
            # reward, done along with extra info for the update.
            self.record(state, scaled_action, next_state, reward, done, raw_action, log_prob, value)
            state = next_state
        # Once the episode (rollout) is complete, update the networks.
        self.optimize()

    def evaluate_episode(self, render: bool = True) -> tuple[list[Image.Image], dict]:
        state, info = self.env.reset()
        done = False
        frames = []
        while not done:
            if render:
                frame = self.env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
            scaled_action, _, _, _ = self.get_action_and_value(state, deterministic=True)
            next_state, reward, terminated, truncated, info = self.env.step(scaled_action)
            self.memory.push(state, scaled_action, next_state, reward, done)
            done = terminated or truncated
            state = next_state
        return frames, info

    def load_state_dict(self, agent_state_dict, episode=None, **_):
        self.actor.load_state_dict(agent_state_dict["actor_state_dict"])
        self.critic.load_state_dict(agent_state_dict["critic_state_dict"])
        self.actor_optimizer.load_state_dict(agent_state_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(agent_state_dict["critic_optimizer_state_dict"])

    def state_dict(self):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        } 
