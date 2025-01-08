import sys
import hydra
import torch

from gymnasium import spaces

from .agent import Agent


sys.path.append("src/")
from utils.replay import ReplayMemory, Transition


class DDPG(Agent):
    def __init__(
        self,
        env,
        memory,
        network,
        training,
        device=torch.device("cuda:0"),
        bins=100,
        **_,
    ):
        super().__init__()
        self.env = env
        self.device = device

        # Parse memory, network, and training configurations
        self.memory_cfg = memory
        self.network_cfg = network
        self.training_cfg = training
        self.memory = ReplayMemory(self.memory_cfg.capacity)

        # Determine action and observation spaces
        n_actions = env.action_space.shape[0] if isinstance(env.action_space, spaces.Box) else bins
        n_observations = env.observation_space.shape[0]

        # Initialize actor and critic networks
        self.actor = hydra.utils.instantiate(
            config=self.network_cfg.actor,
            n_observations=n_observations,
            n_actions=n_actions,
        ).to(self.device)

        self.actor_target = hydra.utils.instantiate(
            config=self.network_cfg.actor,
            n_observations=n_observations,
            n_actions=n_actions,
        ).to(self.device)

        self.critic = hydra.utils.instantiate(
            config=self.network_cfg.critic,
            n_observations=n_observations,
            n_actions=n_actions,
        ).to(self.device)

        self.critic_target = hydra.utils.instantiate(
            config=self.network_cfg.critic,
            n_observations=n_observations,
            n_actions=n_actions,
        ).to(self.device)

        # Synchronize target networks
        self._sync_target_networks()

        # Optimizers and loss
        self.actor_optimizer = hydra.utils.instantiate(
            config=self.training_cfg.actor_optimizer,
            params=self.actor.parameters(),
        )

        self.critic_optimizer = hydra.utils.instantiate(
            config=self.training_cfg.critic_optimizer,
            params=self.critic.parameters(),
        )

        self.criterion = hydra.utils.instantiate(config=self.training_cfg.criterion)

    def _sync_target_networks(self):
        """Synchronize the target networks with the main networks."""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state):
        """Select an action using the actor network."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action

    def record(self, state, action, next_state, reward, done):
        """Record the transition in the replay memory."""
        self.memory.push(state, action, next_state, reward, done)

    def optimize(self, batch_size):
        """Optimize the actor and critic networks."""
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.FloatTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device).unsqueeze(1)

        # Update Critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            q_target = rewards + self.training_cfg.gamma * target_q * (1 - dones)
        q_values = self.critic(states, actions)
        critic_loss = self.criterion(q_values, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        policy_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor_target, self.actor, self.training_cfg.tau)
        self._soft_update(self.critic_target, self.critic, self.training_cfg.tau)

    def _soft_update(self, target, source, tau):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def load_state_dict(self, checkpoint):
        """Load the model and optimizer state dictionaries."""
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    def state_dict(self):
        """Return the model and optimizer state dictionaries."""
        return {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
