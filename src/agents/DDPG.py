import sys
import torch

from .agent import Agent


sys.path.append("src/")
from utils.networks import Actor, Critic
from utils.replay import Transition


class DDPG(Agent):
    def __init__(
        self,
        env,
        memory,
        hidden_size=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        device="cpu",
        **kwargs,
    ):
        super().__init__(env, memory, device)
        self.gamma = gamma
        self.tau = tau

        # Actor and Critic Networks
        self.actor = Actor(env.observation_space.shape[0], hidden_size, env.action_space.shape[0]).to(device)
        self.actor_target = Actor(env.observation_space.shape[0], hidden_size, env.action_space.shape[0]).to(device)
        self.critic = Critic(env.observation_space.shape[0] + env.action_space.shape[0], hidden_size, 1).to(device)
        self.critic_target = Critic(env.observation_space.shape[0] + env.action_space.shape[0], hidden_size, 1).to(
            device,
        )

        # Synchronize target networks
        self._sync_target_networks()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.criterion = torch.nn.MSELoss()

    def _sync_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action

    def record(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def optimize(self, batch_size):
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
            q_target = rewards + self.gamma * target_q * (1 - dones)
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
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def load_state_dict(self, checkpoint):
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    def state_dict(self):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
