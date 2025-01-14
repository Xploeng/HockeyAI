import collections
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.autograd import Variable

from .agent import Agent


sys.path.append("src/")
from utils.networks import Actor, Critic
from utils.replay import ReplayMemory, Transition


class DDPG(Agent):
    def __init__(
        self,
        env,
        memory,
        training,
        hidden_size=256,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        gamma=0.99,
        tau=1e-2,
        device=torch.device("cuda:0"),
        **_,
    ):
        super().__init__()
        self.env = env
        self.device = device
        self.batch_size = training.batch_size
        self.gamma = gamma
        self.tau = tau

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.memory_cfg = memory

        self.memory = ReplayMemory(self.memory_cfg.capacity)
        self.noise = OUNoise(env.action_space)

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions).to(self.device)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.criterion = nn.MSELoss()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_learning_rate)

    def select_action(self, state):
        """Select an action using the actor network."""
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0, 0]
        return action

    def record(self, state, action, next_state, reward, done):
        """Record the transition in the replay memory."""
        self.memory.push(state, action, next_state, reward, done)

    def optimize(self, batch_size):
        """Optimize the actor and critic networks."""
        if len(self.memory) < batch_size:
            return

        try:
            transitions = self.memory.sample(batch_size)["transitions"]
        except ValueError as e:
            print(e)
            return
        # transitions = self.memory.sample(batch_size)["transitions"]
        # print(transitions["transitions"])
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.FloatTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).to(self.device).unsqueeze(1)

        # Critic loss
        q_values = self.critic(states, actions)
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            q_target = rewards + self.gamma * target_q * (1 - dones)
        critic_loss = self.criterion(q_values, q_target)

        # Actor loss
        policy_loss = -self.critic(states, self.actor(states)).mean()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

    def _soft_update(self, target, source, tau):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train_episode(self, eval = False) -> None:
        state, _ = self.env.reset()
        self.noise.reset()

        done = False
        step = 0

        while not done:
            action = self.select_action(state)
            action = self.noise.select_action(action, step)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated

            if terminated:
                next_state = None

            self.record(state, action, next_state, reward, done)
            if not eval:
                self.optimize(self.batch_size)

            state = next_state
            step += 1

    def evaluate_episode(self) -> tuple[list[Image.Image], dict]:
        state, info = self.env.reset()

        done = False
        step = 0
        frames = []

        while not done:
            # Render the environment and save the frames
            frame = self.env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

            # Action selection and recording the transition
            action = self.select_action(state)
            action = self.noise.select_action(action, step)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.record(state.tolist(), action, next_state, reward, done)

            if not done:
                state = next_state
                step += 1

        return frames, info

    def load_state_dict(self, agent_state_dict, episode=None, **_):
        """Load the model and optimizer state dictionaries."""
        self.actor.load_state_dict(agent_state_dict["actor_state_dict"])
        self.actor_target.load_state_dict(agent_state_dict["actor_target_state_dict"])
        self.critic.load_state_dict(agent_state_dict["critic_state_dict"])
        self.critic_target.load_state_dict(agent_state_dict["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(agent_state_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(agent_state_dict["critic_optimizer_state_dict"])

        self.memory = agent_state_dict["memory"]
        self.steps_done = len(self.memory)

    def state_dict(self):
        """Return the model and optimizer state dictionaries."""
        return collections.OrderedDict(
            {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "memory": self.memory,
            },
        )


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def select_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
