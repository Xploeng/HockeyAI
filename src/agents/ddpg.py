import collections
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium.spaces import Box
from icecream import ic
from PIL import Image
from torch.autograd import Variable

from .agent import Agent


sys.path.append("src/")
from utils.networks import Actor, Critic
from utils.noise import OUNoise, PinkNoise
from utils.replay import ReplayMemory, Transition


torch.autograd.set_detect_anomaly(True)


class DDPG(Agent):
    def __init__(
        self,
        env,
        memory,
        training,
        opponent,
        hidden_size=256,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        gamma=0.99,
        tau=1e-2,
        device=torch.device("cuda:0"),
        mode="train",
        **_,
    ):
        super().__init__()
        self.env = env
        self.mode = mode
        self.opponent = opponent
        self.device = device
        self.batch_size = training.batch_size
        self.gamma = gamma
        self.tau = tau

        self.memory_cfg = memory
        self.memory = ReplayMemory(self.memory_cfg.capacity)

        self.hockey = True if opponent is not None or self.mode == "opponent" else False
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        if self.hockey:
            out_actions = int(self.num_actions / 2)

            self.agent_action_space = Box(
                low=env.action_space.low[:out_actions],
                high=env.action_space.high[:out_actions],
                dtype=env.action_space.dtype,
            )

            self.noise = OUNoise(self.agent_action_space)

            self.actor = Actor(self.num_states, hidden_size, out_actions).to(self.device)
            self.actor_target = Actor(self.num_states, hidden_size, out_actions).to(self.device)
            self.critic = Critic(self.num_states + self.num_actions, hidden_size, out_actions).to(self.device)
            self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, out_actions).to(self.device)

        else:
            self.agent_action_space = env.action_space
            self.noise = OUNoise(self.agent_action_space)

            # Networks
            self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
            self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
            self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions).to(self.device)
            self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions).to(
                self.device,
            )

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.criterion = nn.MSELoss()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_learning_rate)

    def select_action(self, state, action_space=None):
        """Select an action using the actor network."""
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0]  #  , 0]

        if action_space is not None:
            return np.clip(action, action_space.low, action_space.high)

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

        # Compute target Q values
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            if self.hockey:
                # ic(next_states.shape)
                if self.opponent.opp_type == "basic":
                    target_opponent_actions = self._batch_opponent_actions(next_states)
                else: # agent opponent use batched states (way faster)
                    target_opponent_actions = self.opponent.act(next_states.unsqueeze(0))
                target_actions = torch.cat([target_actions, target_opponent_actions], dim=1)
            target_q = self.critic_target(next_states, target_actions)
            q_target = rewards + self.gamma * target_q * (1 - dones)

        # Compute current Q values
        # ic(states.shape, actions.shape)
        q_values = self.critic(states, actions)

        # Critic loss
        critic_loss = self.criterion(q_values, q_target)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        if self.hockey:
            agent_actions = self.actor(states)
            if self.opponent.opp_type == "basic":
                opponent_actions = self._batch_opponent_actions(states)
            else: # agent opponent use batched states (way faster)
                opponent_actions = self.opponent.act(states)
            joint_actions = torch.cat([agent_actions, opponent_actions], dim=1)
            policy_loss = -self.critic(states, joint_actions).mean()
        else:
            policy_loss = -self.critic(states, self.actor(states)).mean()

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

    def _batch_opponent_actions(self, states):
        actions = []
        for state in states:
            state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
            actions.append(self.opponent.act(state_np))
        return torch.FloatTensor(np.array(actions)).to(states.device)

    def train_episode(self) -> None:
        state, _ = self.env.reset()
        self.noise.reset()

        done = False
        step_idx = 0

        while not done:
            action = self.select_action(state)
            action = self.noise.select_action(action, step_idx)  # Exploration noise
            action_opp = self.opponent.act(state) if self.hockey else None
            action_opp = action_opp.cpu().numpy() if isinstance(action_opp, torch.Tensor) else action_opp
            next_state, done = self.step(state, action, action_opp)

            self.optimize(self.batch_size)

            state = next_state
            step_idx += 1

    def step(self, state, action, action_opp=None):
        # Stack actions for hockey env
        # ic(action.shape, action_opp.shape)
        action = np.hstack([action, action_opp]) if self.hockey else action

        # Take a step in the environment
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated

        if terminated:
            next_state = np.zeros(self.num_states)

        self.record(state, action, next_state, reward, done)
        return next_state, done

    def evaluate_episode(self, render: bool = True) -> tuple[list[Image.Image], dict]:
        state, info = self.env.reset()

        done = False
        frames = []

        while not done:
            if render:
                # Choose rendering parameters based on the environment type
                render_kwargs = {"mode": "rgb_array"} if self.hockey else {}
                frame = self.env.render(**render_kwargs)
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            # Action selection (No exploration noise during evaluation)
            action = self.select_action(state, self.agent_action_space)

            # Hockey env opponent action and stacking if necessary
            action_opp = self.opponent.act(state) if self.hockey else None
            action_opp = action_opp.cpu().numpy() if isinstance(action_opp, torch.Tensor) else action_opp
            action = np.hstack([action, action_opp]) if self.hockey else action

            # Take a step in the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.record(state.tolist(), action, next_state, reward, done)
            state = next_state

        return frames, info

    def load_state_dict(self, agent_state_dict, episode=None, **_):
        """Load the model and optimizer state dictionaries."""
        self.actor.load_state_dict(agent_state_dict["actor_state_dict"])
        self.actor_target.load_state_dict(agent_state_dict["actor_target_state_dict"])
        self.critic.load_state_dict(agent_state_dict["critic_state_dict"])
        self.critic_target.load_state_dict(agent_state_dict["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(agent_state_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(agent_state_dict["critic_optimizer_state_dict"])

        # self.memory = agent_state_dict["memory"]
        # self.steps_done = len(self.memory)

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
                # "memory": self.memory,
            },
        )
