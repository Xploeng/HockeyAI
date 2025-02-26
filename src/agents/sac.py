import collections
import sys
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim

from gymnasium.spaces import Box
from icecream import ic
from PIL import Image

from .agent import Agent


sys.path.append("src/")
from utils.networks import SACActor, SACCritic
from utils.replay import ReplayMemory, Transition


torch.autograd.set_detect_anomaly(True)


class SAC(Agent):
    def __init__(
        self,
        env,
        memory,
        training,
        opponent,
        hidden_size=400,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
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
        self.alpha = alpha

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
            self.actor = SACActor(self.num_states, hidden_size, out_actions).to(self.device)
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_learning_rate)

            self.critic = SACCritic(self.num_states, self.num_actions, hidden_size).to(self.device)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_learning_rate)

            self.critic_target = SACCritic(self.num_states, self.num_actions, hidden_size).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            self.agent_action_space = env.action_space
            self.actor = SACActor(self.num_states, hidden_size, self.num_actions).to(self.device)
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_learning_rate)

            self.critic = SACCritic(self.num_states, self.num_actions, hidden_size).to(self.device)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_learning_rate)

            self.critic_target = SACCritic(self.num_states, self.num_actions, hidden_size).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        # temperature parameter for entropy regularization
        self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=actor_learning_rate)

        # target entropy (-dim(action))
        if self.hockey:
            self.target_entropy = -out_actions
        else:
            self.target_entropy = -self.num_actions

        self.steps_done = 0

    def select_action(self, state, deterministic=False, action_space=None):
        """Select an action using the actor network."""
        # during evaluation, use deterministic actions!
        action = self.actor.select_action(state, deterministic)
        if action_space is not None:
            return np.clip(action, action_space.low, action_space.high)
        return action

    def record(self, state, action, next_state, opp_next_state, reward, done):
        """Record the transition in the replay memory."""
        self.memory.push(state, action, next_state, opp_next_state, reward, done)

    def optimize(self, batch_size):
        """Optimize the actor and critic networks using the sampled batch."""
        if len(self.memory) < batch_size:
            return
        try:
            transitions = self.memory.sample(batch_size)["transitions"]
        except ValueError as e:
            print(e)
            return
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.FloatTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        opp_next_states = torch.FloatTensor(np.array(batch.opp_next_state)).to(self.device) if self.hockey else None
        dones = torch.FloatTensor(np.array(batch.done)).to(self.device).unsqueeze(1)

        # Critic Update
        with torch.no_grad():
            # Sample next action from actor (and compute log_prob)
            next_agent_action, next_log_prob, _ = self.actor.sample(next_states)
            if self.hockey:
                # Get opponent actions for next states
                if self.opponent.opp_type == "basic":
                    next_opponent_actions = self._batch_opponent_actions(opp_next_states)
                else:
                    next_opponent_actions = self.opponent.act(next_states.unsqueeze(0))
                    next_opponent_actions = torch.tensor(opp_next_states).to(self.device).squeeze(0)
                next_joint_action = torch.cat([next_agent_action, next_opponent_actions], dim=1)
            else:
                next_joint_action = next_agent_action

            # Compute target Q using target critic
            # min of both target Q networks
            # target modulated by entropy regularization
            target_q1, target_q2 = self.critic_target(next_states, next_joint_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.gamma * (target_q - torch.exp(self.log_alpha) * next_log_prob) * (1 - dones)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        agent_action, log_prob, _ = self.actor.sample(states)
        if self.hockey:
            if self.opponent.opp_type == "basic":
                opponent_actions = self._batch_opponent_actions(opp_next_states)
            else:
                opponent_actions = self.opponent.act(opp_next_states)
                opponent_actions = torch.tensor(opponent_actions).to(self.device).squeeze(0)
            joint_actions = torch.cat([agent_action, opponent_actions], dim=1)
        else:
            joint_actions = agent_action

        actor_loss = (torch.exp(self.log_alpha) * log_prob - self.critic.min_q(states, joint_actions)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature (alpha) Update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft Update Target Critic
        self._soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def _soft_update(self, target, source, tau):
        """Soft update model parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _batch_opponent_actions(self, states):
        """Batch opponent actions for hockey env."""
        actions = []
        # Process each state sequentially (or parallelize if opponent.act supports batching)
        for state in states:
            state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
            actions.append(self.opponent.act(state_np))
        return torch.FloatTensor(np.array(actions)).to(states.device)

    def train_episode(self) -> None:
        """Train the agent for a single episode."""
        state, _ = self.env.reset()
        done = False
        critic_losses, actor_losses, alpha_losses = [], [], []
        while not done:
            action = self.select_action(state)
            # When interacting with the environment, the opponent acts as before
            opp_state = self.env.obs_agent_two() if self.hockey else None
            action_opp = self.opponent.act(opp_state) if self.hockey else None
            action_opp = action_opp.cpu().numpy() if isinstance(action_opp, torch.Tensor) else action_opp
            next_state, done = self.step(state, action, action_opp)
            losses = self.optimize(self.batch_size)
            if losses is not None:
                critic_loss, actor_loss, alpha_loss = losses
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                alpha_losses.append(alpha_loss)
            state = next_state
            self.steps_done += 1
        return critic_losses, actor_losses, alpha_losses

    def step(self, state, action, action_opp=None):
        """Take a step in the environment using the generated action."""
        # Stack actions for hockey env
        action = np.hstack([action, action_opp]) if self.hockey else action
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        opp_next_state = self.env.obs_agent_two() if self.hockey else None
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated
        if terminated:
            next_state = np.zeros(self.num_states)
        self.record(state, action, next_state, opp_next_state, reward, done)
        return next_state, done

    def evaluate_episode(self, render: bool = True) -> tuple[list[Image.Image], dict]:
        """Evaluate the agent for a single episode. Deterministic actions are used."""
        state, info = self.env.reset()
        done = False
        frames = []
        while not done:
            if render:
                render_kwargs = {"mode": "rgb_array"} if self.hockey else {}
                frame = self.env.render(**render_kwargs)
                if frame is not None:
                    frames.append(Image.fromarray(frame))
            opp_state = self.env.obs_agent_two() if self.hockey else None
            action_opp = self.opponent.act(opp_state) if self.hockey else None
            action = self.select_action(state, deterministic=True, action_space=self.agent_action_space)
            action = np.hstack([action, action_opp]) if self.hockey else action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            opp_next_state = self.env.obs_agent_two() if self.hockey else None
            done = terminated or truncated
            self.record(state.tolist(), action, next_state, opp_next_state, reward, done)
            state = next_state
        return frames, info

    def load_state_dict(self, agent_state_dict, episode=None, **_):
        """Load the agent's state from a checkpoint to allow for continuos training."""
        self.actor.load_state_dict(agent_state_dict["actor_state_dict"])
        self.critic.load_state_dict(agent_state_dict["critic_state_dict"])
        self.critic_target.load_state_dict(agent_state_dict["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(agent_state_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(agent_state_dict["critic_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(agent_state_dict["alpha_optimizer_state_dict"])

    def state_dict(self):
        """Return the agent's state as a dictionary for checkpoint saving."""
        return collections.OrderedDict(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            },
        )
