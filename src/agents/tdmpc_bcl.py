import collections
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append("src/")

from gymnasium.spaces import Box
from PIL import Image
from torch.distributions import Normal

from agents import Agent
from utils.networks import WorldModel, Actor, ValueFunction
from utils.replay import ReplayMemory, Transition

class TDMPC_BCL(Agent):
    def __init__(
        self,
        env,
        memory,
        training,
        opponent=None,
        hidden_size=256,
        latent_dim=50,
        action_dim=None,
        horizon=5,
        n_samples=512,
        mixture_coef=0.05,
        temperature=0.5,  # Initial temperature
        min_buffer_size=1000,
        reward_weight=0.1,
        gamma=0.99,
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
        self.horizon = horizon
        self.n_samples = n_samples
        self.mixture_coef = mixture_coef
        self.initial_temperature = temperature  # Store initial value
        self.temperature = temperature        # Current temperature, will decay
        self.temperature_decay = 0.99         # Decay factor per episode
        self.min_buffer_size = min_buffer_size
        self.reward_weight = reward_weight
        self.gamma = gamma
        self.alpha = 1.0                      # Weight for behavioral cloning loss

        self.memory_cfg = memory
        self.memory = ReplayMemory(self.memory_cfg.capacity)

        self.hockey = True if opponent is not None or self.mode == "opponent" else False
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0] if action_dim is None else action_dim

        if self.hockey:
            out_actions = int(self.num_actions / 2)
            self.agent_action_space = Box(
                low=env.action_space.low[:out_actions],
                high=env.action_space.high[:out_actions],
                dtype=env.action_space.dtype,
            )
            self.action_dim = out_actions
        else:
            self.agent_action_space = env.action_space
            self.action_dim = self.num_actions

        # Convert action bounds to tensors
        self.action_low = torch.FloatTensor(self.agent_action_space.low).to(device)
        self.action_high = torch.FloatTensor(self.agent_action_space.high).to(device)

        # Networks
        self.world_model = WorldModel(
            obs_dim=self.num_states,
            action_dim=self.action_dim,
            hidden_dim=hidden_size,
            latent_dim=latent_dim
        ).to(device)
        
        self.actor = Actor(
            self.num_states,
            hidden_size,
            self.action_dim
        ).to(device)
        
        self.value_function = ValueFunction(
            self.num_states,
            hidden_size
        ).to(device)

        # Optimizers with gradient clipping
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=training.world_model_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=training.actor_lr)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=training.value_lr)
        
        self.max_grad_norm = 1.0  # For gradient clipping
        self.losses = []          # To track losses

    def select_action(self, state, evaluate=False):
        """Select action using MPC planning when training, or actor for evaluation"""
        assert state.shape[-1] == self.num_states, f"Expected state dim {self.num_states}, got {state.shape}"
        state = torch.FloatTensor(state).to(self.device)
        
        if evaluate or self.mode != "train":
            with torch.no_grad():
                action = self.actor(state)
                return action.cpu().numpy()

        # MPC Planning with batched operations
        state = state.repeat(self.n_samples, 1)
        actions = torch.zeros(self.n_samples, self.horizon, self.action_dim).to(self.device)
        
        with torch.no_grad():
            current_state = state
            for t in range(self.horizon):
                actor_actions = self.actor(current_state)
                noise = torch.randn_like(actor_actions) * self.temperature
                actions[:, t] = torch.clamp(actor_actions + noise, self.action_low, self.action_high)
                current_state = self.world_model.predict_next_state(current_state, actions[:, t])

        # Evaluate trajectories
        values = self.evaluate_trajectories(current_state, actions)
        best_idx = values.argmax()
        
        return actions[best_idx, 0].cpu().numpy()

    def batched_select_action(self, states):
        """Batched MPC planning to select best initial actions for multiple states"""
        B = states.size(0)  # Batch size
        total_samples = B * self.n_samples
        states_expanded = states.unsqueeze(1).repeat(1, self.n_samples, 1).view(total_samples, -1)
        actions = torch.zeros(B, self.n_samples, self.horizon, self.action_dim).to(self.device)
        
        with torch.no_grad():
            current_states = states_expanded.clone()
            for t in range(self.horizon):
                actor_actions = self.actor(current_states).view(B, self.n_samples, self.action_dim)
                noise = torch.randn_like(actor_actions) * self.temperature
                actions[:, :, t, :] = torch.clamp(actor_actions + noise, self.action_low, self.action_high)
                next_states = self.world_model.predict_next_state(
                    current_states, actions[:, :, t, :].view(total_samples, -1)
                )
                current_states = next_states
        
        # Evaluate trajectories
        values = self.batched_evaluate_trajectories(states, actions)
        best_indices = values.view(B, self.n_samples).argmax(dim=1)
        best_actions = actions[torch.arange(B), best_indices, 0, :]
        return best_actions

    def evaluate_trajectories(self, state, actions):
        """Evaluate trajectories for a single state"""
        total_value = torch.zeros(self.n_samples).to(self.device)
        current_state = state
        
        for t in range(self.horizon):
            next_state = self.world_model.predict_next_state(current_state, actions[:, t])
            reward = self.world_model.predict_reward(current_state, actions[:, t], next_state)
            value = self.value_function(next_state).squeeze(-1)
            
            if reward.dim() > 1:
                reward = reward.squeeze(-1)
            
            total_value += (self.gamma ** t) * reward
            current_state = next_state
        
        total_value += (self.gamma ** self.horizon) * self.value_function(current_state).squeeze(-1)
        return total_value

    def batched_evaluate_trajectories(self, states, actions):
        """Evaluate trajectories for a batch of states"""
        B, N, H, _ = actions.shape
        total_samples = B * N
        total_value = torch.zeros(total_samples).to(self.device)
        current_states = states.unsqueeze(1).repeat(1, N, 1).view(total_samples, -1)
        
        for t in range(H):
            actions_t = actions[:, :, t, :].view(total_samples, -1)
            next_states = self.world_model.predict_next_state(current_states, actions_t)
            rewards = self.world_model.predict_reward(current_states, actions_t, next_states)
            if rewards.dim() > 1:
                rewards = rewards.squeeze(-1)
            
            total_value += (self.gamma ** t) * rewards
            current_states = next_states
        
        total_value += (self.gamma ** H) * self.value_function(current_states).squeeze(-1)
        return total_value

    def record(self, state, action, next_state, reward, done):
        """Record transition in replay memory, including opponent action if available"""
        assert len(state) == self.num_states, f"State dim mismatch: expected {self.num_states}, got {len(state)}"
        assert len(next_state) == self.num_states, f"Next state dim mismatch: expected {self.num_states}, got {len(next_state)}"
        
        # For hockey environment, action contains both agent and opponent actions
        if self.hockey and len(action) == self.num_actions:
            agent_action = action[:self.action_dim]
            opp_action = action[self.action_dim:]
            self.memory.push(state, (agent_action, opp_action), next_state, reward, done)
        else:
            self.memory.push(state, action, next_state, reward, done)

    def optimize(self, batch_size):
        """Optimize world model, actor, and value function with behavioral cloning"""
        if len(self.memory) < max(batch_size, self.min_buffer_size):
            return

        transitions = self.memory.sample(batch_size)["transitions"]
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(np.array(batch.done)).to(self.device).unsqueeze(1)

        # Handle actions differently for hockey environment
        if self.hockey:
            # Unpack agent and opponent actions
            agent_actions, opp_actions = zip(*batch.action)
            actions = torch.FloatTensor(np.array(agent_actions)).to(self.device)
            opp_actions = torch.FloatTensor(np.array(opp_actions)).to(self.device)
            
            # Concatenate actions for world model prediction
            combined_actions = torch.cat([actions, opp_actions], dim=1)
            pred_next_states = self.world_model.predict_next_state(states, combined_actions)
        else:
            actions = torch.FloatTensor(np.array(batch.action)).to(self.device)
            pred_next_states = self.world_model.predict_next_state(states, actions)

        # Update World Model
        pred_rewards = self.world_model.predict_reward(states, actions, next_states)
        state_loss = torch.nn.functional.mse_loss(pred_next_states, next_states)
        reward_loss = torch.nn.functional.mse_loss(pred_rewards, rewards)
        world_model_loss = state_loss + self.reward_weight * reward_loss
        
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.max_grad_norm)
        self.world_model_optimizer.step()

        # Update Value Function
        with torch.no_grad():
            next_values = self.value_function(next_states)
            target_values = rewards + (1 - dones) * self.gamma * next_values

        current_values = self.value_function(states)
        value_loss = torch.nn.functional.mse_loss(current_values, target_values)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        # Update Actor with Behavioral Cloning
        actor_actions = self.actor(states)
        next_states_pred = self.world_model.predict_next_state(states, actor_actions)
        pred_rewards = self.world_model.predict_reward(states, actor_actions, next_states_pred)
        pred_values = self.value_function(next_states_pred)
        actor_loss = -(pred_rewards + self.gamma * pred_values).mean()

        # Behavioral cloning loss
        mpc_actions = self.batched_select_action(states)
        bc_loss = torch.nn.functional.mse_loss(actor_actions, mpc_actions)
        total_actor_loss = actor_loss + self.alpha * bc_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.losses.append(world_model_loss.item())

    def train_episode(self) -> float:
        """Train for one episode and return total reward, with temperature decay"""
        state, _ = self.env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = self.select_action(state)
            # Get opponent state and action
            opp_state = self.env.obs_agent_two() if self.hockey else None
            action_opp = self.opponent.act(opp_state) if self.hockey else None
            next_state, done = self.step(state, action, action_opp)
            total_reward += self.last_reward.item()
            self.optimize(self.batch_size)
            state = next_state
        
        # Decay temperature after each episode
        self.temperature = max(0.01, self.temperature * self.temperature_decay)
        return total_reward

    def step(self, state, action, action_opp=None):
        """Execute one step in the environment"""
        full_action = np.hstack([action, action_opp]) if self.hockey else action
        next_state, reward, terminated, truncated, _ = self.env.step(full_action)
        self.last_reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done = terminated or truncated

        if terminated:
            next_state = np.zeros(self.num_states)

        # Store both agent and opponent actions when in hockey environment
        memory_action = full_action if self.hockey else action
        self.record(state, memory_action, next_state, self.last_reward, done)
        return next_state, done

    def evaluate_episode(self, render: bool = True) -> tuple[list[Image.Image], dict, float]:
        """Evaluate one episode, returning frames, info, and total reward"""
        state, info = self.env.reset()
        done = False
        frames = []
        total_reward = 0.0

        while not done:
            if render:
                render_kwargs = {"mode": "rgb_array"} if self.hockey else {}
                frame = self.env.render(**render_kwargs)
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            action = self.select_action(state, evaluate=True)
            # Get opponent state and action
            opp_state = self.env.obs_agent_two() if self.hockey else None
            action_opp = self.opponent.act(opp_state) if self.hockey else None
            full_action = np.hstack([action, action_opp]) if self.hockey else action
            
            next_state, reward, terminated, truncated, info = self.env.step(full_action)
            total_reward += reward
            done = terminated or truncated
            
            # Record only the agent's action, not the full action
            memory_action = action
            self.record(state, memory_action, next_state, torch.tensor([reward], device=self.device), done)
            state = next_state

        return frames, info

    def load_state_dict(self, agent_state_dict, episode=None, **_):
        self.world_model.load_state_dict(agent_state_dict["world_model_state_dict"])
        self.actor.load_state_dict(agent_state_dict["actor_state_dict"])
        self.value_function.load_state_dict(agent_state_dict["value_function_state_dict"])
        
        if self.mode == "train":
            self.world_model_optimizer.load_state_dict(agent_state_dict["world_model_optimizer_state_dict"])
            self.actor_optimizer.load_state_dict(agent_state_dict["actor_optimizer_state_dict"])
            self.value_optimizer.load_state_dict(agent_state_dict["value_optimizer_state_dict"])
            self.memory = agent_state_dict["memory"]

    def state_dict(self):
        return collections.OrderedDict({
            "world_model_state_dict": self.world_model.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "value_function_state_dict": self.value_function.state_dict(),
            "world_model_optimizer_state_dict": self.world_model_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "memory": self.memory,
        })