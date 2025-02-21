import collections
import sys
import hydra
import numpy as np
import torch

from gymnasium import spaces
from PIL import Image

from .agent import Agent


sys.path.append("src/")
from utils.replay import Transition


class RainbowNAF(Agent):
    def __init__(
        self,
        env,
        opponent,
        memory,
        network,
        training,
        n_memory=None,
        device=torch.device("cuda:0"),
        mode='train',
        **_,
    ):
        super().__init__()
        torch.set_default_dtype(torch.float32)

        self.env = env
        self.opponent = opponent
        self.device = device
        self.training = training
        self.mode = mode

        if opponent is not None or self.mode == 'opponent':
            self.hockey = True
        else:
            self.hockey = False

        self.n_actions = env.action_space.shape[0]
        if self.hockey: # as the action space is defined for both players
            self.n_actions = int(self.n_actions/2)
        
        n_observations = env.observation_space.shape[0]

        # Categorical DQN
        self.policy_net = hydra.utils.instantiate(
            config=network,
            n_observations=n_observations,
            n_actions=self.n_actions,
        ).to(self.device)
        self.target_net = hydra.utils.instantiate(
            config=network,
            n_observations=n_observations,
            n_actions=self.n_actions,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = hydra.utils.instantiate(
            config=training.optimizer,
            params=self.policy_net.parameters(),
        )
        self.criterion = hydra.utils.instantiate(config=training.criterion)

        # PER
        self.memory = hydra.utils.instantiate(config=memory)
        self.beta = memory.beta
        self.priority_eps = memory.priority_eps

        # N step memory
        n_step_config = n_memory
        self.use_n_step = n_step_config is not None and n_step_config.n_steps > 1
        if self.use_n_step:
            self.n_step = n_step_config.n_steps
            self.memory_n = hydra.utils.instantiate(config=n_step_config)

        self.episode = 0
        self.episodes = training.episodes
        self.rewards = []

    def select_action(self, state):
        self.steps_done += 1
        self.policy_net.eval()
        with torch.no_grad():
            action, _, _ = self.policy_net(state)
        self.policy_net.train()
        return action

    def record(self, state, action, next_state, reward, done):
        return self.memory.push(state, action, next_state, reward, done)

    def update_target(self, tau):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def train_episode(self) -> None:
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        done = False
        self.episode += 1

        # Increase beta
        fraction = min(self.episode / self.episodes, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
        self.reward = 0

        while not done:
            action = self.select_action(state)

            opp_action = self.opponent.act(state) if self.hockey else None
            next_state, reward, done = self.step(state, action, opp_action)
            
            self.reward += reward

            self.optimize(**self.training)

            state = next_state
        
    def step(self, state, action, action_opp=None):
        act = action.squeeze().cpu().numpy()
        if self.hockey:
            act = np.hstack([act, action_opp])
        

        next_state, reward, terminated, truncated, info = self.env.step(act)
        if self.hockey:
            reward += info["reward_touch_puck"] + info["reward_puck_direction"]
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done = terminated or truncated

        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.use_n_step:
            one_step_transition = self.memory_n.push(*Transition(state, action, next_state, reward, done))
        else:
            one_step_transition = Transition(state, action, next_state, reward, done)

        if one_step_transition is not None:
            self.memory.push(*one_step_transition)

        return next_state, reward, done

    def _compute_loss(self, samples, batch_size, gamma):
        transitions = samples["transitions"]
        # Transpose the batch
        batch = Transition(*zip(*transitions))
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        assert len(non_final_next_states) == batch_size, "Non final next states should have the same size as the batch"

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # get the V value of the next state from target network
        with torch.no_grad():
            _, _, V_next = self.target_net(non_final_next_states)
            
        # Compute Q targets
        expected_Q = reward_batch.unsqueeze(-1) + gamma * V_next
        
        # Compute expected Q values
        _, Q_values, _ = self.policy_net(state_batch, action_batch)

        # Q-loss
        elementwise_loss = torch.nn.functional.mse_loss(Q_values, expected_Q, reduction='none')
        
        # TD-error
        td_error = torch.abs(Q_values - expected_Q).detach().cpu().numpy()
                        
        return elementwise_loss, td_error

    def optimize(self, batch_size, gamma, tau, **_):
        # Sample from memory
        if len(self.memory) < batch_size:
            return
        samples = self.memory.sample(batch_size, self.beta)
        weights, indices = samples["weights"], samples["indices"]
        weights = torch.tensor(
            weights,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Calculate one-step loss and importance weights
        elementwise_loss, td_error = self._compute_loss(samples, batch_size, gamma)

        loss = torch.mean(elementwise_loss * weights)

        # N-step loss
        if self.use_n_step:
            gamma = gamma**self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            n_elementwise_loss, td_error = self._compute_loss(samples, batch_size, gamma)
            elementwise_loss += n_elementwise_loss
            td_error += td_error

            loss = torch.mean(elementwise_loss * weights)

        self.losses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Update target network
        self.update_target(tau=tau)

        # PER: update priorities
        new_priorities = td_error + self.priority_eps
        self.memory.update_priorities(indices, new_priorities)

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def evaluate_episode(self, render: bool = True) -> tuple[list[Image.Image], dict]:
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        done = False
        frames = []

        while not done:
            # Render the environment and save the frames
            frame = self.env.render(mode="rgb_array") if render else None

            if frame is not None:
                frames.append(Image.fromarray(frame))

            # Action selection and recording the transition
            action = self.select_action(state)
            act = action.squeeze().cpu().numpy()
            opp_action = self.opponent.act(state) if self.hockey else None
            if self.hockey:
                act = np.hstack([act, opp_action])

            next_state, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated

            self.record(state, action, next_state, reward, done)

            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                state = next_state

        return frames, info

    def load_state_dict(
        self,
        agent_state_dict,
        episode=None,
        **_,
    ):
        self.target_net.load_state_dict(agent_state_dict["network_state_dict"])
        self.policy_net.load_state_dict(agent_state_dict["network_state_dict"])
        if self.mode == 'train':
            self.optimizer.load_state_dict(agent_state_dict["optimizer_state_dict"])

            self.memory = agent_state_dict["memory"]
            self.memory_n = agent_state_dict["memory_n"]
            self.steps_done = episode

    def state_dict(self) -> collections.OrderedDict:
        return collections.OrderedDict(
            {"network_state_dict": self.policy_net.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(),
             "memory": self.memory,
             "memory_n": self.memory_n,},
        )
