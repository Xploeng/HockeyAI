import collections
import math
import random
import sys
import hydra
import torch

from gymnasium import spaces
from PIL import Image

from .agent import Agent


sys.path.append("src/")
from utils.replay import ReplayMemory, Transition


class DeepQLearning(Agent):
    def __init__(
        self,
        env,
        memory,
        network,
        training,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        device=torch.device("cuda:0"),
        bins=100,
        **_,
    ):
        super().__init__()
        self.env = env
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device
        self.memory_cfg = memory
        self.network_cfg = network
        self.training_cfg = training

        self.memory = ReplayMemory(self.memory_cfg.capacity)

        n_actions = env.action_space.n if isinstance(env.action_space, spaces.Discrete) else bins
        n_observations = env.observation_space.shape[0]

        self.policy_net = hydra.utils.instantiate(
            config=self.network_cfg,
            n_observations=n_observations,
            n_actions=n_actions,
        ).to(self.device)

        self.target_net = hydra.utils.instantiate(
            config=self.network_cfg,
            n_observations=n_observations,
            n_actions=n_actions,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = hydra.utils.instantiate(
            config=self.training_cfg.optimizer,
            params=self.policy_net.parameters(),
        )
        self.criterion = hydra.utils.instantiate(config=self.training_cfg.criterion)

    def record(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay,
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize(self, batch_size, gamma, tau, **_):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)["transitions"]

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None],
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = self.criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )

        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.update_target(tau=tau)

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

        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.record(state, action, next_state, reward, done)
            self.optimize(**self.training_cfg)

            state = next_state

    def evaluate_episode(self) -> tuple[list[Image.Image], dict]:
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        done = False
        frames = []

        while not done:

            # Render the environment and save the frames
            frame = self.env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

            # Action selection and recording the transition
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action.item())
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
        if self.target_net is not None:
            self.target_net.load_state_dict(agent_state_dict["network_state_dict"])

        self.policy_net.load_state_dict(agent_state_dict["network_state_dict"])

        self.memory = agent_state_dict["memory"]
        self.steps_done = len(self.memory)
        optimizer_state_dict = agent_state_dict["optimizer_state_dict"]
        if self.optimizer is not None and optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

    def state_dict(self) -> collections.OrderedDict:
        return collections.OrderedDict(
            {"network_state_dict": self.policy_net.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(),
             "memory": self.memory},
        )
