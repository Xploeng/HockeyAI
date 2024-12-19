import collections
import math
import random
import sys
import torch

from .agent import Agent


sys.path.append("src/")
from utils.replay import Transition


class DeepQLearning(Agent):
    def __init__(
        self,
        env,
        memory,
        policy_net,
        target_net = None,
        optimizer = None,
        criterion = None,
        device = "cuda:0",
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 1000,
        **_,
    ):
        super().__init__()
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.env = env
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

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
        transitions = self.memory.sample(batch_size)

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
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

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
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * tau + target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def load_state_dict(
        self,
        agent_state_dict,
        optimizer_state_dict=None,
        episode=None,
        **_,
    ):
        if self.target_net is not None:
            self.target_net.load_state_dict(agent_state_dict["network_state_dict"])

        self.policy_net.load_state_dict(agent_state_dict["network_state_dict"])

        self.memory = agent_state_dict["memory"]
        self.steps_done = len(self.memory)
        if self.optimizer is not None and optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

    def state_dict(self) -> collections.OrderedDict:
        return collections.OrderedDict(
            {"network_state_dict": self.policy_net.state_dict(), "memory": self.memory},
        )
