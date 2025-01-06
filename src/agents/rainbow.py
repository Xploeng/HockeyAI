import collections
import math
import random
import sys
import torch

from .DQL import DeepQLearning


sys.path.append("src/")
from utils.replay import Transition


class Rainbow(DeepQLearning):
    def __init__(
        self,
        env,
        memory,
        policy_net,
        target_net=None,
        optimizer=None,
        criterion=None,
        device="cuda:0",
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        **_,
    ):
        super().__init__(
            policy_net=policy_net,
            target_net=target_net,
            memory=memory,
            env=env,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
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
                self.target_net(non_final_next_states).gather(
                    1,
                    self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True),
                ),
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
