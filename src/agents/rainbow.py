import sys
import torch

from .dql import DeepQLearning


sys.path.append("src/")
import time
from utils.replay import Transition


class Rainbow(DeepQLearning):
    def __init__(
        self,
        env,
        memory,
        policy_net,
        episodes=0,
        target_net=None,
        optimizer=None,
        criterion=None,
        device="cuda:0",
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        # PER parameters
        beta=0.6,
        prior_eps=1e-6,
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

        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        self.episodes = episodes
        self.v_min = 0.0
        self.v_max = 200.0
        self.atom_size = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

    def select_action(self, state):
        self.steps_done += 1
        return self.policy_net(state).max(1).indices.view(1, 1)

    def _compute_categorical_loss(self, samples, batch_size, gamma):
        transitions, weights, indices = (
            samples["transitions"],
            samples["weights"],
            samples["indices"],
        )
        weights = torch.tensor(
            weights,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Transpose the batch
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

        # Categorical loss
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.policy_net(non_final_next_states).argmax(1)
            next_dist = self.target_net.dist(non_final_next_states)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = reward_batch[non_final_mask].unsqueeze(-1) + gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (batch_size - 1) * self.atom_size, batch_size)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device, dtype=torch.double)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.policy_net.dist(state_batch)
        log_p = torch.log(dist[range(batch_size), action_batch.squeeze(1)])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        loss = torch.mean(elementwise_loss * weights)

        return loss, elementwise_loss, indices

    def _compute_loss(self, samples, batch_size, gamma):
        transitions, weights, indices = (
            samples["transitions"],
            samples["weights"],
            samples["indices"],
        )
        weights = torch.tensor(
            weights,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Transpose the batch
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

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(
                    non_final_next_states,
                )
                .gather(
                    1,
                    self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True),
                )
                .squeeze()
            )

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute loss
        elementwise_loss = self.criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )
        loss: torch.Tensor = torch.mean(elementwise_loss * weights)

        return loss, elementwise_loss, indices

    def optimize(self, batch_size, gamma, tau, episode, **_):
        # Timing the different components

        # Increase beta
        fraction = min(episode / self.episodes, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        # Sample from memory
        if len(self.memory) < batch_size:
            return
        samples = self.memory.sample(batch_size, self.beta)

        loss, elementwise_loss, indices = self._compute_categorical_loss(samples, batch_size, gamma)

        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Update target network
        self.update_target(tau=tau)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.policy_net.reset_noise()
        self.target_net.reset_noise()
