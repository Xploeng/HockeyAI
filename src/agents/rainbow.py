import sys
import torch

from .dql import DeepQLearning


sys.path.append("src/")
from utils.replay import Transition
import time


class Rainbow(DeepQLearning):
    def __init__(
        self,
        env,
        memory,
        policy_net,
        episodes,
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

    def optimize(self, batch_size, gamma, tau, episode, **_):
        # Timing the different components
        start_time = time.time()

        # Sample from memory
        if len(self.memory) < batch_size:
            return
        samples = self.memory.sample(batch_size, self.beta)
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

        sample_time = time.time()

        # Increase beta
        fraction = min(episode / self.episodes, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

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

        transpose_time = time.time()

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

        compute_q_time = time.time()

        # Compute loss
        elementwise_loss = self.criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )
        loss = torch.mean(elementwise_loss * weights)

        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        compute_loss_time = time.time()

        # Update target network
        self.update_target(tau=tau)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        update_time = time.time()

        # Write timings to file
        with open(
            "/home/jules/Documents/Uni/3.Semester/RL/HockeyAI/timings.txt",
            "a",
        ) as f:
            f.write(f"Sample time: {sample_time - start_time}\n")
            f.write(f"Transpose time: {transpose_time - sample_time}\n")
            f.write(f"Compute Q time: {compute_q_time - transpose_time}\n")
            f.write(f"Compute loss time: {compute_loss_time - compute_q_time}\n")
            f.write(f"Update time: {update_time - compute_loss_time}\n")
            f.write(f"Total time: {update_time - start_time}\n\n")
