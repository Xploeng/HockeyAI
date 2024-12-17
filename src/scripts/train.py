import sys

from itertools import count
import gymnasium as gym
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from IPython import display


sys.path.append("src/")
from agents import *  # noqa: F403
from utils import *  # noqa: F403


@hydra.main(config_path="../configs/", config_name="config", version_base=None)
def run_training(cfg):
    # set up matplotlib
    plt.ion()

    env = gym.make(cfg.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    n_actions = env.action_space.n
    state, info = env.reset()
    n_oberservations = len(state)

    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Initialize the replay buffer
    memory = hydra.utils.instantiate(config=cfg.memory)

    # Initialize the policy and target network
    policy_net = hydra.utils.instantiate(
        config=cfg.network,
        n_observations=n_oberservations,
        n_actions=n_actions,
    ).to(device=device)
    target_net = hydra.utils.instantiate(
        config=cfg.network,
        n_observations=n_oberservations,
        n_actions=n_actions,
    ).to(device=device)
    target_net.load_state_dict(policy_net.state_dict())

    # Initialize the optimizer
    optimizer = hydra.utils.instantiate(
        config=cfg.training.optimizer,
        params=policy_net.parameters(),
    )
    # Initialize the loss function
    criterion = hydra.utils.instantiate(config=cfg.training.criterion)

    # Initialize the agent
    agent = hydra.utils.instantiate(
        config=cfg.agent,
        n_actions=n_actions,
        n_observations=n_oberservations,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        criterion=criterion,
        memory=memory,
        env=env,
        device=device,
    )

    episode_durations = []

    # training loop
    for _ in range(cfg.training.episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize(**cfg.training)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * cfg.training.tau + target_net_state_dict[key] * (
                    1 - cfg.training.tau
                )
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break


def plot_durations(episode_durations, show_result=False):
    is_ipython = "inline" in matplotlib.get_backend()
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


if __name__ == "__main__":
    run_training()
