#! bin/env/python3

import argparse
import glob
import json
import os
import sys
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from matplotlib.animation import PillowWriter
from omegaconf import DictConfig
from tqdm import tqdm


sys.path.append("src/")
from agents import Agent  # noqa: E402, F403, I001
from utils import DiscreteActionWrapper, ReplayMemory  # noqa: E402, F403, I001


def evaluate_model(cfg: DictConfig) -> None:
    """
    Evaluates a single model for a given configuration.

    Args:
        cfg (DictConfig): The hydra configuration of the agent
        file_path (str): The destination path for the resulting plots

    Returns:
        None
    """
    device = torch.device(cfg.device)
    if cfg.verbose:
        print("\n\nInitialize environment and model")
        print(f"\nDevice: {device}")

    # Initialize the environment
    # video_folder = os.path.join("src/outputs", cfg.agent.name, "videos")
    # os.makedirs(video_folder, exist_ok=True)
    env = gym.make(cfg.env) # , render_mode="rgb_array")
    # env = RecordVideo(env, video_folder=video_folder, name_prefix="eval")
    env = RecordEpisodeStatistics(env)

    memory: ReplayMemory = hydra.utils.instantiate(config=cfg.memory)

    if isinstance(env.action_space, spaces.Box):
        n_actions = cfg.bins
        env = DiscreteActionWrapper(env, bins=cfg.bins)
    elif isinstance(env.action_space, spaces.Discrete):
        n_actions = env.action_space.n

    if isinstance(env.observation_space, spaces.Box):
        n_oberservations = env.observation_space.shape[0]
    else:
        state, info = env.reset()
        n_oberservations = len(state)

    # Initialize the agents network
    policy_net = hydra.utils.instantiate(
        config=cfg.network,
        n_observations=n_oberservations,
        n_actions=n_actions,
    ).to(device=device)

    # Set up model
    agent: Agent = hydra.utils.instantiate(
        config=cfg.agent,
        env=env,
        memory=memory,
        policy_net=policy_net,
        n_actions=n_actions,
        n_observations=n_oberservations,
        device=device,
    )
    # Load checkpoint from file
    checkpoint_path = os.path.join(
        "src/outputs",
        cfg.agent.name,
        "checkpoints",
        f"{cfg.agent.name}_last.ckpt",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(**checkpoint)

    # Evaluate the model
    print(f"\nEvaluating the model on {cfg.testing.episodes} episodes.")
    episode_stats = {}
    for episode in range(cfg.testing.episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        rewards = []
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                state = next_state
            else:
                return_code = "termination" if terminated else "truncation"
                print(f"Episode finished due to {return_code}.")
                if "episode" in info:
                    stats = info["episode"]
                    stats["rewards"] = rewards
                    episode_stats.update({episode:stats})

    # Save episode statistics
    episode_stats_dir = os.path.join("src/outputs", cfg.agent.name, "episode_statistics")
    os.makedirs(episode_stats_dir, exist_ok=True)
    stats_file_path = os.path.join(episode_stats_dir, f"episode_statistics_{cfg.agent.name}.json")
    with open(stats_file_path, "w") as stats_file:
        json.dump(episode_stats, stats_file, indent=4)
    if cfg.verbose:
        print(f"Episode statistics saved to {stats_file_path}")


def plot_rewards(agent_out_dir, n_episodes=1, show=True):
    """
    Saves a plot of the rewards.

    Args:
        rewards (list): List of rewards obtained during evaluation.
        file_path (str): The destination path for the resulting plot.

    Returns:
        None
    """
    file_path = glob.glob(agent_out_dir + "/episode_statistics/*.json")[0]
    episode_stats = load_episode_statistics(file_path, n_episodes)

    for episode, stats in episode_stats.items():
        figure_name = f"rewards_episode_{episode}"
        rewards = stats["rewards"]

        fig, ax = plt.subplots(1, 1, figsize=[7, 3])
        ax.plot(range(len(rewards)), rewards, label="Reward")
        ax.set_xlabel("Time")
        ax.set_ylabel("Reward")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(agent_out_dir, "figures", f"{figure_name}.png"), bbox_inches="tight")
        if show:
            plt.show()

def load_episode_statistics(file_path, n_episodes=1) -> dict:
    with open(file_path) as file:
        data = json.load(file)
    return {int(k): data[k] for k in list(data.keys())[:n_episodes]}

def animate_episode(env, model, device, file_path):
    pass  # TODO


def run_evaluations(
    configuration_dir_list: str,
    device: str,
    silent: bool = False,
) -> None:
    """
    Evaluates a model with the given configuration.

    Args:
        configuration_dir_list (str): A list of hydra configuration directories to the models for evaluation.
        device (str): The device where the evaluations are performed.

    Returns:
        None
    """

    for configuration_dir in configuration_dir_list:
        # Initialize the hydra configurations for this forecast
        config_path = os.path.join("..", configuration_dir, ".hydra")
        # print(f"\nEvaluating models in {config_path}")
        with hydra.initialize(version_base=None, config_path=config_path):
            cfg = hydra.compose(config_name="config")
            cfg.device = device
            cfg.verbose = not silent

        if cfg.seed:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        evaluate_model(cfg=cfg)

        # Visualize the results
        agent_output_dir = os.path.join("src/outputs", cfg.agent.name)
        figures_dir = os.path.join(agent_output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        plot_rewards(
            agent_output_dir,
            n_episodes=cfg.testing.episodes,
            show=cfg.testing.show_figures,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with a given configuration. Particular properties of the configuration can be "
        "overwritten, as listed by the -h flag.",
    )
    parser.add_argument(
        "-c",
        "--configuration-dir-list",
        nargs="*",
        default=["configs"],
        help="List of directories where the configuration files of all models to be evaluated lie.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="The device to run the evaluation. Any of ['cpu' (default), 'cuda', 'mpg'].",
    )
    parser.add_argument(
        "-s",
        "--silent",
        default=False,
        help="Silent mode to prevent printing results to console and visualizing plots dynamically.",
    )

    run_args = parser.parse_args()
    run_evaluations(
        configuration_dir_list=run_args.configuration_dir_list,
        device=run_args.device,
        silent=run_args.silent,
    )

    print("Done.")
