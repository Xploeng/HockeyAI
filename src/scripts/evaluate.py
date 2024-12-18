#! bin/env/python3

import argparse
import json
import os
import sys
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from matplotlib.animation import PillowWriter
from omegaconf import DictConfig
from tqdm import tqdm


sys.path.append("src/")
from agents import *  # noqa: E402, F403, I001
from utils import *  # noqa: E402, F403, I001


def evaluate_model(cfg: DictConfig) -> None:
    """
    Evaluates a single model for a given configuration.

    :param cfg: The hydra configuration of the model
    :param file_path: The destination path for the resulting plots
    """

    if cfg.verbose:
        print("\n\nInitialize environment and model")
    device = th.device(cfg.device)

    # Initialize the environment
    env = gym.make(cfg.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Set up model
    model = hydra.utils.instantiate(config=cfg.agent).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.agent._target_} was trained with {trainable_params} trainable parameters")

    # Load checkpoint from file
    checkpoint_path = os.path.join("src/outputs", cfg.agent.name, "checkpoints", f"{cfg.agent.name}_best.ckpt")
    if cfg.verbose:
        print(f"\tRestoring model from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["network_state_dict"])

    # Evaluate the model
    if cfg.verbose:
        print("\nEvaluating the model")
    state, info = env.reset()
    state = th.tensor(state, dtype=th.float32, device=device).unsqueeze(0)
    done = False
    rewards = []

    while not done:
        action = model(state).max(1).indices.view(1, 1)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
        if not done:
            next_state = th.tensor(next_state, dtype=th.float32, device=device).unsqueeze(0)
            state = next_state

    total_reward = sum(rewards)
    print("Total reward:", total_reward)

    # Extract episode statistics
    episode_stats = env.episode_statistics

    # Save episode statistics
    episode_stats_dir = os.path.join("src/outputs", cfg.agent.name, "episode_statistics")
    os.makedirs(episode_stats_dir, exist_ok=True)
    stats_file_path = os.path.join(episode_stats_dir, f"episode_statistics_{cfg.agent.name}.json")
    with open(stats_file_path, "w") as stats_file:
        json.dump(episode_stats, stats_file, indent=4)
    if cfg.verbose:
        print(f"Episode statistics saved to {stats_file_path}")

    return rewards


def reward_plot(rewards, file_path, show=False):
    """
    Saves a plot of the rewards.

    Args:
        rewards (list): List of rewards obtained during evaluation.
        file_path (str): The destination path for the resulting plot.

    Returns:
        None
    """
    print(f"\tSaving reward plot to {file_path}")
    fig, ax = plt.subplots(1, 1, figsize=[8, 2])
    ax.plot(range(len(rewards)), rewards, label="Reward")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "rewards.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        

def animate_episode(env, model, device, file_path):
    pass # TODO


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
            th.manual_seed(cfg.seed)

        file_path = os.path.join("src/outputs", str(cfg.agent.name), "figures")
        os.makedirs(file_path, exist_ok=True)
        rewards = evaluate_model(cfg=cfg)
        reward_plot(rewards, file_path)


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
