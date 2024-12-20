import argparse
import glob
import json
import os
import sys
import warnings

from dataclasses import asdict, dataclass
from typing import Any
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
from omegaconf import DictConfig
from PIL import Image


sys.path.append("src/")
from agents import Agent
from utils import DiscreteActionWrapper, ReplayMemory


@dataclass
class EpisodeStatistics:
    episode: int
    rewards: list[float]
    info: dict[str, Any]


def save_json(data: Any, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def save_gif(frames: list[Image.Image], file_path: str, duration: int = 50) -> None:
    frames[0].save(
        file_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def initialize_environment(cfg: DictConfig) -> gym.Env:
    env = gym.make(cfg.env, render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)

    if isinstance(env.action_space, spaces.Box):
        env = DiscreteActionWrapper(env, bins=cfg.bins)

    return env


def initialize_agent(cfg: DictConfig, env: gym.Env, device: torch.device) -> Agent:
    n_actions = env.action_space.n if isinstance(env.action_space, spaces.Discrete) else cfg.bins
    n_observations = env.observation_space.shape[0]

    policy_net = hydra.utils.instantiate(
        config=cfg.network,
        n_observations=n_observations,
        n_actions=n_actions,
    ).to(device=device)

    memory: ReplayMemory = hydra.utils.instantiate(config=cfg.memory)

    agent: Agent = hydra.utils.instantiate(
        config=cfg.agent,
        env=env,
        memory=memory,
        policy_net=policy_net,
        n_actions=n_actions,
        n_observations=n_observations,
        device=device,
    )

    checkpoint_path = os.path.join(
        "src/outputs",
        cfg.agent.name,
        "checkpoints",
        f"{cfg.agent.name}_last.ckpt",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(**checkpoint)

    return agent


def evaluate_model(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)

    env = initialize_environment(cfg)
    agent = initialize_agent(cfg, env, device)

    animation_dir = os.path.join("src/outputs", cfg.agent.name, "animations")
    os.makedirs(animation_dir, exist_ok=True)

    episode_stats_dir = os.path.join("src/outputs", cfg.agent.name, "episode_statistics")
    os.makedirs(episode_stats_dir, exist_ok=True)

    all_episode_stats = {}

    print(f"\nEvaluating the model on {cfg.testing.episodes} episodes.")
    for episode in range(cfg.testing.episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        rewards = []
        frames = []

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                state = next_state

        episode_stats = EpisodeStatistics(episode=episode, rewards=rewards, info=info)
        all_episode_stats[episode] = asdict(episode_stats)

        gif_path = os.path.join(animation_dir, f"episode_{episode}.gif")
        save_gif(frames, gif_path)
        print(f"Episode {episode} animation saved as {gif_path}")

    stats_file_path = os.path.join(episode_stats_dir, f"episode_statistics_{cfg.agent.name}.json")
    save_json(all_episode_stats, stats_file_path)
    print(f"Episode statistics saved to {stats_file_path}")


def plot_rewards(agent_out_dir: str, n_episodes: int = 1, show: bool = True) -> None:
    file_path = glob.glob(agent_out_dir + "/episode_statistics/*.json")[0]

    with open(file_path) as f:
        episode_stats = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 3))

    for episode, stats in list(episode_stats.items())[:n_episodes]:
        rewards = stats["rewards"]
        ax.plot(range(len(rewards)), rewards, label=f"Rewards_{episode}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.tight_layout()

    figure_path = os.path.join(agent_out_dir, "figures", f"rewards_0:{episode}.png")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, bbox_inches="tight")

    if show:
        plt.show()


def run_evaluations(configuration_dir_list: list[str], device: str, silent: bool = False) -> None:
    for configuration_dir in configuration_dir_list:
        config_path = os.path.join("..", configuration_dir, ".hydra")

        with hydra.initialize(version_base=None, config_path=config_path):
            cfg = hydra.compose(config_name="config")
            cfg.device = device
            cfg.verbose = not silent

        if cfg.seed:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        evaluate_model(cfg=cfg)

        agent_output_dir = os.path.join("src/outputs", cfg.agent.name)
        plot_rewards(
            agent_output_dir,
            n_episodes=cfg.testing.episodes,
            show=cfg.testing.show_figures,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with a given configuration.")
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

    args = parser.parse_args()
    run_evaluations(
        configuration_dir_list=args.configuration_dir_list,
        device=args.device,
        silent=args.silent,
    )
    print("Done.")
