import argparse
import os
import sys
import warnings

from dataclasses import asdict
import gymnasium as gym
import hockey
import hydra
import numpy as np
import torch

from gymnasium import spaces
from omegaconf import DictConfig
from tqdm import tqdm


sys.path.append("src/")
from agents import Agent
from utils import DiscreteActionWrapper
from utils.visuals import (
    EpisodeStatistics,
    plot_q_function_all_dims,
    plot_rewards,
    plot_wins_vs_losses,
    save_gif,
    save_json,
)


def initialize_environment(cfg: DictConfig):
    opp = None
    if cfg.env.name == "Hockey-v0":
        env = hockey.hockey_env.HockeyEnv()
        opp = hydra.utils.instantiate(cfg.env.opponent)
    else:
        env = gym.make(cfg.env.name, render_mode="rgb_array")

    # Check if env continuous and agent not continuous -> wrap env
    agent_continuous = cfg.agent.requires_continues_action_space
    if isinstance(env.action_space, spaces.Box) and not agent_continuous:
        env = DiscreteActionWrapper(env, bins=cfg.agent.bins)
    elif isinstance(env.action_space, spaces.Discrete) and agent_continuous:
        raise ValueError(
            f"Agent requires a continuous action space, but {cfg.env} has a discrete action space.",
        )
    return env, opp


def initialize_agent(cfg: DictConfig, env: gym.Env, opponent, device: torch.device, checkpoint_path: str) -> Agent:
    agent_continuous = cfg.agent.requires_continues_action_space
    env_continuous = isinstance(env.action_space, spaces.Box)
    if agent_continuous and not env_continuous:
        raise ValueError("The agent requires a continuous action space, but the environment has a discrete one.")

    agent: Agent = hydra.utils.instantiate(
        config=cfg.agent,
        env=env,
        opponent=opponent,
        device=device,
        recursive=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(**checkpoint)

    return agent


def evaluate_model(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)

    animation_dir = os.path.join("src/outputs", cfg.agent.name, "animations")
    os.makedirs(animation_dir, exist_ok=True)
    episode_stats_dir = os.path.join("src/outputs", cfg.agent.name, "episode_statistics")
    os.makedirs(episode_stats_dir, exist_ok=True)
    figures_dir = os.path.join("src/outputs", cfg.agent.name, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        "src/outputs",
        cfg.agent.name,
        "checkpoints",
        f"{cfg.agent.name}_last.ckpt",
    )

    # Initialize the environment and agent
    env, opp = initialize_environment(cfg)
    agent = initialize_agent(cfg, env, opp, device, checkpoint_path)

    # Fresh recordings (clear training recordings)
    agent.memory.clear()

    # Plot the Q-function for all state dimensions
    # plot_q_function_all_dims(agent, cfg.env, figures_dir)

    wins = 0
    draws = 0
    losses = 0

    # Start the evaluation
    all_episode_stats = {}
    print(f"\nEvaluating the model on {cfg.testing.episodes} episodes.")
    for episode in tqdm(range(cfg.testing.episodes)):
        frames, info = agent.evaluate_episode(render=cfg.testing.render)

        # Keep track of episode statistics and save the animation
        rewards, states = agent.memory.rewards, agent.memory.states
        episode_stats = EpisodeStatistics(episode=episode, rewards=rewards, states=states, info=info)
        all_episode_stats[episode] = asdict(episode_stats)
        agent.memory.clear()

        if cfg.testing.hockey:
            if info["winner"] == 1:
                wins += 1
            elif info["winner"] == 0:
                draws += 1
            else:
                losses += 1
        if cfg.testing.render:
            gif_path = os.path.join(animation_dir, f"episode_{episode}.gif")
            save_gif(frames, gif_path)
            print(f"Episode {episode} animation saved as {gif_path}")

    # Save the episode statistics as json
    stats_file_path = os.path.join(episode_stats_dir, f"episode_statistics_{cfg.agent.name}.json")
    save_json(all_episode_stats, stats_file_path)
    print(f"Episode statistics saved to {stats_file_path}")

    # Plot the results as a pie chart
    plot_wins_vs_losses(wins, draws, losses, figures_dir, show=cfg.testing.show_figures)


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
